import scipy.signal
import math
import torch
import torch.nn as nn
import numpy as np
import trace
import torchaudio.functional as F
import os
import librosa
#import time

#Nyquist rate (fs/2)
nyq = 24000
fs = nyq*2

torch.set_default_dtype(torch.float32)

#TIME=True

class Renderer(nn.Module):

    def __init__(self, 
                n_surfaces,
                total_length=13*fs, RIR_length=96000, filter_length=1023, source_kernel_length=1023,
                surface_freqs=[63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                dir_freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                n_fibonacci = 128, sharpness = 8,
                late_network_style = "ResidualOnly",
                spline_indices = [200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 44000, 48000, 56000, 70000, 80000],
                delay_noise=True,
                no_transmission=True,
                no_spline=False,
                point_cutoff=0,
                train_sharpness=False
                ):

        super().__init__()

        #Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("RIR_length:\t"+str(RIR_length))

        #Arguments
        self.n_surfaces = n_surfaces

        self.total_length = total_length
        self.RIR_length = RIR_length
        self.filter_length = filter_length
        self.source_kernel_length = source_kernel_length

        self.surface_freqs = torch.tensor(surface_freqs)
        self.dir_freqs = torch.tensor(dir_freqs)

        self.n_fibonacci = n_fibonacci

        if not train_sharpness:
            self.sharpness = sharpness
        else:
            print("Training Sharpness")
            self.sharpness = nn.Parameter(torch.ones(1)*sharpness)


        self.late_network_style = late_network_style

        self.spline_indices = spline_indices
        self.delay_noise = delay_noise
        self.no_transmission = no_transmission
        self.no_spline = no_spline

        #Other Attributes
        self.n_surface_freqs = len(surface_freqs)
        self.n_dir_freqs = len(dir_freqs)
        self.ts = torch.arange(self.RIR_length).to(self.device)
        self.times = self.ts/fs
        self.sigmoid = nn.Sigmoid()

        #Early and Late reflection initialization
        self.init_early()
        self.init_late()

        #Spline stuff
        self.n_spline = len(spline_indices)
        self.IK = get_time_interpolator(n_target=RIR_length, indices=torch.tensor(spline_indices)).to(self.device)   
        self.spline_values = nn.Parameter(torch.linspace(-5, 5, self.n_spline))

        #For full-sound estimation
        self.source_audio = nn.Parameter(torch.randn(total_length)/4)
        self.full_residual = nn.Parameter(torch.zeros(total_length))



        #Fixing 1/r rolloff
        self.point_cutoff = point_cutoff

    def init_early(self):


        # Initializing "Energy Vector", which stores the coefficients for each surface
        if not self.no_transmission:
            #Second axis indices are diffuse reflection, specular reflection, transmission, and absorption
            A = torch.zeros(self.n_surfaces,4,self.n_surface_freqs)
            A[:,1,:] = 4
            A[:,2,:] = 1

        else:
            #Second axis indices are specular reflection and absorption
            A = torch.zeros(self.n_surfaces,2,self.n_surface_freqs)
            A[:,0,:] = 4
            A[:,1,:] = 1
        self.energy_vector = nn.Parameter(A)



        #Interpolating the octave-spaced frequency values onto a finer grid
        n_freq_samples = 1 + 2 ** int(math.ceil(math.log(self.filter_length, 2))) #Next Power of two plus one
        self.freq_grid = torch.linspace(0.0, nyq, n_freq_samples) #Setting up the grid

        surface_freq_indices = torch.round(self.surface_freqs*((n_freq_samples-1)/nyq)).int() #Indices on the grid where the surface frequencies are
        self.surface_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, freq_indices=surface_freq_indices).to(self.device) #Linear Interpolator, n_freqs x n_grid.

        dir_freq_indices = torch.round(self.dir_freqs*((n_freq_samples-1)/nyq)).int()
        self.dir_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, freq_indices=dir_freq_indices).to(self.device)
        self.window = torch.Tensor(scipy.fft.fftshift(scipy.signal.get_window("hamming", self.filter_length, fftbins=False))).to(self.device) # Use the window method for filter design



        #Source Kernel
        source_kernel = torch.zeros(self.source_kernel_length)
        source_kernel[0] = 0.1
        self.source_kernel = nn.Parameter(source_kernel)





        #Directivity Pattern
        print("Spherical Heatmap")
        print("N = " + str(self.n_fibonacci))
        self.sphere_points = torch.tensor(fibonacci_sphere(self.n_fibonacci)).to(self.device)
        self.directivity_sphere = nn.Parameter(torch.ones(self.n_fibonacci, self.n_dir_freqs))




        #Rate of Decay
        self.decay = nn.Parameter(5*torch.ones(1))


        #1/r^2 Term
        #self.near_field_factor = nn.Parameter(torch.zeros(1))
        self.near_field_factor = 0



    def init_late(self):

        #Late Stage Amplitude/Decay Network
        if self.late_network_style == "ResidualOnly":
            self.RIR_residual = nn.Parameter(torch.zeros(self.RIR_length))
        else:
            raise ValueError


    def render_early(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
 
        n_paths = loc.delays.shape[0]

        #time_render_start = time.perf_counter()

        """
        Computing Reflection Response
        """
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) 
        amplitudes = torch.sqrt(energy_coeffs)

        #mask is n_paths * n_surfaces * 2 * 1
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        #gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.no_transmission:
            gains_profile = (amplitudes[:,1:3,:].unsqueeze(0)**mask).unsqueeze(-1)
        else:
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) ==0
            gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        #reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)

        #time_ref_response = time.perf_counter()


        """
        Computing Directivity Response
        """
        # Sharpness - shape (1)
        # sphere_points - shape (n_sphere_points)
        # directivity_sphere - shape (n_sphere_points, n_dir_freqs)
        # dots - shape (n_paths, 3)
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = start_directions_normalized @ torch.Tensor(source_basis).double().cuda()
            dots =  start_directions_normalized_transformed @ (self.sphere_points).T

        #Shape - n_paths x n_sphere_points
        else:
            dots = start_directions_normalized @ (self.sphere_points).T

        #Normalized weights for each directivity bin
        weights = torch.exp(-self.sharpness*(1-dots))
        weights = weights/(torch.sum(weights, dim=-1).view(-1, 1))

        #Shape - n_paths x n_sphere points x n_dir_freqs
        weighted = weights.unsqueeze(-1) * self.directivity_sphere
        directivity_profile = torch.sum(weighted, dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        directivity_amplitude_response = torch.exp(directivity_response)


        #time_dir_response = time.perf_counter()


        """
        Computing overall frequency response, minimum phase transform
        """


        frequency_response = directivity_amplitude_response*reflection_frequency_response
        #np.save('frequency_response.npy', frequency_response.detach().cpu().numpy())

        phases = hilbert_one_sided(safe_log(frequency_response), device=self.device)
        fx2 = frequency_response*torch.exp(1j*phases)
        out_full = torch.fft.irfft(fx2)
        out = out_full[...,:self.filter_length] * self.window

        #np.save('out.npy', out.detach().cpu().numpy())


        #time_inversion = time.perf_counter()


        """
        Compiling RIR
        """
        reflection_kernels = torch.zeros(n_paths, self.RIR_length).to(self.device)
        


        if self.delay_noise:
            noises = 7*torch.randn(n_paths, 1).to(self.device)

        for i in range(n_paths):            
            if self.delay_noise:
                delay = loc.delays[i] + torch.round(noises[i]).int()
            else:
                delay = loc.delays[i]
            

            #140/delay gives us the radius in meters
            reflection_kernels[i, delay:delay+out.shape[-1]] = out[i]*(       (140/(max(self.point_cutoff,delay)))  +    (self.near_field_factor)*((140/delay)**2)          )

            if self.no_transmission:
                reflection_kernels = reflection_kernels*paths_without_transmissions.reshape(-1,1).to(self.device)
        
        #time_compile = time.perf_counter()


        if hrirs is not None:
            reflection_kernels = torch.unsqueeze(reflection_kernels, dim=1) #n_paths x 1 x length
            reflection_kernels = F.fftconvolve(reflection_kernels, hrirs.to(self.device)) # hrirs are n_paths x 2 x length
            RIR_early = torch.sum(reflection_kernels, axis=0)
            RIR_early = F.fftconvolve( (self.source_kernel - torch.mean(self.source_kernel)).view(1,-1), RIR_early)[...,:self.RIR_length]
            RIR_early = RIR_early*(self.sigmoid(self.decay)**self.times)
            return RIR_early

        RIR_early = torch.sum(reflection_kernels, axis=0)
        RIR_early = F.fftconvolve(self.source_kernel - torch.mean(self.source_kernel), RIR_early)[:self.RIR_length]
        RIR_early = RIR_early*(self.sigmoid(self.decay)**self.times)

        #time_convolve = time.perf_counter()
        
        # print("TIMES1")
        # print(time_render_start)
        # print(time_ref_response-time_render_start)
        # print(time_dir_response-time_render_start)
        # print(time_inversion-time_render_start)
        # print(time_compile-time_render_start)
        # print(time_convolve-time_render_start, flush=True)

        return RIR_early, reflection_kernels
        
    def render_late(self, loc):        
        if self.late_network_style == "ResidualOnly":
            late = self.RIR_residual
        else:
            raise ValueError
        
        return late

    def render_RIR(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        early, _ = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)


        while torch.sum(torch.isnan(early)) > 0:
            print("nan found - trying again")
            early, _ = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)


        late = self.render_late(loc=loc)

        if not self.no_spline:
            self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
            RIR = late*self.spline + early*(1-self.spline)
        else:
            RIR = late+early
        return RIR

    def render_full(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        RIR = self.render_RIR(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        if hrirs is None:
            output = F.fftconvolve(RIR, self.source_audio)[...,:self.total_length]
        else:
            output = F.fftconvolve(RIR, self.source_audio.view(1,-1))[...,:self.total_length]

        return output + self.full_residual



class ListenerLocation():
    def __init__(self, source_xyz, listener_xyz, n_surfaces, reflections, transmissions, delays, start_directions, end_directions = None):
        self.source_xyz = source_xyz
        self.listener_xyz = listener_xyz
        self.reflection_mask = gen_counts(reflections,n_surfaces)
        self.transmission_mask = gen_counts(transmissions,n_surfaces)
        self.delays = torch.tensor(delays)
        self.start_directions_normalized = torch.tensor(start_directions/np.linalg.norm(start_directions, axis=-1).reshape(-1, 1))

        if end_directions is not None:
            self.end_directions_normalized = torch.tensor(end_directions/np.linalg.norm(end_directions, axis=-1).reshape(-1, 1))
        else:
            self.end_directions_normalized = None

def get_listener(source_xyz, listener_xyz, surfaces, speed_of_sound=None, max_order=5, load_dir = None, load_num=None, parallel_surface_pairs=None, max_axial_order=50):
    
    if load_dir is None:
        if speed_of_sound is None:
            raise ValueError("Need Speed of Sound if Computing from Scratch!")
        reflections, transmissions, delays, start_directions, end_directions = trace.get_reflections_transmissions_and_delays(source=source_xyz, dest=listener_xyz, surfaces=surfaces, speed_of_sound=speed_of_sound, max_order=max_order,parallel_surface_pairs=parallel_surface_pairs, max_axial_order=max_axial_order)

    else:
        print("Listener Loading From" + load_dir)
        reflections = np.load(os.path.join(load_dir,"reflections/"+str(load_num)+".npy"), allow_pickle=True)
        transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(load_num)+".npy"), allow_pickle=True)
        delays = np.load(os.path.join(load_dir, "delays/"+str(load_num)+".npy"))
        start_directions = np.load(os.path.join(load_dir, "starts/"+str(load_num)+".npy"))
        end_directions = np.load(os.path.join(load_dir, "ends/"+str(load_num)+".npy"))

    L = ListenerLocation(source_xyz=source_xyz, listener_xyz=listener_xyz, n_surfaces=len(surfaces), reflections=reflections, transmissions=transmissions, delays=delays, start_directions = start_directions, end_directions = end_directions)

    return L

def get_interpolator(n_freq_target, freq_indices):
    result = torch.zeros(len(freq_indices),n_freq_target)
    diffs = torch.diff(freq_indices)

    for i,index in enumerate(freq_indices):  
        if i==0:
            linterp = torch.cat((torch.ones(freq_indices[0]), 1-torch.arange(diffs[0])/diffs[0]))
            result[i,0:freq_indices[1]] = linterp
        elif i==len(freq_indices)-1:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], torch.ones(n_freq_target-freq_indices[i])))
            result[i,freq_indices[i-1]:] = linterp
        else:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], 1-torch.arange(diffs[i])/diffs[i]))
            result[i,freq_indices[i-1]:freq_indices[i+1]] = linterp

    return result

def gen_counts(surface_indices, n_surfaces):
    n_reflections = len(surface_indices)
    result = torch.zeros(n_reflections, n_surfaces)
    for i in range(n_reflections):
        for j in surface_indices[i]:
            result[i,j] +=1
    return result

def get_time_interpolator(n_target, indices):
    result = torch.zeros(len(indices),n_target)
    diffs = torch.diff(indices)

    for i,index in enumerate(indices):  
        if i==0:
            linterp = torch.cat((torch.arange(indices[0])/indices[0], 1-torch.arange(diffs[0])/diffs[0]))
            result[i,0:indices[1]] = linterp
        elif i==len(indices)-1:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], torch.ones(n_target-indices[i])))
            result[i,indices[i-1]:] = linterp
        else:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], 1-torch.arange(diffs[i])/diffs[i]))
            result[i,indices[i-1]:indices[i+1]] = linterp

    return result

#Assume x is ODD length
def hilbert_one_sided(x, device):
    N = 2*x.shape[-1] - 1
    Xf = torch.fft.irfft(x, n=N)
    h = torch.zeros(N).to(device)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    x = torch.fft.rfft(Xf * h)
    return torch.imag(x)


def fibonacci_sphere(n_samples):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def safe_log(x, eps=1e-9):
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)
