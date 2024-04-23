import scipy.signal
import math
import torch
import torch.nn as nn
import numpy as np
import trace
import torchaudio.functional as F
import os
import librosa

"""
Set the sampling rate to 48 kHz
"""
nyq = 24000
fs = nyq*2
torch.set_default_dtype(torch.float32)

class Renderer(nn.Module):
    """
    Class for a RIR renderer.

    Constructor Parameters
    ----------
    n_surfaces: int
        number of surfaces to model in the room
    RIR_length: int
        length of the RIR in samples
    filter_length: int
        length of the reflection's contribution to the RIR, in samples
    source_response_length: int
        length of the convolutional kernel used to model the sound source
    surface_freqs: list of int
        frequencies to fit each surface's reflection response at, the rest are interpolated
    dir_freqs: list of int
        frequencies to fit the source's directivity response at, the rest are interpolated
    n_fibonacci: int
        number of points to distribute on the unit sphere,
        at which where the speaker's directivity is modeled
    spline_indices: list of int
        times (in samples) at which the late/early stage spline is fit
    toa_perturb: bool
        if times of arrival are perturbed (used during training)
    model_transmission: bool
        if we are modeling surface transmission as well.

    """
    def __init__(self, 
                n_surfaces,
                RIR_length=96000, filter_length=1023, source_response_length=1023,
                surface_freqs=[32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                dir_freqs = [32, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
                n_fibonacci = 128, sharpness = 8,
                late_stage_model = "UniformResidual",
                spline_indices = [200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                                  6000, 7000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
                                  22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000,
                                  38000, 40000, 44000, 48000, 56000, 70000, 80000],
                toa_perturb=True,
                model_transmission=False):

        super().__init__()

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Arguments
        self.n_surfaces = n_surfaces
        self.RIR_length = RIR_length
        self.filter_length = filter_length
        self.source_response_length = source_response_length
        self.surface_freqs = torch.tensor(surface_freqs)
        self.dir_freqs = torch.tensor(dir_freqs)
        self.n_fibonacci = n_fibonacci
        self.sharpness = sharpness
        self.late_stage_model = late_stage_model
        self.spline_indices = spline_indices
        self.toa_perturb = toa_perturb
        self.model_transmission = model_transmission
        self.no_spline = no_spline

        # Other Attributes
        self.n_surface_freqs = len(surface_freqs)
        self.n_dir_freqs = len(dir_freqs)
        self.samples = torch.arange(self.RIR_length).to(self.device)
        self.times = self.samples/fs
        self.sigmoid = nn.Sigmoid()

        # Early and Late reflection initialization
        self.init_early()
        self.init_late()

        # Spline
        self.n_spline = len(spline_indices)
        self.IK = get_time_interpolator(n_target=RIR_length, indices=torch.tensor(spline_indices)).to(self.device)   
        self.spline_values = nn.Parameter(torch.linspace(-5, 5, self.n_spline))

    def init_early(self):

        # Initializing "Energy Vector", which stores the coefficients for each surface
        if self.model_transmission:
            # Second axis indices are specular reflection, transmission, and absorption
            A = torch.zeros(self.n_surfaces,3, self.n_surface_freqs)
        else:
            # Second axis indices are specular reflection and absorption
            A = torch.zeros(self.n_surfaces,2,self.n_surface_freqs)

        # Setting up Frequency responses
        n_freq_samples = 1 + 2 ** int(math.ceil(math.log(self.filter_length, 2)))
        self.freq_grid = torch.linspace(0.0, nyq, n_freq_samples)
        surface_freq_indices = torch.round(self.surface_freqs*((n_freq_samples-1)/nyq)).int() 
        self.surface_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, freq_indices=surface_freq_indices).to(self.device) 
        dir_freq_indices = torch.round(self.dir_freqs*((n_freq_samples-1)/nyq)).int()
        self.dir_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, freq_indices=dir_freq_indices).to(self.device)
        self.window = torch.Tensor(scipy.fft.fftshift(scipy.signal.get_window("hamming", self.filter_length, fftbins=False))).to(self.device)

        # Source Response
        source_response = torch.zeros(self.source_response_length)
        source_response[0] = 0.1 # Initialize to identity
        self.source_response = nn.Parameter(source_response)

        # Directivity Pattern
        self.sphere_points = torch.tensor(fibonacci_sphere(self.n_fibonacci)).to(self.device)
        self.directivity_sphere = nn.Parameter(torch.ones(self.n_fibonacci, self.n_dir_freqs))

        # Parameter for energy decay over time
        self.decay = nn.Parameter(5*torch.ones(1))


    def init_late(self):
        """Initializing the late-stage model - other methods besides a uniform residual may be explored in the future"""
        if self.late_stage_model == "UniformResidual":
            self.RIR_residual = nn.Parameter(torch.zeros(self.RIR_length))
        else:
            raise ValueError("Invalid Residual Mode")

    def render_early(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):=
        """
        Renders the early-stage RIR

        Parameters
        ----------
        loc: ListenerLocation
            characterizes the location at which we render the early-stage RIR
        hrirs: np.array (n_paths x 2 x h_rir_length)
            head related IRs for each reflection path's direction
        source_axis_1: np.array (3,)
            first axis specifying virtual source rotation,
            default is None which is (1,0,0)
        source_axis_2: np.array (3,)
            second axis specifying virtual source rotation,
            default is None which is (0,1,0)        
        """

        """
        Computing Reflection Response
        """
        n_paths = loc.delays.shape[0]
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) # Conservation of energy
        amplitudes = torch.sqrt(energy_coeffs)

        # mask is n_paths * n_surfaces * 2 * 1 - 1 at (path, surface, 0) indicates
        # path reflects off surface
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        # gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.model_transmission:  
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) == 0
        gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        # reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)


        """
        Computing Directivity Response
        """
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        # If there is speaker rotation
        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = start_directions_normalized @ torch.Tensor(source_basis).double().cuda()
        
        dots =  start_directions_normalized_transformed @ (self.sphere_points).T
        
        # Normalized weights for each directivity bin
        weights = torch.exp(-self.sharpness*(1-dots))
        weights = weights/(torch.sum(weights, dim=-1).view(-1, 1))
        weighted = weights.unsqueeze(-1) * self.directivity_sphere
        directivity_profile = torch.sum(weighted, dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        directivity_amplitude_response = torch.exp(directivity_response)


        """
        Computing overall frequency response, minimum phase transform
        """
        frequency_response = directivity_amplitude_response*reflection_frequency_response
        phases = hilbert_one_sided(safe_log(frequency_response), device=self.device)
        fx2 = frequency_response*torch.exp(1j*phases)
        out_full = torch.fft.irfft(fx2)
        out = out_full[...,:self.filter_length] * self.window


        """
        Compiling RIR
        """
        reflection_kernels = torch.zeros(n_paths, self.RIR_length).to(self.device)
    
        if self.toa_perturb:
            noises = 7*torch.randn(n_paths, 1).to(self.device)

        for i in range(n_paths):            
            if self.toa_perturb:
                delay = loc.delays[i] + torch.round(noises[i]).int()
            else:
                delay = loc.delays[i]

            # 140/delay gives us the radius in meters
            reflection_kernels[i, delay:delay+out.shape[-1]] = out[i]*(140/(delay))

            if not self.model_transmission:
                reflection_kernels = reflection_kernels*paths_without_transmissions.reshape(-1,1).to(self.device)
        
        if hrirs is not None:
            reflection_kernels = torch.unsqueeze(reflection_kernels, dim=1) # n_paths x 1 x length
            reflection_kernels = F.fftconvolve(reflection_kernels, hrirs.to(self.device)) # hrirs are n_paths x 2 x length
            RIR_early = torch.sum(reflection_kernels, axis=0) 
            RIR_early = F.fftconvolve( (self.source_response - torch.mean(self.source_response)).view(1,-1), RIR_early)[...,:self.RIR_length]
        else:
            RIR_early = torch.sum(reflection_kernels, axis=0)
            RIR_early = F.fftconvolve(self.source_response - torch.mean(self.source_response), RIR_early)[:self.RIR_length]
        
        RIR_early = RIR_early*(self.sigmoid(self.decay)**self.times)
        return RIR_early
        
    def render_late(self, loc):    
        """Renders the late-stage RIR. Future work may implement other ways of modeling the late-stage."""    
        if self.late_stage_model == "UniformResidual":
            late = self.RIR_residual
        else:
            raise ValueError("Invalid Residual Mode")
        return late

    def render_RIR(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR.""""
        early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR


class ListenerLocation():
    """
    Class for a Listener Locations renderer.

    Constructor Parameters
    ----------
    source_xyz: np.array (3,)
        xyz location of the sound source in meters.
    listener_xyz: np.array (3,)
        xyz location of the listener location in meters.
    n_surfaces: number of surfaces
    reflections: list of list of int. Indices of surfaces that each path reflects on.
    transmission: list of list of int. Indices of surfaces that each path transmits through.
    delays: np.array (n_paths,)
        time delays in samples for each path.
    start_directions: np.array(n_paths, 3)
        vectors in the start directions of each path
    end_directions: np.array(n_paths, 3)
        vectors indicating the direction at which each path enters the listener.
    """
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

def get_listener(source_xyz, listener_xyz, surfaces, speed_of_sound=None, max_order=5, load_dir=None, load_num=None, parallel_surface_pairs=None, max_axial_order=50):
    """Function to get a ListenerLocation. If load_dir is provided, loads precomputed paths"""
    if load_dir is None: 

        # Tracing from Scratch
        if speed_of_sound is None:
            raise ValueError("Need Speed of Sound if Computing from Scratch!")
        reflections, transmissions, delays, start_directions, end_directions = trace.get_reflections_transmissions_and_delays(source=source_xyz, dest=listener_xyz, surfaces=surfaces, speed_of_sound=speed_of_sound, max_order=max_order,parallel_surface_pairs=parallel_surface_pairs, max_axial_order=max_axial_order)

    else:
        # Loading precomputed paths
        print("Listener Loading From" + load_dir)
        reflections = np.load(os.path.join(load_dir,"reflections/"+str(load_num)+".npy"), allow_pickle=True)
        transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(load_num)+".npy"), allow_pickle=True)
        delays = np.load(os.path.join(load_dir, "delays/"+str(load_num)+".npy"))
        start_directions = np.load(os.path.join(load_dir, "starts/"+str(load_num)+".npy"))
        end_directions = np.load(os.path.join(load_dir, "ends/"+str(load_num)+".npy"))

    L = ListenerLocation(source_xyz=source_xyz, listener_xyz=listener_xyz, n_surfaces=len(surfaces), reflections=reflections, transmissions=transmissions, delays=delays, start_directions = start_directions, end_directions = end_directions)

    return L

def get_interpolator(n_freq_target, freq_indices):
    """Function to return a tensor that helps with efficient linear frequency interpolation"""
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
    """Generates a (n_paths, n_surfaces) 0-1 mask indicating reflections"""
    n_reflections = len(surface_indices)
    result = torch.zeros(n_reflections, n_surfaces)
    for i in range(n_reflections):
        for j in surface_indices[i]:
            result[i,j] += 1
    return result

def get_time_interpolator(n_target, indices):
    """Function to return a tensor that helps with efficient linear interpolation"""
    result = torch.zeros(len(indices),n_target)
    diffs = torch.diff(indices)

    for i,index in enumerate(indices):  
        if i == 0:
            linterp = torch.cat((torch.arange(indices[0])/indices[0], 1-torch.arange(diffs[0])/diffs[0]))
            result[i,0:indices[1]] = linterp
        elif i == len(indices)-1:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], torch.ones(n_target-indices[i])))
            result[i,indices[i-1]:] = linterp
        else:
            linterp = torch.cat((torch.arange(diffs[i-1])/diffs[i-1], 1-torch.arange(diffs[i])/diffs[i]))
            result[i,indices[i-1]:indices[i+1]] = linterp

    return result

def hilbert_one_sided(x, device):
    """
    Returns minimum phases for a given log-frequency response x.
    Assume x.shape[-1] is ODD
    """
    N = 2*x.shape[-1] - 1
    Xf = torch.fft.irfft(x, n=N)
    h = torch.zeros(N).to(device)
    h[0] = 1
    h[1:(N + 1) // 2] = 2
    x = torch.fft.rfft(Xf * h)
    return torch.imag(x)


def fibonacci_sphere(n_samples):
    """Distributes n_samples on a unit fibonacci_sphere"""
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)

    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2
        radius = math.sqrt(1 - y * y)

        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)



def safe_log(x, eps=1e-9):
    """Prevents Taking the log of a non-positive number"""
    safe_x = torch.where(x <= eps, eps, x)
    return torch.log(safe_x)
