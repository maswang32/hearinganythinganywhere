import os
import math
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torchaudio.functional as F
import trace1

#############################
from scipy.special import sph_harm

import matplotlib.pyplot as plt################
##############################


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
        length of each reflection's contribution to the RIR, in samples
    source_response_length: int
        length of the convolutional kernel used to model the sound source
    surface_freqs: list of int
        frequencies in Hz to fit each surface's reflection response at, the rest are interpolated
    dir_freqs: list of int
        frequencies in Hz to fit the source's directivity response at, the rest are interpolated
    n_fibonacci: int
        number of points to distribute on the unit sphere,
        at which where the speaker's directivity is fit.
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
                model_transmission=False,
                fs=48000):

        super().__init__()

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu" 
        self.nyq = fs/2

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

        ######################################################
        # Beampattern orders cutoff frequencies
        self.bp_ord_cut_freqs = nn.Parameter(torch.Tensor([70, 400, 800, 1000, 1300, 2000]))
        #######################################################

    def init_early(self):
        ##################### A si potrebbe inizializzare in modo diverso conoscendo i materiali di cui sono fatte le superfici ###################
         
        # Initializing "Energy Vector", which stores the coefficients for each surface
        if self.model_transmission:
            # Second axis indices are specular reflection, transmission, and absorption
            A = torch.zeros(self.n_surfaces,3, self.n_surface_freqs)
        else:
            # Second axis indices are specular reflection and absorption
            A = torch.zeros(self.n_surfaces,2,self.n_surface_freqs)
        
        self.energy_vector = nn.Parameter(A)

        # Setting up Frequency responses
        n_freq_samples = 1 + 2 ** int(math.ceil(math.log(self.filter_length, 2))) #????????why filter length
        #print("n_freq_samples", n_freq_samples)
        self.freq_grid = torch.linspace(0.0, self.nyq, n_freq_samples)
        #print("self.freq_grid", self.freq_grid)
        surface_freq_indices = torch.round(self.surface_freqs*((n_freq_samples-1)/self.nyq)).int() 
        #print("self.surfaces_freq_indices", surface_freq_indices)

        self.surface_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples, 
                                                          freq_indices=surface_freq_indices).to(self.device)

        dir_freq_indices = torch.round(self.dir_freqs*((n_freq_samples-1)/self.nyq)).int()
        self.dir_freq_interpolator = get_interpolator(n_freq_target=n_freq_samples,
                                                      freq_indices=dir_freq_indices).to(self.device)

        self.window = torch.Tensor(
            scipy.fft.fftshift(scipy.signal.get_window("hamming", self.filter_length, fftbins=False))).to(self.device)

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

    def render_early(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
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

        Returns
        -------
        RIR_early - (N,) tensor, early-stage RIR
        """

        """
        Computing Reflection Response
        """
        n_paths = loc.delays.shape[0]
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) # Conservation of energy
        amplitudes = torch.sqrt(energy_coeffs).to(self.device)

        # mask is n_paths * n_surfaces * 2 * 1 - 1 at (path, surface, 0) indicates
        # path reflects off surface
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        # gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.model_transmission:  
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) == 0
            #print("paths_without_transmissions",paths_without_transmissions, len(paths_without_transmissions))
        gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        # reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(
            torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)##########??????perché moltiplica coefficiente riflessione e assorbimento?


        """
        Computing Directivity Response
        """
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        # If there is speaker rotation
        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = (
                start_directions_normalized @ torch.Tensor(source_basis).double().cuda())
            dots =  start_directions_normalized_transformed @ (self.sphere_points).T
        else:
            dots = start_directions_normalized @ (self.sphere_points).T

        
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

            # factor/delay gives us the 1/(radius in meters)
            factor = (2*self.nyq)/343
            reflection_kernels[i, delay:delay+out.shape[-1]] = out[i]*(factor/(delay))

            if not self.model_transmission:
                reflection_kernels = reflection_kernels*paths_without_transmissions.reshape(-1,1).to(self.device)

            ######################################################
            if (i==0):  
                plt.plot(reflection_kernels[i].detach().cpu())
                plt.title("Plot del pad sig")
                plt.xlabel("Indice")
                plt.ylabel("Valore")
                plt.grid(True)
                plt.show() 
            #######################################################

        if hrirs is not None:
            reflection_kernels = torch.unsqueeze(reflection_kernels, dim=1) # n_paths x 1 x length
            reflection_kernels = F.fftconvolve(reflection_kernels, hrirs.to(self.device)) # hrirs are n_paths x 2 x length
            RIR_early = torch.sum(reflection_kernels, axis=0) 
            RIR_early = F.fftconvolve(
                (self.source_response - torch.mean(self.source_response)).view(1,-1), RIR_early)[...,:self.RIR_length]
        else:
            RIR_early = torch.sum(reflection_kernels, axis=0)
            RIR_early = F.fftconvolve(
                self.source_response - torch.mean(self.source_response), RIR_early)[:self.RIR_length]
        
        RIR_early = RIR_early*(self.sigmoid(self.decay)**self.times)
        return RIR_early
    
    ######################################
    angular_sensitivities_em64=[{'frequency_range': (20, 1000), 'angle': 90}, {'frequency_range': (1000, 5000), 'angle': 60}, {'frequency_range': (5000, 20000), 'angle': 45}]
    ######################################

    ##############################################################################
    def render_early_with_directions(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None, angular_sensitivities= angular_sensitivities_em64, listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
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
        angular_sensitivities: list of dictionaries
            list specifying the angular sensitivity (in degrees) for different ranges of frequencies        

        Returns
        -------
        RIR_early_by_direction - list of dictionaries (N,) for each frequency_range, for each direction, a time domain early RIR is specified
        """

        """
        Computing Reflection Response
        """
        n_paths = loc.delays.shape[0]
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) # Conservation of energy
        amplitudes = torch.sqrt(energy_coeffs).to(self.device)

        # mask is n_paths * n_surfaces * 2 * 1 - 1 at (path, surface, 0) indicates
        # path reflects off surface
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        # gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.model_transmission:  
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) == 0
        gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        # reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(
            torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)##########??????perché moltiplica coefficiente riflessione e assorbimento?


        """
        Computing Directivity Response
        """
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        # If there is speaker rotation
        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = (
                start_directions_normalized @ torch.Tensor(source_basis).double().cuda())
            dots =  start_directions_normalized_transformed @ (self.sphere_points).T
        else:
            dots = start_directions_normalized @ (self.sphere_points).T

        
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

            # factor/delay gives us the 1/(radius in meters)
            factor = (2*self.nyq)/343
            reflection_kernels[i, delay:delay+out.shape[-1]] = out[i]*(factor/(delay))

            if not self.model_transmission:
                reflection_kernels = reflection_kernels*paths_without_transmissions.reshape(-1,1).to(self.device)
        
        if hrirs is not None:
            reflection_kernels = torch.unsqueeze(reflection_kernels, dim=1) # n_paths x 1 x length
            reflection_kernels = F.fftconvolve(reflection_kernels, hrirs.to(self.device)) # hrirs are n_paths x 2 x length
        '''
            RIR_early = torch.sum(reflection_kernels, axis=0) 
            RIR_early = F.fftconvolve(
                (self.source_response - torch.mean(self.source_response)).view(1,-1), RIR_early)[...,:self.RIR_length]
        else:
            RIR_early = torch.sum(reflection_kernels, axis=0)
            RIR_early = F.fftconvolve(
                self.source_response - torch.mean(self.source_response), RIR_early)[:self.RIR_length]
        '''

        norms = np.linalg.norm(loc.end_directions_normalized, axis=-1).reshape(-1,1)
        incoming_listener_directions = -loc.end_directions_normalized/norms
        

        #Make sure listener_forward and listener_left are orthogonal
        assert np.abs(np.dot(listener_forward, listener_left)) < 0.01

        listener_up = np.cross(listener_forward, listener_left)
        listener_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

        #Compute Azimuths and Elevation
        listener_coordinates = incoming_listener_directions @ listener_basis
        azimuths = np.degrees(np.arctan2(listener_coordinates[:, 1], listener_coordinates[:, 0]))
        elevations = np.degrees(np.arctan(listener_coordinates[:, 2]/np.linalg.norm(listener_coordinates[:, 0:2],axis=-1)+1e-8))

        RIR_early_by_direction = initialize_directional_list(angular_sensitivities, self.RIR_length, device= self.device)
        for i in range(n_paths):
            for j in range(len(RIR_early_by_direction)):

                #print("range considerato", RIR_early_by_direction[j]['frequency_range'][0], RIR_early_by_direction[j]['frequency_range'][1])
                #print("direzione di arrivo", azimuths[i], elevations[i])
                
                beampattern_weights = calculate_weights(azimuths[i], elevations[i], int(180/angular_sensitivities[j]['angle']))
                signal_to_add = apply_bandpass_filter(reflection_kernels[i], RIR_early_by_direction[j]['frequency_range'][0], RIR_early_by_direction[j]['frequency_range'][1], fs = self.nyq * 2)
                pattern_max = beam_pattern(azimuths[i], elevations[i], beampattern_weights, int(180/angular_sensitivities[j]['angle']))
                
                for response in RIR_early_by_direction[j]['responses']:
                    #print("beampattern parameters,", response['direction'][0], response['direction'][1], beampattern_weights, int(180/angular_sensitivities[j]['angle']))
                    response['response'] += signal_to_add * beam_pattern(response['direction'][0], response['direction'][1], beampattern_weights, int(180/angular_sensitivities[j]['angle'])) / pattern_max
                    #print("direzioni considerate", response['direction'][0], response['direction'][1],)
                    #print("attenuazione", beam_pattern(response['direction'][0], response['direction'][1], beampattern_weights, int(180/angular_sensitivities[j]['angle'])) / pattern_max)
                

        for interval in RIR_early_by_direction:
            for r in interval['responses']:
                r['response']= F.fftconvolve(
                    self.source_response - torch.mean(self.source_response), r['response'])[:self.RIR_length]
                r['response'] = r['response']*(self.sigmoid(self.decay)**self.times)

        return RIR_early_by_direction
    ##############################################################################

    ##############################################################################
    # Doesn't support hrirs and toa_perturb
    def render_early_with_learned_beampatterns(self, loc, source_axis_1=None, source_axis_2=None, angular_sensitivity= 60, listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
        """
        Renders the early-stage RIR

        Parameters
        ----------
        loc: ListenerLocation
            characterizes the location at which we render the early-stage RIR
        source_axis_1: np.array (3,)
            first axis specifying virtual source rotation,
            default is None which is (1,0,0)
        source_axis_2: np.array (3,)
            second axis specifying virtual source rotation,
            default is None which is (0,1,0)
        angular_sensitivity: float
            angle that determines the number of directions considered      

        Returns
        -------
        directional_freq_responses - list of dictionaries 
             for each direction, a time domain (t_response) and a frequency domain(f_response) early RIR are specified
        """

        """
        Computing Reflection Response
        """
        n_paths = loc.delays.shape[0]
        energy_coeffs = nn.functional.softmax(self.energy_vector, dim=-2) # Conservation of energy
        amplitudes = torch.sqrt(energy_coeffs).to(self.device)

        # mask is n_paths * n_surfaces * 2 * 1 - 1 at (path, surface, 0) indicates
        # path reflects off surface
        mask = torch.stack((loc.reflection_mask, loc.transmission_mask), dim=-1).unsqueeze(-1).to(self.device)
        
        # gains_profile is n_paths * n_surfaces * 2 * num_frequencies * 1
        if not self.model_transmission:  
            paths_without_transmissions = torch.sum(loc.transmission_mask, dim=-1) == 0
        gains_profile = (amplitudes[:,0:2,:].unsqueeze(0)**mask).unsqueeze(-1)

        # reflection_frequency_response = n_paths * n_freq_samples
        reflection_frequency_response = torch.prod(torch.prod(
            torch.sum(self.surface_freq_interpolator*gains_profile, dim=-2),dim=-3),dim=-2)##########??????perché moltiplica coefficiente riflessione e assorbimento?


        """
        Computing Directivity Response
        """
        start_directions_normalized = loc.start_directions_normalized.to(self.device)

        # If there is speaker rotation
        if source_axis_1 is not None and source_axis_2 is not None:
            source_axis_3 = np.cross(source_axis_1 ,source_axis_2)
            source_basis = np.stack( (source_axis_1, source_axis_2, source_axis_3), axis=-1)
            start_directions_normalized_transformed = (
                start_directions_normalized @ torch.Tensor(source_basis).double().cuda())
            dots =  start_directions_normalized_transformed @ (self.sphere_points).T
        else:
            dots = start_directions_normalized @ (self.sphere_points).T

        
        # Normalized weights for each directivity bin
        weights = torch.exp(-self.sharpness*(1-dots))
        weights = weights/(torch.sum(weights, dim=-1).view(-1, 1))
        weighted = weights.unsqueeze(-1)* self.directivity_sphere
        directivity_profile = torch.sum(weighted, dim=1)
        directivity_response = torch.sum(directivity_profile.unsqueeze(-1) * self.dir_freq_interpolator, dim=-2)
        directivity_amplitude_response = torch.exp(directivity_response)

        """
        Computing overall frequency response, minimum phase transform
        """
        frequency_response = directivity_amplitude_response*reflection_frequency_response

        #print("frequency response", frequency_response)#########################


        """
        Introducing delays ####################################################
        """

        max_delay = max(loc.delays)

        frequency_response_with_delays = torch.Tensor().to(self.device)


        for i in range(len(frequency_response)):

            """
            ENTERING THE TIME DOMAIN
            """
            phases = hilbert_one_sided(safe_log(frequency_response[i]), device=self.device)
            fx2 = frequency_response[i]*torch.exp(1j*phases)

            sig = torch.fft.irfft(fx2)
            
            '''
            if (i==0):  
                plt.plot(sig.detach().cpu())
                plt.title("antitrasformata")
                plt.xlabel("Indice")
                plt.ylabel("Valore")
                plt.grid(True)
                plt.show()
            '''
                
            del_pad_sig = torch.cat([torch.zeros(loc.delays[i]).to(self.device), sig, torch.zeros(max_delay - loc.delays[i]).to(self.device)])
            #del_pad_sig = torch.cat([torch.zeros(loc.delays[i]).to(self.device), sig, torch.zeros(self.RIR_length - loc.delays[i]).to(self.device)])##################################
            
            '''
            if (i==0):  
                plt.plot(del_pad_sig.detach().cpu())
                plt.title("Seganle paddato e ritardato")
                plt.xlabel("Indice")
                plt.ylabel("Valore")
                plt.grid(True)
                plt.show()
            '''


            factor = (2*self.nyq)/343
            del_pad_sig = del_pad_sig*(factor/(loc.delays[i])) #attenuazione proporzionaloe alla lunghezza del path


            '''
            if (i==0):  
                plt.plot(del_pad_sig.detach().cpu())
                plt.title("segnale attenuato")
                plt.xlabel("Indice")
                plt.ylabel("Valore")
                plt.grid(True)
                plt.show()
            '''
            
            """
            BACK TO THE FREQUENCY DOMAIN
            """

            new_freq_resp = torch.fft.rfft(del_pad_sig)

            ##########qui si potrebbe pensare di ridurre la dimensione di new_freq_response (tutte alla stessa lunghezza)
            ##########perdi risoluzione ma più facile la computazione (altrimenti sono più lunghe anche di quelle gestite nel codice originale)

            '''
            plt.plot(torch.fft.irfft(new_freq_resp).detach())
            plt.title("Plot after antitransforming")
            plt.xlabel("Indice")
            plt.ylabel("Valore")
            plt.grid(True)
            plt.show()
            '''
            frequency_response_with_delays = torch.cat((frequency_response_with_delays, new_freq_resp.unsqueeze(0)), dim =0)


        frequency_response= frequency_response_with_delays

        #print("frequency response shape", frequency_response.shape)
        

        norms = np.linalg.norm(loc.end_directions_normalized, axis=-1).reshape(-1,1)
        incoming_listener_directions = -loc.end_directions_normalized/norms
        

        #Make sure listener_forward and listener_left are orthogonal
        assert np.abs(np.dot(listener_forward, listener_left)) < 0.01

        listener_up = np.cross(listener_forward, listener_left)
        listener_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

        #Compute Azimuths and Elevation
        listener_coordinates = incoming_listener_directions @ listener_basis
        azimuths = np.degrees(np.arctan2(listener_coordinates[:, 1], listener_coordinates[:, 0]))
        elevations = np.degrees(np.arctan(listener_coordinates[:, 2]/np.linalg.norm(listener_coordinates[:, 0:2],axis=-1)+1e-8))

        directional_freq_responses = initialize_directional_list_for_beampattern(angular_sensitivity, len(frequency_response[0]), self.device)############secondo parametro ok se c'è almeno un path (si potra scrivere pù elegant tipo con dim=1)
        n_orders = len (self.bp_ord_cut_freqs)

        cutoffs = self.bp_ord_cut_freqs.detach()


        self.freq_grid = torch.linspace(0.0, self.nyq, len(frequency_response[0]))########################################

        for i in range(n_paths):
            print("path: ", i + 1, "of", n_paths)
            print("incoming direction", azimuths[i], elevations[i])
            if(self.model_transmission or paths_without_transmissions[i]):
                for j in range(len(frequency_response[0])):
                    bp_weights = calculate_weights_all_orders(self.freq_grid[j], azimuths[i], elevations[i], cutoffs)
                    for direction in directional_freq_responses:
                        #print("beampattern prameters, ", direction['angle'][0], direction['angle'][1], bp_weights, n_orders)
                        direction['f_response'][j] += frequency_response[i][j] * beam_pattern(direction['angle'][0], direction['angle'][1], bp_weights, n_orders)############è ok mantenerlo come complesso?

        padding = self.RIR_length - len(directional_freq_responses[0]['f_response']) ##########padding necessario per farli lunghi self.RIR_length

        for r in directional_freq_responses:

            """
            TIME DOMAIN
            """
            out_full = torch.fft.irfft(r['f_response'])############provo a mettere questo

            new_window = torch.Tensor(
            scipy.fft.fftshift(scipy.signal.get_window("hamming", len(out_full), fftbins=False))).to(self.device)#necessario creare una nuova window perchè doveva avere la giusta dimensione

            r['t_response'] = out_full * new_window #########window serve?


            r['t_response'] = torch.cat((r['t_response'], torch.zeros(padding).to(self.device))) ##li rendo uguali in lunghezza a quello che sarà la late response


            r['t_response']= F.fftconvolve(
                    self.source_response - torch.mean(self.source_response), r['t_response'])[:self.RIR_length]
            r['t_response'] = r['t_response']*((self.sigmoid(self.decay)**self.times)) #[:len(r['t_response'])])################aggiunto il cut??

        return directional_freq_responses
    
    ##########################################################################################################################
        
    def render_late(self, loc):    
        """Renders the late-stage RIR. Future work may implement other ways of modeling the late-stage."""    
        if self.late_stage_model == "UniformResidual":
            late = self.RIR_residual
        else:
            raise ValueError("Invalid Residual Mode")
        return late

    def render_RIR(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None):
        """Renders the RIR."""
        early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        while torch.sum(torch.isnan(early)) > 0: # Check for numerical issues
            print("nan found - trying again")
            early = self.render_early(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2)

        late = self.render_late(loc=loc)

        # Blend early and late stage together using spline
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)
        RIR = late*self.spline + early*(1-self.spline)
        return RIR
    
    ###############################################################################

    def render_RIR_by_directions(self, loc, hrirs=None, source_axis_1=None, source_axis_2=None, angular_sensitivities= angular_sensitivities_em64,listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
        """Renders the RIR."""
        frequency_list = self.render_early_with_directions(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2, angular_sensitivities= angular_sensitivities, listener_forward=listener_forward, listener_left=listener_left)

        for interval in frequency_list:
            for r in interval['responses']:
                while torch.sum(torch.isnan(r['response'])) > 0: # Check for numerical issues
                    print("nan found - trying again")
                    frequency_list = self.render_early_with_directions(loc=loc, hrirs=hrirs, source_axis_1=source_axis_1, source_axis_2=source_axis_2, angular_sensitivities= angular_sensitivities, listener_forward=listener_forward, listener_left=listener_left)

        late = self.render_late(loc=loc)
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)


    
        for interval in frequency_list:
            signal_to_add = apply_bandpass_filter(late, interval['frequency_range'][0], interval['frequency_range'][1], fs=self.nyq * 2) / len(interval['responses'])
            for r in interval['responses']:
                r['response'] = signal_to_add*self.spline + r['response']*(1-self.spline)


        return frequency_list
    ###################################################################################

    ###############################################################################

    def render_RIR_learned_beampattern(self, loc, source_axis_1=None, source_axis_2=None, angular_sensitivity= 60,listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
        """Renders the RIR."""
        directional_freq_responses = self.render_early_with_learned_beampatterns(loc=loc, source_axis_1=source_axis_1, source_axis_2=source_axis_2, angular_sensitivity= angular_sensitivity, listener_forward=listener_forward, listener_left=listener_left)

        for r in directional_freq_responses:
            while torch.sum(torch.isnan(r['t_response'])) > 0: # Check for numerical issues
                print("nan found - trying again")
                directional_freq_responses = self.render_early_with_learned_beampatterns(loc=loc, source_axis_1=source_axis_1, source_axis_2=source_axis_2, angular_sensitivity= angular_sensitivity, listener_forward=listener_forward, listener_left=listener_left)

        late = self.render_late(loc=loc)
        self.spline = torch.sum(self.sigmoid(self.spline_values).view(self.n_spline,1)*self.IK, dim=0)

        signal_to_add = late / len(directional_freq_responses)
    
        for r in directional_freq_responses:    
            
            r['t_response'] = signal_to_add*self.spline + r['t_response']*(1-self.spline)


        return directional_freq_responses
    ###################################################################################

    

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
    reflections: list of list of int. 
        Indices of surfaces that each path reflects on.
    transmission: list of list of int. 
        Indices of surfaces that each path transmits through.
    delays: np.array (n_paths,)
        time delays in samples for each path.
    start_directions: np.array(n_paths, 3)
        vectors in the start directions of each path
    end_directions: np.array(n_paths, 3)
        vectors indicating the direction at which each path enters the listener.
    """
    def __init__(self,
                 source_xyz,
                 listener_xyz,
                 n_surfaces,
                 reflections,
                 transmissions,
                 delays,
                 start_directions,
                 end_directions = None):

        self.source_xyz = source_xyz
        self.listener_xyz = listener_xyz
        self.reflection_mask = gen_counts(reflections,n_surfaces)
        self.transmission_mask = gen_counts(transmissions,n_surfaces)
        self.delays = torch.tensor(delays)
        self.start_directions_normalized = torch.tensor(
            start_directions/np.linalg.norm(start_directions, axis=-1).reshape(-1, 1))

        if end_directions is not None:
            self.end_directions_normalized = torch.tensor(
                end_directions/np.linalg.norm(end_directions, axis=-1).reshape(-1, 1))
        else:
            self.end_directions_normalized = None

def get_listener(source_xyz, listener_xyz, surfaces, load_dir=None, load_num=None,
                 speed_of_sound=343, max_order=5,  parallel_surface_pairs=None, max_axial_order=50):
    """
    Function to get a ListenerLocation. If load_dir is provided, loads precomputed paths

    Parameters
    ----------
    source_xyz: (3,) array indicating the source's location
    listener_xyz: (3,) array indicating the listener's location
    surface: list of Surface, surfaces comprising room geometry
    load_dir: directory of precomputed paths, if None, traces from scratch.
    speed_of_sounds: in m/s
    max_order: maximum reflection order to trace to, if tracing from scratch.
    load_num: within the directory of precomputed paths, the index to load from.
    parallel_surface_pairs: list of list of int, surface pairs to do axial boosting, provided as indices in the 'surfaces' argument.
    max_axial_order: max reflection order for parallel surfaces    

    Returns
    -------
    ListenerLocation, characterizing the listener location in the room.
    """
    if load_dir is None: 
        # Tracing from Scratch
        reflections, transmissions, delays, start_directions, end_directions = (
            trace1.get_reflections_transmissions_and_delays(
            source=source_xyz, dest=listener_xyz, surfaces=surfaces, speed_of_sound=speed_of_sound,
            max_order=max_order,parallel_surface_pairs=parallel_surface_pairs, max_axial_order=max_axial_order)
        )

    else:
        # Loading precomputed paths
        print("Listener Loading From" + load_dir)
        reflections = np.load(os.path.join(load_dir,"reflections/"+str(load_num)+".npy"), allow_pickle=True)
        transmissions = np.load(os.path.join(load_dir, "transmissions/"+str(load_num)+".npy"), allow_pickle=True)
        delays = np.load(os.path.join(load_dir, "delays/"+str(load_num)+".npy"))
        start_directions = np.load(os.path.join(load_dir, "starts/"+str(load_num)+".npy"))
        end_directions = np.load(os.path.join(load_dir, "ends/"+str(load_num)+".npy"))

    L = ListenerLocation(
        source_xyz=source_xyz,
        listener_xyz=listener_xyz,
        n_surfaces=len(surfaces),
        reflections=reflections,
        transmissions=transmissions,
        delays=delays,
        start_directions = start_directions,
        end_directions = end_directions)

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
    """Generates a (n_paths, n_surfaces) 0-1 mask indicating reflections"""#?????se riflette più di una volta sulla stessa superficie il valore sarà >1?? si
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

###########################################################################

def initialize_directional_list(angular_sensitivities, signal_length, device):
    frequency_range_list = []
    for characteristic in angular_sensitivities:
        frequency_dict = dict()
        frequency_dict['frequency_range'] = characteristic['frequency_range']
        directional_responses = []
        used_angle = 180/(int(180/characteristic['angle']))
        azimuths = np.arange(0, 360, used_angle)
        elevations = np.arange(-90, 90+ used_angle, used_angle)
        for elevation in elevations:
            if (elevation == -90 or elevation == 90):
                direction_dict = dict()
                direction_dict['direction'] = [0, elevation]#-90 and +90 elevations are considered having azimuth=0
                direction_dict['response'] = torch.zeros(signal_length).to(device)
                directional_responses.append(direction_dict)
            else:   
                for azimuth in azimuths:
                    direction_dict = dict()
                    direction_dict['direction'] = [azimuth, elevation]
                    direction_dict['response'] = torch.zeros(signal_length).to(device)
                    directional_responses.append(direction_dict)

        frequency_dict['responses'] = directional_responses
        frequency_range_list.append(frequency_dict)

    return frequency_range_list

#############################################################################

###########################################################################

def initialize_directional_list_for_beampattern(angular_sensitivity, n_freq_samples, device):

    frequency_responses_list = []
    used_angle = 180/(int(180/angular_sensitivity))
    azimuths = np.arange(0, 360, used_angle)
    elevations = np.arange(-90, 90+ used_angle, used_angle)
    for elevation in elevations:
        if (elevation == -90 or elevation == 90):
            direction_dict = dict()
            direction_dict['angle'] = [0, elevation]#-90 and +90 elevations are considered having azimuth=0
            direction_dict['f_response'] = torch.zeros(n_freq_samples, dtype=torch.complex64).to(device)#############provo con complessi
            frequency_responses_list.append(direction_dict)
        else:   
            for azimuth in azimuths:
                direction_dict = dict()
                direction_dict['angle'] = [azimuth, elevation]
                direction_dict['f_response'] = torch.zeros(n_freq_samples, dtype=torch.complex64).to(device)
                frequency_responses_list.append(direction_dict)

    return frequency_responses_list

#############################################################################

'''
#####################################################################
#gives the key to insert a path into a dictionary of RIR_by_direction
#forward is [0,1,0], left is [-1,0,0]
def get_direction_key(azimuth, elevation):

    negative = elevation < 0
    elevation = abs(elevation)
    
    if negative:
        if elevation <= 7.5:
            suffix = "0,0"
        elif 7.5 <= elevation <  16.25:
            suffix = "-15,0"
        elif 16.25 <= elevation < 21.25:
            suffix = "-17,5"
        elif 21.25 <= elevation < 27.5:
            suffix = "-25,0"
        elif 27.5 <= elevation < 32.65:
            suffix = "-30,0"
        elif 32.65 <= elevation < 40.15:
            suffix = "-35,3"
        elif 40.15 <= elevation < 49.5:
            suffix = "-45,0"
        elif 49.5 <= elevation < 57:
            suffix = "-54,0"
        elif 57 <= elevation < 62.4:
            suffix = "-60,0"
        elif 62.4 <= elevation < 69.9:
            suffix = "-64,8"        
        elif 69.9 <= elevation < 78:
            suffix = "-75,0"
        elif elevation >= 78:
            suffix = "-81,0"
    else:
        if elevation <= 7.5:
            suffix = "0,0"
        elif 7.5 <= elevation <  16.25:
            suffix = "15,0"
        elif 16.25 <= elevation < 21.25:
            suffix = "17,5"
        elif 21.25 <= elevation < 27.5:
            suffix = "25,0"
        elif 27.5 <= elevation < 32.65:
            suffix = "30,0"
        elif 32.65 <= elevation < 40.15:
            suffix = "35,3"
        elif 40.15 <= elevation < 49.5:
            suffix = "45,0"
        elif 49.5 <= elevation < 57:
            suffix = "54,0"
        elif 57 <= elevation < 62.4:
            suffix = "60,0"
        elif 62.4 <= elevation < 69.9:
            suffix = "64,8"        
        elif 69.9 <= elevation < 82.5:
            suffix = "75,0"
        elif elevation >= 82.5:
            suffix = "90,0"

    azimuth = str(int(np.round(azimuth) % 360))
    key = "azi_" + azimuth + ",0_ele_" + suffix

    return key        
################################################################################
'''
'''
##################################################################
def butter_bandpass(lowcut, highcut, fs=48000, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a
    
###############################################################

def butter_bandpass_filter(data, lowcut, highcut, fs=48000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = torch.tensor(scipy.signal.lfilter(b, a, np.array(data.detach())))######################questo detach va bene?? 
    return y
'''
################################################################

def sinc_filter(cutoff, fs=48000, num_taps=513, device = 'cpu'):
    """Create a sinc filter kernel."""
    t = torch.arange(-(num_taps - 1) / 2, (num_taps - 1) / 2 + 1,  device=device)
    t = t / fs
    return torch.where(t == 0, 2 * cutoff, torch.sin(2 * np.pi * cutoff * t) / (np.pi * t))

#################################################################

def bandpass_filter(low_cutoff, high_cutoff, fs= 48000, num_taps=513, device = 'cpu'):
    """Create a bandpass filter using sinc function."""
    # Low-pass filter with cutoff = high_cutoff
    hlp = sinc_filter(high_cutoff, fs, num_taps,  device=device)
    # High-pass filter with cutoff = low_cutoff
    hhp = sinc_filter(low_cutoff, fs, num_taps,  device=device)
    # Bandpass is the difference between the two
    hbp = hlp - hhp
    # Apply a window function, e.g., Hamming window, to improve performance
    window = torch.hamming_window(num_taps, periodic=False,  device=device)
    hbp *= window
    return hbp
########################################################################

def apply_bandpass_filter(signal, low_cutoff, high_cutoff, fs = 48000, num_taps=513):
    """Creates a bandpass filter and applies it to a signal using convolution."""
    device = signal.device
    filter_kernel = bandpass_filter(low_cutoff, high_cutoff, fs, num_taps, device).to(device)###################forse ridondante
    # Add a batch dimension and channel dimension if necessary
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(1)

    # Ensure the filter kernel has the right dimensions
    filter_kernel = filter_kernel.unsqueeze(0).unsqueeze(0)
    padding = (num_taps - 1) // 2
    # Convolve the signal with the filter kernel
    filtered_signal = nn.functional.conv1d(signal, filter_kernel, padding=padding)
    return filtered_signal.squeeze()
################################################################################

#tentativo usare beam pattern per contributi direzionali di ogni path

###########################################################
def calculate_weights(azimuth_incoming, elevation_incoming, l_max):
    """
    Compute the weights W_{lm} for the direction of arrival of the signal.
    
    :param azimuth_incoming: Azimuth angle of the direction of arrival (in degrees).
    :param elevation_incoming: Elevation angle of the direction of arrival (in degrees).
    :param l_max: Maximum order to consider for the spheric harmonics.
    :return: Dictionary of weights W_{lm}.
    """
    azimuth_incoming = - azimuth_incoming
    phi_0 = np.deg2rad(azimuth_incoming)
    theta_0 = np.deg2rad(90 - elevation_incoming)

    bp_weights = {}
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_0, theta_0)
            bp_weights[(l, m)] = Y_lm
    
    return bp_weights

#################################################

###########################################################
def calculate_weights_all_orders(frequency, azimuth_incoming, elevation_incoming, bp_orders_cutoffs):
    """
    Compute the weights W_{lm} for the direction of arrival of the signal.
    
    :param azimuth_incoming: Azimuth angle of the direction of arrival (in degrees).
    :param elevation_incoming: Elevation angle of the direction of arrival (in degrees).
    :param l_max: Maximum order to consider for the spheric harmonics.
    :return: Dictionary of weights W_{lm}.
    """
    azimuth_incoming = - azimuth_incoming
    phi_0 = np.deg2rad(azimuth_incoming)
    theta_0 = np.deg2rad(90 - elevation_incoming)

    l_max = len(bp_orders_cutoffs)

    bp_weights = {}
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_0, theta_0)
            if l!=0:
                Y_lm *= sigmoid(frequency-bp_orders_cutoffs[l-1])
            bp_weights[(l, m)] = Y_lm
    
    return bp_weights

#################################################

#################################################
def beam_pattern(azimuth, elevation, bp_weights, l_max):
    """
    Compute beam pattern in a specific direction.
    
    :param azimuth: Azimuth angle (in degrees).
    :param elevation: Elevation angle (in degrees).
    :param weights: Dictionary of weights W_{lm}.
    :param l_max: Maximum order of the considered spheric harmonics.
    :return: Amplitude of the beam pattern in the specified direction.
    """
    azimuth=azimuth
    phi = np.deg2rad(azimuth)
    theta = np.deg2rad(90 - elevation)

    pattern = 0.0
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            pattern += bp_weights[(l, m)] * Y_lm
    
    return np.abs(pattern)

#####################################################

####################################################
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
#######################################################
