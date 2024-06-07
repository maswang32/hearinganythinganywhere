import numpy as np
import torch
import os
import scipy.io.wavfile
import plotly.graph_objects as go
import trace
import render
import config

HRIR_dataset_dir = config.HRIR_dataset_dir

def get_HRIR(azimuth, elevation):
    """
    Returns 2-channel (2x256) HRIR given an azimuth and elevation angle in degrees.
    Azimuth angles are anti-clockwise.
    """
    negative = elevation < 0
    elevation = abs(elevation)
    
    if negative:
        if elevation <= 7.5:
            suffix = "0,0.wav"
        elif 7.5 <= elevation <  16.25:
            suffix = "-15,0.wav"
        elif 16.25 <= elevation < 21.25:
            suffix = "-17,5.wav"
        elif 21.25 <= elevation < 27.5:
            suffix = "-25,0.wav"
        elif 27.5 <= elevation < 32.65:
            suffix = "-30,0.wav"
        elif 32.65 <= elevation < 40.15:
            suffix = "-35,3.wav"
        elif 40.15 <= elevation < 49.5:
            suffix = "-45,0.wav"
        elif 49.5 <= elevation < 57:
            suffix = "-54,0.wav"
        elif 57 <= elevation < 62.4:
            suffix = "-60,0.wav"
        elif 62.4 <= elevation < 69.9:
            suffix = "-64,8.wav"        
        elif 69.9 <= elevation < 78:
            suffix = "-75,0.wav"
        elif elevation >= 78:
            suffix = "-81,0.wav"
    else:
        if elevation <= 7.5:
            suffix = "0,0.wav"
        elif 7.5 <= elevation <  16.25:
            suffix = "15,0.wav"
        elif 16.25 <= elevation < 21.25:
            suffix = "17,5.wav"
        elif 21.25 <= elevation < 27.5:
            suffix = "25,0.wav"
        elif 27.5 <= elevation < 32.65:
            suffix = "30,0.wav"
        elif 32.65 <= elevation < 40.15:
            suffix = "35,3.wav"
        elif 40.15 <= elevation < 49.5:
            suffix = "45,0.wav"
        elif 49.5 <= elevation < 57:
            suffix = "54,0.wav"
        elif 57 <= elevation < 62.4:
            suffix = "60,0.wav"
        elif 62.4 <= elevation < 69.9:
            suffix = "64,8.wav"        
        elif 69.9 <= elevation < 82.5:
            suffix = "75,0.wav"
        elif elevation >= 82.5:
            suffix = "90,0.wav"

    azimuth = str(int(np.round(azimuth) % 360))
    path = os.path.join(HRIR_dataset_dir, "azi_" + azimuth + ",0_ele_" + suffix)
    _, hrir = scipy.io.wavfile.read(path)

    # 32 bit - convert to float
    hrir = hrir.T/2147483648
    return hrir

def compute_hrirs(incoming_listener_directions, listener_forward, listener_left):
    """
    Returns a stack of head-related IRs for each listener direction.

    Parameters
    ----------
    incoming_listener_directions: (P,3). Points towards the listener.
    listener_forward: (3,). Vector pointing in the forward direction of listener.
    listener_left: (3,). Vector pointing to the left of the listener.

    Returns
    -------
    (P,2,256) np.array of HRIRs
    """
    norms = np.linalg.norm(incoming_listener_directions, axis=-1).reshape(-1,1)
    incoming_listener_directions = -incoming_listener_directions/norms
    listener_forward = listener_forward/np.linalg.norm(listener_forward)

    #listener_left points left (right-handed coordinate system with listener_forward)
    listener_left = listener_left/np.linalg.norm(listener_left)

    #Make sure listener_forward and listener_left are orthogonal
    assert np.abs(np.dot(listener_forward, listener_left)) < 0.01

    listener_up = np.cross(listener_forward, listener_left)
    head_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

    #Compute Azimuths and Elevation
    head_coordinates = incoming_listener_directions @ head_basis
    azimuths = np.degrees(np.arctan2(head_coordinates[:, 1], head_coordinates[:, 0]))
    elevations = np.degrees(np.arctan(head_coordinates[:, 2]/np.linalg.norm(head_coordinates[:, 0:2],axis=-1)+1e-8))

    #Retrieve HRIRs
    h_rirs = np.zeros((incoming_listener_directions.shape[0], 2, 256))
    for i in range(incoming_listener_directions.shape[0]):
        h_rirs[i] = get_HRIR(azimuth=azimuths[i], elevation=elevations[i])

    return h_rirs


def render_binaural(R, source_xyz, source_axis_1, source_axis_2,
                       listener_xyz, listener_forward, listener_left,
                       surfaces, speed_of_sound=None,
                       load_dir=None, load_num=None,
                       parallel_surface_pairs=None, max_order=5, max_axial_order=50,
                       plot=False):

    """
    Renders binaural RIRs using our model.

    Parameters
    ----------
    R: Renderer
    source_xyz: (3,) location of source 
    source_axis_1: (3,) axis 1 of source coordinate system 
    source_axis_2: (3,) axis 2 of source coordinate system 
    listener_xyz: (3,) listener location
    listener_forward: (3,) listener forward
    listener_left: (3,) listener left
    surfaces: list of Surface (trace.Surface) comprising the room's geometry
    speed_of_sound: speed of sound
    load_dir: directory to load precomputed reflection paths, None if recomputing.
    load_num: index ({load_num}.npy) in load_dir corresponding to the location.
    plot: bool, if we plot the location of the source/listener/surfaces
    parallel_surface_pairs: list of list of int, surface pairs to do axial boosting, 
        given as indices in the 'surfaces' argument.
    max_order: max reflection order
    max_axial_order: max reflection order for parallel surfaces    

    Returns
    -------
    predicted binaural RIR, (T,)
    """

    L = render.get_listener(source_xyz=source_xyz,
                            listener_xyz=listener_xyz,
                            surfaces=surfaces,
                            speed_of_sound=speed_of_sound,
                            load_dir=load_dir,
                            load_num=load_num,
                            parallel_surface_pairs=parallel_surface_pairs,
                            max_order=max_order,
                            max_axial_order=max_axial_order)

    hrirs = torch.tensor(compute_hrirs(incoming_listener_directions=L.end_directions_normalized,
                                        listener_forward=listener_forward,
                                        listener_left=listener_left)).to(R.device)

    if plot:
        listener_up = np.cross(listener_forward ,listener_left)
        head_basis = np.stack((listener_forward, listener_forward, listener_up), axis=-1)

        fig = go.Figure()
        trace.plot_surfaces(fig, surfaces)
        trace.plot_points(fig, listener_xyz, "Listener")
        fig = trace.plot_points(fig, source_xyz, "Source")
    
        for i,vec in enumerate(head_basis):
            if i == 0:
                label="Listener Direction"
            elif i == 1:
                label = "Ear Axis"
            elif i == 2:
                label = "Top of Head"
            fig.add_trace(go.Scatter3d(x=[listener_xyz[0],listener_xyz[0]+vec[0]],
                                       y=[listener_xyz[1], listener_xyz[1]+vec[1]],
                                       z=[listener_xyz[2], listener_xyz[2]+vec[2]],
                                       mode='lines', name=label,line=dict(width=5)))
        fig.show()

    with torch.no_grad():
        rir = R.render_RIR(loc=L,
                           hrirs=hrirs,
                           source_axis_1=source_axis_1,
                           source_axis_2=source_axis_2).detach().cpu().numpy()
        return rir