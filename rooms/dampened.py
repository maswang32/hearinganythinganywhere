import trace as G
import numpy as np
import dataset


"""
Importing this document automatically loads data from the treated room dataset
"""

cm = 0.01
speed_of_sound = 343.2171717171717
speed_of_sound = 343

"""
Locations of all surfaces
"""
max_x = 485*cm
max_y = 519.5*cm
max_z = 273.1*cm

rear_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, max_z],
                                  [max_x, 0, 0]]))

front_wall = G.Surface(np.array([[0, max_y, 0],
                                  [max_x, max_y, 0],
                                  [0, max_y, max_z]]))


floor = G.Surface(np.array([[0, 0, 0],
                                [max_x, 0, 0],
                                [0, max_y, 0]]))


ceiling = G.Surface(np.array([[0, 0, max_z],
                                [0, max_y, max_z],
                               [max_x, 0, max_z]]))

left_wall = G.Surface(np.array([[0, 0, 0],
                                [0, max_y, 0],
                                 [0, 0, max_z]]))

right_wall = G.Surface(np.array([[max_x, 0, 0],[max_x, 0, max_z],                               
                                [max_x, max_y, 0]]))


walls = [rear_wall, front_wall, floor, ceiling, left_wall, right_wall]

surfaces_to_plot = [rear_wall, floor,  left_wall]


parallel_surface_pairs = [[2,3]]

panel_y = (225.5-1.3)*cm
panel_x1 = 191*cm
panel_x2 = 287.5*cm
panel_z1 = 14.5*cm
panel_z2 = 185.5*cm

panel = G.Surface(np.array([[panel_x1, panel_y, panel_z2],[panel_x1, panel_y, panel_z1],                               
                                [panel_x2, panel_y, panel_z2]]))

#Compute Indices
#speaker_xyz = np.array([2.4051, 2.5638, 1.2334]), #Estimated 10/31 using 276 points, TOA. Error = 4 cm.
BaseDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Dampened/dampenedBase",
    speaker_xyz = np.array([2.4542, 2.4981, 1.2654]), #Estimated 11/14, error 11 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)

#speaker_xyz = np.array([2.4612, 2.6852, 1.2366]),  #Estimated 10/31 using 120 points, TOA. Error = 11 cm.
RotationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Dampened/dampenedRotation/",
    speaker_xyz=np.array([2.4595, 2.6748, 1.0659]), #Estimated 11/14, error = 17 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)


#speaker_xyz = np.array([1.12, 0.79, 1.2366]),  #Measured, Z borrowed from above
TranslationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Dampened/dampenedTranslation",
    speaker_xyz=np.array([1.2621, 0.5605, 1.2404]), #error 27 cm, 11/14
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)




#speaker_xyz = np.array([2.4051, 2.5638, 1.2334])
PanelDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Dampened/dampenedPanel",
    speaker_xyz = np.array([2.4052, 2.5292, 1.3726]), # 11/4 - error  = 18 cm.
    all_surfaces = walls+[panel],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)