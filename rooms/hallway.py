import numpy as np
import config
import trace1 as G
import rooms.dataset as dataset

"""
Importing this document automatically loads data from the Hallway dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces
"""
inches = 0.0254
cm = 0.01
max_x = (1.532+1.526)/2
max_y = (18.042+18.154)/2
max_z = (2.746+2.765+2.766+2.756+2.749)/5

close_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, max_z],
                                  [max_x, 0, 0]]))

far_wall = G.Surface(np.array([[0, max_y, 0],
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

walls = [close_wall, far_wall, floor, ceiling, left_wall, right_wall]


panel_1_x_1 = 112*cm
panel_1_y_1 = 784.5*cm
panel_1_x_2 = 40*cm
panel_1_y_2 = 841.25*cm
panel_1_z_1 = 14.8*cm
panel_1_z_2 = 185.5*cm

panel_1 = G.Surface(np.array([[panel_1_x_1, panel_1_y_1, panel_1_z_2],
                              [panel_1_x_1, panel_1_y_1, panel_1_z_1],                               
                              [panel_1_x_2, panel_1_y_2, panel_1_z_2]]))
                                
panel_2_y_1 = (906.4+213.2)*cm
panel_2_y_2 = (906.4+213.2)*cm
panel_2_x_1 = 27.7*cm
panel_2_x_2 = 119.5*cm
panel_2_z_1 = 14.8*cm
panel_2_z_2 = 185.5*cm

panel_2 = G.Surface(np.array([[panel_2_x_1, panel_2_y_1, panel_2_z_2],
                              [panel_2_x_1, panel_2_y_1, panel_2_z_1],                               
                              [panel_2_x_2, panel_2_y_2, panel_2_z_2]]))
                                

config_3_panel_1_y_1 = 906.4*cm # This is a different panel (white edges)
config_3_panel_1_y_2 = 906.4*cm
config_3_panel_1_x_1 = 0
config_3_panel_1_x_2 = 110.5*cm
config_3_panel_1_z_1 = 17*cm
config_3_panel_1_z_2 = 187*cm

config_3_panel_1 = G.Surface(np.array([[config_3_panel_1_x_1, config_3_panel_1_y_1, config_3_panel_1_z_2],
                                       [config_3_panel_1_x_1, config_3_panel_1_y_1, config_3_panel_1_z_1],                               
                                       [config_3_panel_1_x_2, config_3_panel_1_y_2, config_3_panel_1_z_2]]))

            
"""
Train, Test indices, Making Dataset class instances
"""

# Default tracing orders
max_order = 5
max_axial_order = 50

#gt speaker - speaker_xyz = np.array([63*cm, (1042.1)*cm, 45*cm])
base_train_indices = [5, 58, 99, 148, 203, 241, 296, 342, 384, 441, 482, 535]
BaseDataset = dataset.Dataset(
    load_dir = config.hallwayBase_path,
    speaker_xyz = np.array([ 0.6870, 10.2452,  0.5367]), #error 20 cm from manual measurement
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = base_train_indices, #Yea!
    valid_indices = dataset.compute_complement_indices(base_train_indices + list(np.arange(288)*2), 576)[::2],
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 576

)


alt_config_train_indices = [5, 10, 15, 16, 35, 25, 44, 42, 48, 57, 62, 67]
alt_valid_indices = dataset.compute_complement_indices(alt_config_train_indices, 72)[::2]

RotationDataset = dataset.Dataset(
    load_dir = config.hallwayRotation_path,
    speaker_xyz = np.array([ 0.5091, 10.4333,  0.5464]), #error 15.5 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = alt_config_train_indices,
    valid_indices = alt_valid_indices,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 72
)

#gt translated speaker - [121*cm, (746+998.5)*cm, 45*cm]
TranslationDataset = dataset.Dataset(
    load_dir = config.hallwayTranslation_path,
    speaker_xyz = np.array([ 1.2120, 17.1969,  0.5444]), #26.5 cm error
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = alt_config_train_indices,
    valid_indices = alt_valid_indices,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 72
)

PanelDataset1 = dataset.Dataset(
    load_dir = config.hallwayPanel1_path,
    speaker_xyz = np.array([ 0.6549, 10.2356,  0.4618]), #19 cm error
    all_surfaces = walls + [panel_1],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = alt_config_train_indices,
    valid_indices = alt_valid_indices,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 72
)


PanelDataset2 = dataset.Dataset(
    load_dir = config.hallwayPanel2_path,
    speaker_xyz = np.array([ 0.5002, 10.1438,  0.3348]), #32 cm error
    all_surfaces = walls + [panel_2],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = alt_config_train_indices,
    valid_indices = alt_valid_indices,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 72
)

PanelDataset3 = dataset.Dataset(
    load_dir = config.hallwayPanel3_path,
    speaker_xyz = np.array([0.4746, 9.9688, 0.2613]), # 51 cm error
    all_surfaces =  walls + [panel_2, config_3_panel_1],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]],
    train_indices = alt_config_train_indices,
    valid_indices = alt_valid_indices,
    max_order = max_order,
    max_axial_order = max_axial_order,
    n_data = 72
)