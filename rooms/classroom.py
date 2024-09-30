import numpy as np
import config
import trace1 as G
import rooms.dataset as dataset


"""
Importing this document automatically loads data from the classroom dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""
inches = 0.0254
max_x = 7.1247
max_y = 7.9248
max_z = 2.7432

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

left_table =G.Surface(np.array([[0, 96*inches, 29*inches],
                                     [30*inches, 96*inches, 28.25*inches],
                                     [0, max_y, 29*inches]]))

right_table = G.Surface(np.array([[max_x, 23.75*inches, 29*inches],
                                       [max_x, max_y, 29*inches],
                                       [max_x - 30*inches, 23.75*inches, 28.25*inches]]))

middle_table = G.Surface(np.array([[4.474256, 89*inches, 29*inches],
                                     [4.474256, max_y, 29*inches],
                                        [2.935758, 89*inches, 28.25*inches]]))
###########################
added_panel = G.Surface(np.array([[1, 5.5, 0.1],
                                     [1, 5.5, 2],
                                        [5, 5.5, 0.1]]))
##avrà le stesse caratteristiche riflettive di middle_table
###########################

walls = [rear_wall, front_wall, floor, ceiling, left_wall, right_wall]
tables = [left_table, right_table, middle_table]

base_surfaces = walls+tables


"""
Train and Test Split
"""

train_indices = np.arange(12)*(57)
valid_indices = dataset.compute_complement_indices(list(train_indices) + list(np.arange(315)*2), 630)[::2]


#Speaker xyz estimated from 12-point TOA, inside speaker, 8.5cm away from manual measurement.
BaseDataset = dataset.Dataset(
   load_dir = config.classroomBase_path,
   speaker_xyz= np.array([3.5838, 5.7230, 1.2294]), 
   all_surfaces = base_surfaces,
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5]],
   train_indices = train_indices,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   n_data = 630
)

####################################
AddedPanelDataset = dataset.Dataset(
   load_dir = config.classroomBase_path,
   speaker_xyz= np.array([3.5838, 5.7230, 1.2294]), 
   all_surfaces = base_surfaces + [added_panel],
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5], [0,9], [1,9]],
   train_indices = train_indices,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   n_data = 630
)
###################################