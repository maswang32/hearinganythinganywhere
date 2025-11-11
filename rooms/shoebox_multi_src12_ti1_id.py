import numpy as np
import config
import trace as G
import rooms.dataset as dataset


"""
Importing this document automatically loads data from the shoebox dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""
max_x = 6.0
max_y = 11.75
max_z = 2.8

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

base_surfaces = walls


"""
Train and Test Split
"""

local_train_indices = [[71], [71], [71], [71], [71], [71], [71], [71], [71], [71], [71], [71]] # 1 Index, L01, L06, L11, L16, L21, L26, L31, L36, L41, L46, L51, L56 are active

# Flatten active source_xyz, load_dir and train_indices lists based on active train_indices
source_xyzs = [
    np.array([3.1215, 8.6979, 1.45]),
    np.array([2.383, 8.458, 1.45 ]),
    np.array([1.8635, 7.881, 1.45]),
    np.array([1.7021, 7.1215, 1.45]),
    np.array([1.942, 6.383, 1.45]),
    np.array([2.519, 5.8635, 1.45]),
    np.array([3.2785, 5.7021, 1.45]),
    np.array([4.017, 5.942, 1.45]),
    np.array([4.5365, 6.519,  1.45]),
    np.array([4.6979, 7.2785, 1.45]),
    np.array([4.458, 8.017, 1.45]),
    np.array([3.881, 8.5365, 1.45])
]

multi_load_dirs = [
    config.shoebox_L01_path,
    config.shoebox_L06_path,
    config.shoebox_L11_path,
    config.shoebox_L16_path,
    config.shoebox_L21_path,
    config.shoebox_L26_path,
    config.shoebox_L31_path,
    config.shoebox_L36_path,
    config.shoebox_L41_path,
    config.shoebox_L46_path,
    config.shoebox_L51_path,
    config.shoebox_L56_path,
]


# Compute global training indices based on local sublists
all_train_indices_flat = []
n_data_per_source = 128

for source_idx, sublist in enumerate(local_train_indices):
    offset = source_idx * n_data_per_source
    all_train_indices_flat.extend([idx + offset for idx in sublist])
    
# Compute valid indices
valid_indices = [idx for idx in dataset.compute_complement_indices(all_train_indices_flat, 1536) if (0 <= idx < 128 or 256 <= idx < 384 or 512 <= idx < 640 or 768 <= idx < 896 or 1024 <= idx < 1152 or 1280 <= idx < 1408)]

BaseDataset = dataset.Dataset(
   load_dir = multi_load_dirs,
   speaker_xyz = np.stack(source_xyzs),
   all_surfaces = base_surfaces,
   speed_of_sound = speed_of_sound,
   default_binaural_listener_forward = np.array([0,1,0]),
   default_binaural_listener_left = np.array([-1,0,0]),
   parallel_surface_pairs=[[0,1], [2,3], [4,5]],
   train_indices = all_train_indices_flat,
   valid_indices = valid_indices,
   max_order = 5,
   max_axial_order = 10,
   #n_data_per_source = n_data_per_source,
   n_data = 1536
)

print(f"Active Sources: {len(source_xyzs)}, Loaded Sources: {[source_xyz.tolist() for source_xyz in source_xyzs]}")

