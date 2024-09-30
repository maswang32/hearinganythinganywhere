import numpy as np
import rooms.dataset as dataset

import open3d as o3d
import surface_creation_utilities as S


"""
Importing this document automatically loads data from the classroom dataset
"""

speed_of_sound = 343

"""
Locations of all surfaces in Meters
"""
d = o3d.data.LivingRoomPointClouds()
pcd = o3d.io.read_point_cloud(d.paths[30])
#base_surfaces = S.get_surfaces_from_point_cloud(pcd)
base_surfaces = S.get_surfaces_from_point_cloud_with_optimization(pcd, cut_impurity=0.05, steps_per_side=11)

"""
Train and Test Split
"""

train_indices = np.arange(12)*(57)
valid_indices = dataset.compute_complement_indices(list(train_indices) + list(np.arange(315)*2), 630)[::2]
