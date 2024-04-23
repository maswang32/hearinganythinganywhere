import trace as G
import numpy as np
import dataset


speed_of_sound = 343 #approximation



all_surfaces = []

# Apex
x_apex = 3.266
z_apex = 6.086
z_panel = 3.05
x_to_door = 5.387


# x_0_wall
y_max = 13.014
x_0_wall_z = 2.265
x_0_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, x_0_wall_z],
                                  [0, y_max, 0]]))

all_surfaces.append(x_0_wall)



# y_0_wall
x_max = 8.374
y_0_wall_z = 2.644
y_0_wall = G.Surface(np.array([[0, 0, 0], 
                                  [0, 0, y_0_wall_z],
                                  [x_max, 0, 0]]))

all_surfaces.append(y_0_wall)


# y_0_wall_panel
y_0_wall_panel = G.Surface(np.array([[0, 0, z_panel], 
                                  [0, 0, z_apex],
                                  [x_max, 0, z_panel]]))

all_surfaces.append(y_0_wall_panel)

# y_max_wall_x0
y_max_wall_x0 = G.Surface(np.array([[0, y_max, 0], 
                                  [0, y_max, z_panel],
                                  [x_max, y_max, 0]]))
all_surfaces.append(y_max_wall_x0)


# y_max_wall_panel
x_to_door = 5.387
y_max_wall_panel = G.Surface(np.array([[0, y_max, z_panel], 
                                  [0, y_max, z_apex],
                                  [x_max, y_max, z_panel]]))
all_surfaces.append(y_max_wall_panel)


# y_max_wall_x_max
y_door = 12.026
z_door = 2.945
y_door_wall_x_max = G.Surface(np.array([[x_to_door, y_door, 0], 
                                  [x_to_door, y_door, z_door],
                                  [x_max, y_door, 0]]))
all_surfaces.append(y_door_wall_x_max)


# glass_wall
glass_wall = G.Surface(np.array([[x_to_door, y_door, 0], 
                                  [x_to_door, y_door, z_door],
                                  [x_to_door, y_max, 0]]))
all_surfaces.append(glass_wall)


# xmax_wall
x_max_wall = G.Surface(np.array([[x_max, y_door, 0], 
                                [x_max, 0, 0], 
                                  [x_max, y_door, z_panel]]))
all_surfaces.append(x_max_wall)


"""
Overhang
"""
y_overhang = 0.795
z_1overhang = 2.644
z_2overhang = 3.807

# overhang_facing_down
overhang_facing_down = G.Surface(np.array([[0, 0, z_1overhang], 
                                [x_max, 0, z_1overhang], 
                                  [0, y_overhang, z_1overhang]]))

all_surfaces.append(overhang_facing_down)

# overhang_facing_forward
overhang_facing_forward = G.Surface(np.array([[0, y_overhang, z_1overhang], 
                                [0, y_overhang, z_2overhang], 
                                  [x_max, y_overhang, z_1overhang]]))

all_surfaces.append(overhang_facing_forward)

# overhang_facing_up
overhang_facing_up = G.Surface(np.array([[0, 0, z_2overhang], 
                                [0, y_overhang, z_2overhang], 
                                  [x_max, 0, z_2overhang]]))

all_surfaces.append(overhang_facing_up)




# pillars
x_pillar1 = 4.002
x_width1 = 0.646
y_width1 = 0.649
y_pillar1 = 4.488
z_pillar = 3.055


# pillar1
pillar1_1 = G.Surface(np.array([[x_pillar1, y_pillar1, 0], 
                                [x_pillar1, y_pillar1, z_pillar], 
                                  [x_pillar1+x_width1, y_pillar1, 0]]))  
all_surfaces.append(pillar1_1)

pillar1_2 = G.Surface(np.array([[x_pillar1, y_pillar1+y_width1, 0], 
                                [x_pillar1, y_pillar1+y_width1, z_pillar], 
                                  [x_pillar1+x_width1, y_pillar1+y_width1, 0]]))  
all_surfaces.append(pillar1_2)

pillar1_3 = G.Surface(np.array([[x_pillar1, y_pillar1, 0], 
                                [x_pillar1, y_pillar1, z_pillar], 
                                  [x_pillar1, y_pillar1+y_width1, 0]]))  
all_surfaces.append(pillar1_3)

pillar1_4 = G.Surface(np.array([[x_pillar1+x_width1, y_pillar1, 0], 
                                [x_pillar1+x_width1, y_pillar1, z_pillar], 
                                  [x_pillar1+x_width1, y_pillar1+y_width1, 0]]))  
all_surfaces.append(pillar1_4)



x_pillar2 = 4.004
x_width2 = 0.654
y_pillar2_1 = y_pillar1+y_width1
y_pillar2_2 = 7.178
y_width2 = 0.937

# pillar2
pillar2_1 = G.Surface(np.array([[x_pillar2, y_pillar2_1, 0], 
                                [x_pillar2, y_pillar2_2, z_pillar], 
                                  [x_pillar2+x_width2, y_pillar2_1, 0]]))  
all_surfaces.append(pillar2_1)

pillar2_2 = G.Surface(np.array([[x_pillar2, y_pillar2_1+y_width2, 0], 
                                [x_pillar2, y_pillar2_2+y_width2, z_pillar], 
                                  [x_pillar2+x_width2, y_pillar2_1+y_width2, 0]]))  
all_surfaces.append(pillar2_2)

pillar2_3 = G.Surface(np.array([[x_pillar2, y_pillar2_1, 0], 
                                [x_pillar2, y_pillar2_2, z_pillar], 
                                  [x_pillar2, y_pillar2_1+y_width2, 0]]))  
all_surfaces.append(pillar2_3)

pillar2_4 = G.Surface(np.array([[x_pillar2+x_width2, y_pillar2_1, 0], 
                                [x_pillar2+x_width2, y_pillar2_2, z_pillar], 
                                  [x_pillar2+x_width2, y_pillar2_1+y_width2, 0]]))  
all_surfaces.append(pillar2_4)


# pillar3
x_pillar3 = 3.997
x_width3 = 0.647
y_pillar3 = 10.249
y_width3 = 0.649

pillar3_1 = G.Surface(np.array([[x_pillar3, y_pillar3, 0], 
                                [x_pillar3, y_pillar3, z_pillar], 
                                  [x_pillar3+x_width3, y_pillar3, 0]]))  
all_surfaces.append(pillar3_1)

pillar3_2 = G.Surface(np.array([[x_pillar3, y_pillar3+y_width3, 0], 
                                [x_pillar3, y_pillar3+y_width3, z_pillar], 
                                  [x_pillar3+x_width3, y_pillar3+y_width3, 0]]))  
all_surfaces.append(pillar3_2)

pillar3_3 = G.Surface(np.array([[x_pillar3, y_pillar3, 0], 
                                [x_pillar3, y_pillar3, z_pillar], 
                                  [x_pillar3, y_pillar3+y_width3, 0]]))  
all_surfaces.append(pillar3_3)

pillar3_4 = G.Surface(np.array([[x_pillar3+x_width3, y_pillar3, 0], 
                                [x_pillar3+x_width3, y_pillar3, z_pillar], 
                                  [x_pillar3+x_width3, y_pillar3+y_width3, 0]]))  
all_surfaces.append(pillar3_4)



# max_x_panels
z_panel2 = 4.389 
x_max_wall = G.Surface(np.array([[x_max, 0, z_panel], 
                                [x_max, y_max, z_panel], 
                                  [x_max, 0, z_panel2]]))
all_surfaces.append(x_max_wall)


# slant
z_slant2 = 2.942
x_slant = 0.920
slant = G.Surface(np.array([[0, 0, x_0_wall_z], 
                                [0, y_max, x_0_wall_z], 
                                  [x_slant, 0, z_slant2]]))
all_surfaces.append(slant)

# above_slant
z_above_slant = 3.309
above_slant = G.Surface(np.array([[x_slant, 0, z_slant2], 
                                [x_slant, y_max, z_slant2], 
                                  [x_slant, 0, z_above_slant]]))
all_surfaces.append(above_slant)

# floor
floor = G.Surface(np.array([[0, 0, 0], 
                                [0, y_max, 0], 
                                  [x_max, 0, 0]]))
all_surfaces.append(floor)


# door_top
door_top = G.Surface(np.array([[x_to_door, y_max, z_door], 
                                [x_max, y_max, z_door], 
                                  [x_to_door, y_door, z_door]]))
all_surfaces.append(door_top)



# ceiling_x0
ceiling_x0 = G.Surface(np.array([[x_slant, 0, z_above_slant], 
                                [x_slant, y_max, z_above_slant], 
                                  [x_apex, 0, z_apex]]))
all_surfaces.append(ceiling_x0)

# ceiling_x_max
ceiling_x_max = G.Surface(np.array([[x_max, 0, z_panel2], 
                                [x_max, y_max, z_panel2], 
                                  [x_apex, 0, z_apex]]))
all_surfaces.append(ceiling_x_max)


# table
x_width_table = 0.761
y_table = 10.681
z_table = 0.736

table = G.Surface(np.array([[x_max, 0, z_table], 
                                [x_max, y_table, z_table], 
                                  [x_max-x_width_table, 0, z_table]]))

all_surfaces.append(table)


# triangle_table
middle_table_z = 0.906
middle_table_1 =  G.Surface(np.array([[2.927, 2.955, middle_table_z], 
                                [5.613, 4.378, middle_table_z], 
                                  [x_pillar1+0.5*x_width2, 5.444,middle_table_z]]), parallelogram=False)

middle_table_2 =  G.Surface(np.array([[5.614, 7.738, middle_table_z], 
                                [3.217, 7.738, middle_table_z], 
                                  [x_pillar1+0.5*x_width2, 5.444,middle_table_z]]), parallelogram=False)
                               
all_surfaces.append(middle_table_1)
all_surfaces.append(middle_table_2)
parallel_surface_pairs = [[0,7],[1,3],[1,5]]

plot_surfaces = all_surfaces



BaseDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Complex/complexBase",
    speaker_xyz = np.array([ 2.8377, 10.1228,  1.1539]), #Error - 32 cm
    all_surfaces = all_surfaces,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)

RotationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Complex/complexRotation",
    speaker_xyz = np.array([2.762,10.245,0.90]),  #gt
    all_surfaces = all_surfaces,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)

# speaker_xyz = np.array([6.237, 8.180, 0.90])
TranslationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Complex/complexTranslation",
    speaker_xyz = np.array([6.237, 8.180, 0.90]), #gt
    all_surfaces = all_surfaces,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs = parallel_surface_pairs
)