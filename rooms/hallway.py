import trace as G
import numpy as np
import dataset


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

            
left_music_recs = np.load("/viscam/projects/audio_nerf/datasets/real/Hallway/HallwayBase/gt_binaural_left_recordings.npy")
right_music_recs = np.load("/viscam/projects/audio_nerf/datasets/real/Hallway/HallwayBase/gt_binaural_right_recordings.npy")
bin_music_recs = np.stack((left_music_recs,right_music_recs),axis=1)



#gt speaker - speaker_xyz = np.array([63*cm, (1042.1)*cm, 45*cm])
#gt trans speaker - [121*cm, (746+998.5)*cm, 45*cm]
BaseDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayBase",
    speaker_xyz = np.array([ 0.6870, 10.2452,  0.5367]), #error 20 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)


RotationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayRotation",
    speaker_xyz = np.array([ 0.5091, 10.4333,  0.5464]), #error 15.5 cm
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)


TranslationDataset = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayTranslation",
    speaker_xyz = np.array([ 1.2120, 17.1969,  0.5444]), #26.5 cm error
    all_surfaces = walls,
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)

PanelDataset1 = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayPanel1",
    speaker_xyz = np.array([ 0.6549, 10.2356,  0.4618]), #19 cm error
    all_surfaces = walls + [panel_1],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)


PanelDataset2 = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayPanel2",
    speaker_xyz = np.array([ 0.5002, 10.1438,  0.3348]), #32 cm error
    all_surfaces = walls + [panel_2],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)

PanelDataset3 = dataset.Dataset(
    load_dir = "/viscam/projects/audio_nerf/datasets/real2/Hallway/hallwayPanel3",
    speaker_xyz = np.array([0.4746, 9.9688, 0.2613]), # 51 cm error
    all_surfaces =  walls + [panel_2, config_3_panel_1],
    speed_of_sound = speed_of_sound,
    default_binaural_listener_forward = np.array([0,1,0]),
    default_binaural_listener_left = np.array([-1,0,0]),
    parallel_surface_pairs=[[2,3], [4,5]]
)