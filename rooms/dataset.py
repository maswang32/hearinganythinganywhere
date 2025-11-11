import os
import numpy as np


def compute_complement_indices(indices, n_data):
    """Given a list of indices and number of total datapoints, computes complement indices"""
    comp_indices = []
    for i in range(n_data):
        if i not in indices:
                comp_indices.append(i)

    return comp_indices


"""
Determining Training/Valid/Testing Indices for everything (Messy)
"""

class Dataset:

    """
    Class for a subdataset (e.g., classroom base dataset)

    Constructor Parameters
    ----------------------
    load_dir: where the files for the dataset are located
    speaker_xyz: (3,) array, where speaker is in the room setup
    all_surfaces: list of Surface - surfaces definining room's geometry
    speed_of_sound: in m/s
    default_binaural_listener_forward: (3,) direction the binaural mic is facing
    default_binaural_listener_left: (3,) points left out from the binaural mic
    max_order: default reflection order for tracing this dataset
    max_axial_order: default reflection order for parallel walls
    """
    def __init__(self,
                load_dir,
                speaker_xyz,
                all_surfaces,
                speed_of_sound,
                default_binaural_listener_forward,
                default_binaural_listener_left,
                parallel_surface_pairs,
                train_indices,
                valid_indices,
                max_order,
                max_axial_order,
                n_data):

        #More stuff
        self.speaker_xyz = speaker_xyz
        self.all_surfaces = all_surfaces
        self.speed_of_sound = speed_of_sound
        self.default_binaural_listener_forward = default_binaural_listener_forward
        self.default_binaural_listener_left = default_binaural_listener_left
        self.parallel_surface_pairs = parallel_surface_pairs
        self.load_dir = load_dir
        #self.n_data_per_source = n_data_per_source

        #indices
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = compute_complement_indices( list(self.train_indices)+list(self.valid_indices), n_data)

        # Default max order and axial order
        self.max_order = max_order
        self.max_axial_order = max_axial_order

    def load_data(self):
        if isinstance(self.load_dir, list):
            # Multi-Source: concat xyzs, RIRs, mic_numbers
            xyzs_list = []
            RIRs_list = []
            mic_numbers_list = []
            # Optional
            # music_list = []
            # music_dls_list = []
            # bin_music_dls_list = []
            # bin_xyzs_list = []
            # bin_RIRs_list = []
            # bin_music_list = []
            
            for ld in self.load_dir:
                xyzs_list.append(np.load(os.path.join(ld, "xyzs.npy")))
                RIRs_list.append(np.load(os.path.join(ld, "RIRs.npy")))
                mic_numbers_list.append(np.load(os.path.join(ld, "mic_numbers.npy")))
                # music_list.append(np.load(os.path.join(ld, "music.npy"), mmap_mode='r'))
                # music_dls_list.append(np.load(os.path.join(ld, "music_dls.npy"), mmap_mode='r'))
                # bin_music_dls_list.append(np.load(os.path.join(ld, "bin_music_dls.npy"), mmap_mode='r'))
                # bin_xyzs_list.append(np.load(os.path.join(ld, "bin_xyzs.npy"), mmap_mode='r'))
                # bin_RIRs_list.append(np.load(os.path.join(ld, "bin_RIRs.npy"), mmap_mode='r'))
                # bin_music_list.append(np.load(os.path.join(ld, "bin_music.npy"), mmap_mode='r'))

            self.xyzs = np.concatenate(xyzs_list, axis=0)
            self.RIRs = np.concatenate(RIRs_list, axis=0)
            self.mic_numbers = np.concatenate(mic_numbers_list, axis=0)
            # self.music = np.concatenate(music_list, axis=0)
            # self.music_dls = np.concatenate(music_dls_list, axis=0)
            # self.bin_music_dls = np.concatenate(bin_music_dls_list, axis=0)
            # self.bin_xyzs = np.concatenate(bin_xyzs_list, axis=0)
            # self.bin_RIRs = np.concatenate(bin_RIRs_list, axis=0)
            # self.bin_music = np.concatenate(bin_music_list, axis=0)
            #print(f"Loaded xyzs: {self.xyzs.shape}, RIRs: {self.RIRs.shape}, mic_numbers: {self.mic_numbers.shape}")
            print("Loading RIRs and xyzs from multiple source directories:")
            for i, ld in enumerate(self.load_dir):
                print(f"  - {ld}")
            print(f"Loaded {self.RIRs.shape[0]} RIRs and {self.xyzs.shape[0]} xyzs.")

        else:
            # Single-Source: load single files
            self.xyzs = np.load(os.path.join(self.load_dir, "xyzs.npy"))
            self.RIRs = np.load(os.path.join(self.load_dir, "RIRs.npy"))
            # self.music = np.load(os.path.join(self.load_dir, "music.npy"), mmap_mode='r')
            # self.music_dls = np.load(os.path.join(self.load_dir, "music_dls.npy"), mmap_mode='r')
            # self.bin_music_dls = np.load(os.path.join(self.load_dir, "bin_music_dls.npy"), mmap_mode='r')
            # self.bin_xyzs = np.load(os.path.join(self.load_dir, "bin_xyzs.npy"), mmap_mode='r')
            # self.bin_RIRs = np.load(os.path.join(self.load_dir, "bin_RIRs.npy"), mmap_mode='r')
            # self.bin_music = np.load(os.path.join(self.load_dir, "bin_music.npy"), mmap_mode='r')
            self.mic_numbers = np.load(os.path.join(self.load_dir, "mic_numbers.npy"))
            #print(f"Loaded xyzs: {self.xyzs.shape}, RIRs: {self.RIRs.shape}, mic_numbers: {self.mic_numbers.shape}")
            print(f"Loading RIRs and xyzs from {self.load_dir}")
            print(f"Loaded {self.RIRs.shape[0]} RIRs and {self.xyzs.shape[0]} xyzs from {self.load_dir}")
            
all_datasets = ["classroomBase", "dampenedBase", "dampenedRotation",
 "dampenedTranslation", "dampenedPanel", "hallwayBase", "hallwayRotation",
 "hallwayTranslation","hallwayPanel1","hallwayPanel2","hallwayPanel3",
 "complexBase","complexRotation","complexTranslation","room112_simple", "room112_precise", "vilnius1", "vilnius2", "mprir_s5", "mprir_s6", "mprir_s6_precise", "shoebox_offset_00", "shoebox_offset_01", "shoebox_offset_02", "shoebox_offset_03", "shoebox_offset_04", "shoebox_offset_05", "shoebox_offset_06", "shoebox_offset_07", "shoebox_offset_08", "shoebox_offset_09", "shoebox_offset_10", "shoebox_offset_11", "shoebox_offset_12", "shoebox_offset_13", "shoebox_offset_14", "shoebox_offset_15", "shoebox_offset_16", "shoebox_offset_17", "shoebox_offset_18", "shoebox_offset_19", "shoebox_offset_20", "shoebox_offset_21", "shoebox_offset_22", "shoebox_offset_23", "shoebox_offset_24", "shoebox_offset_25", "shoebox_offset_26", "shoebox_offset_27", "shoebox_offset_28", "shoebox_offset_29", "shoebox_offset_30", "shoebox_offset_31", "shoebox_offset_32", "shoebox_offset_33" "shoebox_offset_34",  "shoebox_offset_29_h280", "shoebox_offset_29_h320", "shoebox_L01", "shoebox_L11", "shoebox_L21", "shoebox_L31", "shoebox_L41", "shoebox_L51", "shoebox_L06", "shoebox_L16", "shoebox_L26", "shoebox_L36", "shoebox_L46", "shoebox_L56", "shoebox_multi_src4", "shoebox_multi_src6", "vilnius_multi_1", "vilnius_multi_2", "vilnius_multi_3", "vilnius_multi_4", "vilnius_multi_5", "vilnius_multi_6", "vilnius_multi_7", "vilnius_multi_8", "shoebox_L01_ti1", "shoebox_L01_ti2", "shoebox_L01_ti4", "shoebox_L01_ti8", "shoebox_L01_ti24", "shoebox_multi_src1_ti1", "shoebox_multi_src2_ti1_id", "shoebox_multi_src1_ti2", "shoebox_multi_src1_ti6_id", "shoebox_multi_src1_ti12_id", "shoebox_multi_src1_ti24_id", "shoebox_multi_src2_ti1_diff", "shoebox_multi_src2_ti2_id", "shoebox_multi_src2_ti2_diff", "shoebox_multi_src2_ti3_id", "shoebox_multi_src2_ti3_diff", "shoebox_multi_src2_ti4_id", "shoebox_multi_src2_ti4_diff", "shoebox_multi_src2_ti6_id", "shoebox_multi_src2_ti6_diff", "shoebox_multi_src2_ti12_id", "shoebox_multi_src2_ti12_diff", "shoebox_multi_src3_ti1_id", "shoebox_multi_src3_ti1_diff", "shoebox_multi_src3_ti4_id", "shoebox_multi_src3_ti4_diff", "shoebox_multi_src3_ti6_id", "shoebox_multi_src3_ti6_diff", "shoebox_multi_src3_ti3_id", "shoebox_multi_src3_ti3_diff", "shoebox_multi_src3_ti2_id", "shoebox_multi_src3_ti2_diff", "shoebox_multi_src4_ti1_id", "shoebox_multi_src4_ti1_diff", "shoebox_multi_src4_ti4_id", "shoebox_multi_src4_ti4_diff", "shoebox_multi_src4_ti6_id", "shoebox_multi_src4_ti6_diff", "shoebox_multi_src4_ti3_id", "shoebox_multi_src4_ti3_diff", "shoebox_multi_src4_ti2_id", "shoebox_multi_src4_ti2_diff", "shoebox_multi_src6_ti1_id", "shoebox_multi_src6_ti1_diff", "shoebox_multi_src6_ti2_id", "shoebox_multi_src6_ti2_diff", "shoebox_multi_src6_ti3_id", "shoebox_multi_src6_ti3_diff", "shoebox_multi_src6_ti4_id", "shoebox_multi_src6_ti4_diff", "shoebox_multi_src12_ti1_id", "shoebox_multi_src12_ti1_diff", "shoebox_multi_src12_ti2_id", "shoebox_multi_src12_ti2_diff", "shoebox_offset_far", "shoebox_L1_0", "shoebox_L1_1dBFS", "shoebox_L1_2dBFS", "shoebox_L1_neg1dBFS", "shoebox_L1_neg3dBFS", "shoebox_L1_neg5dBFS", "shoebox_L1_neg7dBFS"]
 
base_datasets = ["classroomBase", "dampenedBase", "hallwayBase", "complexBase", "room112_simple", "room112_precise", "vilnius1", "vilnius2", "mprir_s5", "mprir_s6", "mprir_s6_precise", "shoebox_offset_00", "shoebox_offset_01", "shoebox_offset_02", "shoebox_offset_03", "shoebox_offset_04", "shoebox_offset_05", "shoebox_offset_06", "shoebox_offset_07", "shoebox_offset_08", "shoebox_offset_09", "shoebox_offset_10", "shoebox_offset_11", "shoebox_offset_12", "shoebox_offset_13", "shoebox_offset_14", "shoebox_offset_15", "shoebox_offset_16", "shoebox_offset_17", "shoebox_offset_18", "shoebox_offset_19", "shoebox_offset_20", "shoebox_offset_21", "shoebox_offset_22", "shoebox_offset_23", "shoebox_offset_24", "shoebox_offset_25", "shoebox_offset_26", "shoebox_offset_27", "shoebox_offset_28", "shoebox_offset_29", "shoebox_offset_30", "shoebox_offset_31", "shoebox_offset_32", "shoebox_offset_33" "shoebox_offset_34", "shoebox_offset_29_h280", "shoebox_offset_29_h320", "shoebox_L01", "shoebox_L11", "shoebox_L21", "shoebox_L31", "shoebox_L41", "shoebox_L51", "shoebox_L06", "shoebox_L16", "shoebox_L26", "shoebox_L36", "shoebox_L46", "shoebox_L56", "shoebox_multi_src4", "shoebox_multi_src6", "vilnius_multi_1", "vilnius_multi_2", "vilnius_multi_3", "vilnius_multi_4", "vilnius_multi_5", "vilnius_multi_6", "vilnius_multi_7", "vilnius_multi_8", "shoebox_L01_ti1", "shoebox_L01_ti2", "shoebox_L01_ti4", "shoebox_L01_ti8", "shoebox_L01_ti24", "shoebox_multi_src1_ti1", "shoebox_multi_src2_ti1_id", "shoebox_multi_src1_ti2", "shoebox_multi_src1_ti6_id", "shoebox_multi_src1_ti12_id", "shoebox_multi_src1_ti24_id", "shoebox_multi_src2_ti1_id", "shoebox_multi_src2_ti1_diff", "shoebox_multi_src2_ti2_id", "shoebox_multi_src2_ti2_diff", "shoebox_multi_src2_ti3_id", "shoebox_multi_src2_ti3_diff", "shoebox_multi_src2_ti4_id", "shoebox_multi_src2_ti4_diff", "shoebox_multi_src2_ti6_id", "shoebox_multi_src2_ti6_diff", "shoebox_multi_src2_ti12_id", "shoebox_multi_src2_ti12_diff", "shoebox_multi_src3_ti1_id", "shoebox_multi_src3_ti1_diff", "shoebox_multi_src3_ti4_id", "shoebox_multi_src3_ti4_diff", "shoebox_multi_src3_ti6_id", "shoebox_multi_src3_ti6_diff", "shoebox_multi_src3_ti3_id", "shoebox_multi_src3_ti3_diff", "shoebox_multi_src3_ti2_id", "shoebox_multi_src3_ti2_diff", "shoebox_multi_src4_ti1_id", "shoebox_multi_src4_ti1_diff", "shoebox_multi_src4_ti4_id", "shoebox_multi_src4_ti4_diff", "shoebox_multi_src4_ti6_id", "shoebox_multi_src4_ti6_diff", "shoebox_multi_src4_ti3_id", "shoebox_multi_src4_ti3_diff", "shoebox_multi_src4_ti2_id", "shoebox_multi_src4_ti2_diff", "shoebox_multi_src6_ti1_id", "shoebox_multi_src6_ti1_diff", "shoebox_multi_src6_ti2_id", "shoebox_multi_src6_ti2_diff", "shoebox_multi_src6_ti3_id", "shoebox_multi_src6_ti3_diff", "shoebox_multi_src6_ti4_id", "shoebox_multi_src6_ti4_diff", "shoebox_multi_src12_ti1_id", "shoebox_multi_src12_ti1_diff", "shoebox_multi_src12_ti2_id", "shoebox_multi_src12_ti2_diff", "shoebox_offset_far", "shoebox_L1_0", "shoebox_L1_1dBFS", "shoebox_L1_2dBFS", "shoebox_L1_neg1dBFS", "shoebox_L1_neg3dBFS", "shoebox_L1_neg5dBFS", "shoebox_L1_neg7dBFS", "shoebox_offset_27_diff20", "shoebox_offset_27_diff400"]


def dataLoader(name):
    #Classroom Dataset
    if name[:9] == "classroom":
        import rooms.classroom as classroom
        if name=="classroomBase":
            D = classroom.BaseDataset
        else:
            raise ValueError('Invalid Dataset Name')

    #Dampened Room Datasets
    elif name[:8] == "dampened":
        import rooms.dampened as dampened
        if name =="dampenedBase":
            D = dampened.BaseDataset
        elif name =="dampenedRotation":
            D = dampened.RotationDataset
        elif name =="dampenedTranslation":
            D =  dampened.TranslationDataset
        elif name == "dampenedPanel":
            D = dampened.PanelDataset
        else:
            raise ValueError('Invalid Dataset Name')
   
    #Hallway Datasets
    elif name[:7] == "hallway":
        import rooms.hallway as hallway
        if name == "hallwayBase":
            D = hallway.BaseDataset
        elif name == "hallwayRotation":
            D = hallway.RotationDataset
        elif name == "hallwayTranslation":
            D =  hallway.TranslationDataset
        elif name == "hallwayPanel1":
            D =  hallway.PanelDataset1
        elif name == "hallwayPanel2":
            D =  hallway.PanelDataset2
        elif name == "hallwayPanel3":
            D = hallway.PanelDataset3
        else:
            raise ValueError('Invalid Dataset Name')
    
    #Complex Datasets
    elif name[:7] == "complex":
        import rooms.complex as complex
        if name == "complexBase":
            D = complex.BaseDataset
        elif name == "complexRotation":
            D = complex.RotationDataset
        elif name == "complexTranslation":
            D = complex.TranslationDataset
        else:
            raise ValueError('Invalid Dataset Name')

    #Room112 Datasets
    elif name == "room112_simple":
        import rooms.room112_simple as room112_simple
        D = room112_simple.BaseDataset
        
    elif name == "room112_precise":
        import rooms.room112_precise as room112_precise
        D = room112_precise.BaseDataset
        
    elif name == "vilnius1":
        import rooms.vilnius1 as vilnius1
        D = vilnius1.BaseDataset
        
    elif name == "vilnius2":
        import rooms.vilnius2 as vilnius2
        D = vilnius2.BaseDataset
        
    elif name == "mprir_s5":
        import rooms.mprir_s5 as mprir_s5
        D = mprir_s5.BaseDataset
        
    elif name == "mprir_s5_precise":
        import rooms.mprir_s5_precise as mprir_s5_precise
        D = mprir_s5_precise.BaseDataset
 
    elif name == "mprir_s6":
        import rooms.mprir_s6 as mprir_s6
        D = mprir_s6.BaseDataset

    elif name == "mprir_s6_precise":
        import rooms.mprir_s6_precise as mprir_s6_precise
        D = mprir_s6_precise.BaseDataset

    elif name == "shoebox_offset_00":
        import rooms.shoebox_offset_00 as shoebox_offset_00
        D = shoebox_offset_00.BaseDataset

    elif name == "shoebox_offset_01":
        import rooms.shoebox_offset_01 as shoebox_offset_01
        D = shoebox_offset_01.BaseDataset
        
    elif name == "shoebox_offset_02":
        import rooms.shoebox_offset_02 as shoebox_offset_02
        D = shoebox_offset_02.BaseDataset
        
    elif name == "shoebox_offset_03":
        import rooms.shoebox_offset_03 as shoebox_offset_03
        D = shoebox_offset_03.BaseDataset

    elif name == "shoebox_offset_04":
        import rooms.shoebox_offset_04 as shoebox_offset_04
        D = shoebox_offset_04.BaseDataset

    elif name == "shoebox_offset_05":
        import rooms.shoebox_offset_05 as shoebox_offset_05
        D = shoebox_offset_05.BaseDataset

    elif name == "shoebox_offset_06":
        import rooms.shoebox_offset_06 as shoebox_offset_06
        D = shoebox_offset_06.BaseDataset

    elif name == "shoebox_offset_07":
        import rooms.shoebox_offset_07 as shoebox_offset_07
        D = shoebox_offset_07.BaseDataset

    elif name == "shoebox_offset_08":
        import rooms.shoebox_offset_08 as shoebox_offset_08
        D = shoebox_offset_08.BaseDataset

    elif name == "shoebox_offset_09":
        import rooms.shoebox_offset_09 as shoebox_offset_09
        D = shoebox_offset_09.BaseDataset

    elif name == "shoebox_offset_10":
        import rooms.shoebox_offset_10 as shoebox_offset_10
        D = shoebox_offset_10.BaseDataset

    elif name == "shoebox_offset_11":
        import rooms.shoebox_offset_11 as shoebox_offset_11
        D = shoebox_offset_11.BaseDataset

    elif name == "shoebox_offset_12":
        import rooms.shoebox_offset_12 as shoebox_offset_12
        D = shoebox_offset_12.BaseDataset
        
    elif name == "shoebox_offset_13":
        import rooms.shoebox_offset_13 as shoebox_offset_13
        D = shoebox_offset_13.BaseDataset
        
    elif name == "shoebox_offset_14":
        import rooms.shoebox_offset_14 as shoebox_offset_14
        D = shoebox_offset_14.BaseDataset
  
    elif name == "shoebox_offset_15":
        import rooms.shoebox_offset_15 as shoebox_offset_15
        D = shoebox_offset_15.BaseDataset

    elif name == "shoebox_offset_16":
        import rooms.shoebox_offset_16 as shoebox_offset_16
        D = shoebox_offset_16.BaseDataset

    elif name == "shoebox_offset_17":
        import rooms.shoebox_offset_17 as shoebox_offset_17
        D = shoebox_offset_17.BaseDataset

    elif name == "shoebox_offset_18":
        import rooms.shoebox_offset_18 as shoebox_offset_18
        D = shoebox_offset_18.BaseDataset

    elif name == "shoebox_offset_19":
        import rooms.shoebox_offset_19 as shoebox_offset_19
        D = shoebox_offset_19.BaseDataset

    elif name == "shoebox_offset_20":
        import rooms.shoebox_offset_20 as shoebox_offset_20
        D = shoebox_offset_20.BaseDataset

    elif name == "shoebox_offset_21":
        import rooms.shoebox_offset_21 as shoebox_offset_21
        D = shoebox_offset_21.BaseDataset

    elif name == "shoebox_offset_22":
        import rooms.shoebox_offset_22 as shoebox_offset_22
        D = shoebox_offset_22.BaseDataset

    elif name == "shoebox_offset_23":
        import rooms.shoebox_offset_23 as shoebox_offset_23
        D = shoebox_offset_23.BaseDataset

    elif name == "shoebox_offset_24":
        import rooms.shoebox_offset_24 as shoebox_offset_24
        D = shoebox_offset_24.BaseDataset
        
    elif name == "shoebox_offset_25":
        import rooms.shoebox_offset_25 as shoebox_offset_25
        D = shoebox_offset_25.BaseDataset

    elif name == "shoebox_offset_26":
        import rooms.shoebox_offset_26 as shoebox_offset_26
        D = shoebox_offset_26.BaseDataset

    elif name == "shoebox_offset_27":
        import rooms.shoebox_offset_27 as shoebox_offset_27
        D = shoebox_offset_27.BaseDataset

    elif name == "shoebox_offset_28":
        import rooms.shoebox_offset_28 as shoebox_offset_28
        D = shoebox_offset_28.BaseDataset

    elif name == "shoebox_offset_29":
        import rooms.shoebox_offset_29 as shoebox_offset_29
        D = shoebox_offset_29.BaseDataset

    elif name == "shoebox_offset_30":
        import rooms.shoebox_offset_30 as shoebox_offset_30
        D = shoebox_offset_30.BaseDataset

    elif name == "shoebox_offset_31":
        import rooms.shoebox_offset_31 as shoebox_offset_31
        D = shoebox_offset_31.BaseDataset

    elif name == "shoebox_offset_32":
        import rooms.shoebox_offset_32 as shoebox_offset_32
        D = shoebox_offset_32.BaseDataset

    elif name == "shoebox_offset_33":
        import rooms.shoebox_offset_33 as shoebox_offset_33
        D = shoebox_offset_33.BaseDataset

    elif name == "shoebox_offset_34":
        import rooms.shoebox_offset_34 as shoebox_offset_34
        D = shoebox_offset_34.BaseDataset

    elif name == "shoebox_offset_29_h280":
        import rooms.shoebox_offset_29_h280 as shoebox_offset_29_h280
        D = shoebox_offset_29_h280.BaseDataset
        
    elif name == "shoebox_offset_29_h320":
        import rooms.shoebox_offset_29_h320 as shoebox_offset_29_h320
        D = shoebox_offset_29_h320.BaseDataset

    elif name == "shoebox_L01":
        import rooms.shoebox_L01 as shoebox_L01
        D = shoebox_L01.BaseDataset

    elif name == "shoebox_L06":
        import rooms.shoebox_L06 as shoebox_L06
        D = shoebox_L06.BaseDataset
        
    elif name == "shoebox_L11":
        import rooms.shoebox_L11 as shoebox_L11
        D = shoebox_L11.BaseDataset

    elif name == "shoebox_L16":
        import rooms.shoebox_L16 as shoebox_L16
        D = shoebox_L16.BaseDataset

    elif name == "shoebox_L21":
        import rooms.shoebox_L21 as shoebox_L21
        D = shoebox_L21.BaseDataset

    elif name == "shoebox_L26":
        import rooms.shoebox_L26 as shoebox_L26
        D = shoebox_L26.BaseDataset

    elif name == "shoebox_L31":
        import rooms.shoebox_L31 as shoebox_L31
        D = shoebox_L31.BaseDataset

    elif name == "shoebox_L36":
        import rooms.shoebox_L36 as shoebox_L36
        D = shoebox_L36.BaseDataset

    elif name == "shoebox_L41":
        import rooms.shoebox_L41 as shoebox_L41
        D = shoebox_L41.BaseDataset

    elif name == "shoebox_L46":
        import rooms.shoebox_L46 as shoebox_L46
        D = shoebox_L46.BaseDataset

    elif name == "shoebox_L51":
        import rooms.shoebox_L51 as shoebox_L51
        D = shoebox_L51.BaseDataset

    elif name == "shoebox_L56":
        import rooms.shoebox_L56 as shoebox_L56
        D = shoebox_L56.BaseDataset

    elif name == "shoebox_multi_src4":
        import rooms.shoebox_multi_src4 as shoebox_multi_scr4
        D = shoebox_multi_scr4.BaseDataset

    elif name == "shoebox_multi_src6":
        import rooms.shoebox_multi_src6 as shoebox_multi_scr6
        D = shoebox_multi_scr6.BaseDataset
        
    elif name == "vilnius_multi_1":
        import rooms.vilnius_multi_1 as vilnius_multi_1
        D = vilnius_multi_1.BaseDataset
        
    elif name == "vilnius_multi_2":
        import rooms.vilnius_multi_2 as vilnius_multi_2
        D = vilnius_multi_2.BaseDataset

    elif name == "vilnius_multi_3":
        import rooms.vilnius_multi_3 as vilnius_multi_3
        D = vilnius_multi_3.BaseDataset

    elif name == "vilnius_multi_4":
        import rooms.vilnius_multi_4 as vilnius_multi_4
        D = vilnius_multi_4.BaseDataset

    elif name == "vilnius_multi_5":
        import rooms.vilnius_multi_5 as vilnius_multi_5
        D = vilnius_multi_5.BaseDataset

    elif name == "vilnius_multi_6":
        import rooms.vilnius_multi_6 as vilnius_multi_6
        D = vilnius_multi_6.BaseDataset

    elif name == "vilnius_multi_7":
        import rooms.vilnius_multi_7 as vilnius_multi_7
        D = vilnius_multi_7.BaseDataset

    elif name == "vilnius_multi_8":
        import rooms.vilnius_multi_8 as vilnius_multi_8
        D = vilnius_multi_8.BaseDataset

    elif name == "shoebox_L01_ti1":
        import rooms.shoebox_L01_ti1 as shoebox_L01_ti1
        D = shoebox_L01_ti1.BaseDataset
        
    elif name == "shoebox_L01_ti2":
        import rooms.shoebox_L01_ti2 as shoebox_L01_ti2
        D = shoebox_L01_ti2.BaseDataset
        
    elif name == "shoebox_L01_ti4":
        import rooms.shoebox_L01_ti4 as shoebox_L01_ti4
        D = shoebox_L01_ti4.BaseDataset

    elif name == "shoebox_L01_ti8":
        import rooms.shoebox_L01_ti8 as shoebox_L01_ti8
        D = shoebox_L01_ti8.BaseDataset

    elif name == "shoebox_L01_ti24":
        import rooms.shoebox_L01_ti24 as shoebox_L01_ti24
        D = shoebox_L01_ti24.BaseDataset

    elif name == "shoebox_multi_src2_ti1_id":
        import rooms.shoebox_multi_src2_ti1_id as shoebox_multi_src2_ti1_id
        D = shoebox_multi_src2_ti1_id.BaseDataset

    elif name == "shoebox_multi_src2_ti1_diff":
        import rooms.shoebox_multi_src2_ti1_diff as shoebox_multi_src2_ti1_diff
        D = shoebox_multi_src2_ti1_diff.BaseDataset

    elif name == "shoebox_multi_src2_ti2_id":
        import rooms.shoebox_multi_src2_ti2_id as shoebox_multi_src2_ti2_id
        D = shoebox_multi_src2_ti2_id.BaseDataset

    elif name == "shoebox_multi_src2_ti2_diff":
        import rooms.shoebox_multi_src2_ti2_diff as shoebox_multi_src2_ti2_diff
        D = shoebox_multi_src2_ti2_diff.BaseDataset

    elif name == "shoebox_multi_src2_ti3_id":
        import rooms.shoebox_multi_src2_ti3_id as shoebox_multi_src2_ti3_id
        D = shoebox_multi_src2_ti3_id.BaseDataset

    elif name == "shoebox_multi_src2_ti3_diff":
        import rooms.shoebox_multi_src2_ti3_diff as shoebox_multi_src2_ti3_diff
        D = shoebox_multi_src2_ti3_diff.BaseDataset

    elif name == "shoebox_multi_src2_ti4_id":
        import rooms.shoebox_multi_src2_ti4_id as shoebox_multi_src2_ti4_id
        D = shoebox_multi_src2_ti4_id.BaseDataset

    elif name == "shoebox_multi_src2_ti4_diff":
        import rooms.shoebox_multi_src2_ti4_diff as shoebox_multi_src2_ti4_diff
        D = shoebox_multi_src2_ti4_diff.BaseDataset

    elif name == "shoebox_multi_src2_ti6_id":
        import rooms.shoebox_multi_src2_ti6_id as shoebox_multi_src2_ti6_id
        D = shoebox_multi_src2_ti6_id.BaseDataset

    elif name == "shoebox_multi_src2_ti6_diff":
        import rooms.shoebox_multi_src2_ti6_diff as shoebox_multi_src2_ti6_diff
        D = shoebox_multi_src2_ti6_diff.BaseDataset

    elif name == "shoebox_multi_src2_ti12_id":
        import rooms.shoebox_multi_src2_ti12_id as shoebox_multi_src2_ti12_id
        D = shoebox_multi_src2_ti12_id.BaseDataset

    elif name == "shoebox_multi_src2_ti12_diff":
        import rooms.shoebox_multi_src2_ti12_diff as shoebox_multi_src2_ti12_diff
        D = shoebox_multi_src2_ti12_diff.BaseDataset

    elif name == "shoebox_multi_src3_ti1_id":
        import rooms.shoebox_multi_src3_ti1_id as shoebox_multi_src3_ti1_id
        D = shoebox_multi_src3_ti1_id.BaseDataset

    elif name == "shoebox_multi_src3_ti1_diff":
        import rooms.shoebox_multi_src3_ti1_diff as shoebox_multi_src3_ti1_diff
        D = shoebox_multi_src3_ti1_diff.BaseDataset

    elif name == "shoebox_multi_src3_ti4_id":
        import rooms.shoebox_multi_src3_ti4_id as shoebox_multi_src3_ti4_id
        D = shoebox_multi_src3_ti4_id.BaseDataset

    elif name == "shoebox_multi_src3_ti4_diff":
        import rooms.shoebox_multi_src3_ti4_diff as shoebox_multi_src3_ti4_diff
        D = shoebox_multi_src3_ti4_diff.BaseDataset
 
    elif name == "shoebox_multi_src3_ti6_id":
        import rooms.shoebox_multi_src3_ti6_id as shoebox_multi_src3_ti6_id
        D = shoebox_multi_src3_ti6_id.BaseDataset

    elif name == "shoebox_multi_src3_ti6_diff":
        import rooms.shoebox_multi_src3_ti6_diff as shoebox_multi_src3_ti6_diff
        D = shoebox_multi_src3_ti6_diff.BaseDataset

    elif name == "shoebox_multi_src3_ti3_id":
        import rooms.shoebox_multi_src3_ti3_id as shoebox_multi_src3_ti3_id
        D = shoebox_multi_src3_ti3_id.BaseDataset

    elif name == "shoebox_multi_src3_ti3_diff":
        import rooms.shoebox_multi_src3_ti3_diff as shoebox_multi_src3_ti3_diff
        D = shoebox_multi_src3_ti3_diff.BaseDataset

    elif name == "shoebox_multi_src3_ti2_id":
        import rooms.shoebox_multi_src3_ti2_id as shoebox_multi_src3_ti2_id
        D = shoebox_multi_src3_ti2_id.BaseDataset

    elif name == "shoebox_multi_src3_ti2_diff":
        import rooms.shoebox_multi_src3_ti2_diff as shoebox_multi_src3_ti2_diff
        D = shoebox_multi_src3_ti2_diff.BaseDataset

    elif name == "shoebox_multi_src4_ti1_id":
        import rooms.shoebox_multi_src4_ti1_id as shoebox_multi_src4_ti1_id
        D = shoebox_multi_src4_ti1_id.BaseDataset

    elif name == "shoebox_multi_src4_ti1_diff":
        import rooms.shoebox_multi_src4_ti1_diff as shoebox_multi_src4_ti1_diff
        D = shoebox_multi_src4_ti1_diff.BaseDataset

    elif name == "shoebox_multi_src4_ti4_id":
        import rooms.shoebox_multi_src4_ti4_id as shoebox_multi_src4_ti4_id
        D = shoebox_multi_src4_ti4_id.BaseDataset

    elif name == "shoebox_multi_src4_ti4_diff":
        import rooms.shoebox_multi_src4_ti4_diff as shoebox_multi_src4_ti4_diff
        D = shoebox_multi_src4_ti4_diff.BaseDataset
 
    elif name == "shoebox_multi_src4_ti6_id":
        import rooms.shoebox_multi_src4_ti6_id as shoebox_multi_src4_ti6_id
        D = shoebox_multi_src4_ti6_id.BaseDataset

    elif name == "shoebox_multi_src4_ti6_diff":
        import rooms.shoebox_multi_src4_ti6_diff as shoebox_multi_src4_ti6_diff
        D = shoebox_multi_src4_ti6_diff.BaseDataset

    elif name == "shoebox_multi_src4_ti3_id":
        import rooms.shoebox_multi_src4_ti3_id as shoebox_multi_src4_ti3_id
        D = shoebox_multi_src4_ti3_id.BaseDataset

    elif name == "shoebox_multi_src4_ti3_diff":
        import rooms.shoebox_multi_src4_ti3_diff as shoebox_multi_src4_ti3_diff
        D = shoebox_multi_src4_ti3_diff.BaseDataset

    elif name == "shoebox_multi_src4_ti2_id":
        import rooms.shoebox_multi_src4_ti2_id as shoebox_multi_src4_ti2_id
        D = shoebox_multi_src4_ti2_id.BaseDataset

    elif name == "shoebox_multi_src4_ti2_diff":
        import rooms.shoebox_multi_src4_ti2_diff as shoebox_multi_src4_ti2_diff
        D = shoebox_multi_src4_ti2_diff.BaseDataset

    elif name == "shoebox_multi_src6_ti1_id":
        import rooms.shoebox_multi_src6_ti1_id as shoebox_multi_src6_ti1_id
        D = shoebox_multi_src6_ti1_id.BaseDataset

    elif name == "shoebox_multi_src6_ti1_diff":
        import rooms.shoebox_multi_src6_ti1_diff as shoebox_multi_src6_ti1_diff
        D = shoebox_multi_src6_ti1_diff.BaseDataset

    elif name == "shoebox_multi_src6_ti2_id":
        import rooms.shoebox_multi_src6_ti2_id as shoebox_multi_src6_ti2_id
        D = shoebox_multi_src6_ti2_id.BaseDataset

    elif name == "shoebox_multi_src6_ti2_diff":
        import rooms.shoebox_multi_src6_ti2_diff as shoebox_multi_src6_ti2_diff
        D = shoebox_multi_src6_ti2_diff.BaseDataset

    elif name == "shoebox_multi_src6_ti3_id":
        import rooms.shoebox_multi_src6_ti3_id as shoebox_multi_src6_ti3_id
        D = shoebox_multi_src6_ti3_id.BaseDataset

    elif name == "shoebox_multi_src6_ti3_diff":
        import rooms.shoebox_multi_src6_ti3_diff as shoebox_multi_src6_ti3_diff
        D = shoebox_multi_src6_ti3_diff.BaseDataset

    elif name == "shoebox_multi_src6_ti4_id":
        import rooms.shoebox_multi_src6_ti4_id as shoebox_multi_src6_ti4_id
        D = shoebox_multi_src6_ti4_id.BaseDataset

    elif name == "shoebox_multi_src6_ti4_diff":
        import rooms.shoebox_multi_src6_ti4_diff as shoebox_multi_src6_ti4_diff
        D = shoebox_multi_src6_ti4_diff.BaseDataset

    elif name == "shoebox_multi_src12_ti1_id":
        import rooms.shoebox_multi_src12_ti1_id as shoebox_multi_src12_ti1_id
        D = shoebox_multi_src12_ti1_id.BaseDataset

    elif name == "shoebox_multi_src12_ti1_diff":
        import rooms.shoebox_multi_src12_ti1_diff as shoebox_multi_src12_ti1_diff
        D = shoebox_multi_src12_ti1_diff.BaseDataset

    elif name == "shoebox_multi_src12_ti2_id":
        import rooms.shoebox_multi_src12_ti2_id as shoebox_multi_src12_ti2_id
        D = shoebox_multi_src12_ti2_id.BaseDataset

    elif name == "shoebox_multi_src12_ti2_diff":
        import rooms.shoebox_multi_src12_ti2_diff as shoebox_multi_src12_ti2_diff
        D = shoebox_multi_src12_ti2_diff.BaseDataset

    elif name == "shoebox_multi_src1_ti1":
        import rooms.shoebox_multi_src1_ti1 as shoebox_multi_src1_ti1
        D = shoebox_multi_src1_ti1.BaseDataset

    elif name == "shoebox_multi_src1_ti2":
        import rooms.shoebox_multi_src1_ti2 as shoebox_multi_src1_ti2
        D = shoebox_multi_src1_ti2.BaseDataset

    elif name == "shoebox_multi_src1_ti6":
        import rooms.shoebox_multi_src1_ti6 as shoebox_multi_src1_ti6
        D = shoebox_multi_src1_ti6.BaseDataset

    elif name == "shoebox_multi_src1_ti12":
        import rooms.shoebox_multi_src1_ti12 as shoebox_multi_src1_ti12
        D = shoebox_multi_src1_ti12.BaseDataset

    elif name == "shoebox_multi_src1_ti24":
        import rooms.shoebox_multi_src1_ti24 as shoebox_multi_src1_ti24
        D = shoebox_multi_src1_ti24.BaseDataset

    elif name == "shoebox_offset_far":
        import rooms.shoebox_offset_far as shoebox_offset_far
        D = shoebox_offset_far.BaseDataset
        
    elif name == "shoebox_offset_27_diff20":
        import rooms.shoebox_offset_27_diff20 as shoebox_offset_27_diff20
        D = shoebox_offset_27_diff20.BaseDataset

    elif name == "shoebox_offset_27_diff400":
        import rooms.shoebox_offset_27_diff400 as shoebox_offset_27_diff400
        D = shoebox_offset_27_diff400.BaseDataset

    elif name == "shoebox_L1_0":
        import rooms.shoebox_L1_0 as shoebox_L1_0
        D = shoebox_L1_0.BaseDataset

    elif name == "shoebox_L1_1dBFS":
        import rooms.shoebox_L1_1dBFS as shoebox_L1_1dBFS
        D = shoebox_L1_1dBFS.BaseDataset

    elif name == "shoebox_L1_2dBFS":
        import rooms.shoebox_L1_2dBFS as shoebox_L1_2dBFS
        D = shoebox_L1_2dBFS.BaseDataset

    elif name == "shoebox_L1_neg1dBFS":
        import rooms.shoebox_L1_neg1dBFS as shoebox_L1_neg1dBFS
        D = shoebox_L1_neg1dBFS.BaseDataset

    elif name == "shoebox_L1_neg3dBFS":
        import rooms.shoebox_L1_neg3dBFS as shoebox_L1_neg3dBFS
        D = shoebox_L1_neg3dBFS.BaseDataset

    elif name == "shoebox_L1_Max_neg5dBFS":
        import rooms.shoebox_L1_Max_neg5dBFS as shoebox_L1_Max_neg5dBFS
        D = shoebox_L1_Max_neg5dBFS.BaseDataset

    elif name == "shoebox_L1_neg7dBFS":
        import rooms.shoebox_L1_neg7dBFS as shoebox_L1_neg7dBFS
        D = shoebox_L1_neg7dBFS.BaseDataset



    else:
        raise ValueError('Invalid Dataset Name')

    D.load_data()
    return D
