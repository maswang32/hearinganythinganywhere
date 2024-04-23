import sys
sys.path.insert(1, "/viscam/projects/audio_nerf/masonstuff/rooms")
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Train-valid splits for each dataset size
"""

def compute_complement_indices(train_indices, n_data):
    test_indices = []
    for i in range(n_data):
        if i not in train_indices:
                test_indices.append(i)

    return test_indices


"""
Determining Training/Valid/Testing Indices for everything (Messy)
"""

train_indices = {}
train_indices_max = {}
valid_indices = {}
valid_indices_max = {}

train_indices[630] = np.arange(12)*(57) #Yea!
train_indices_max[630] = np.arange(315)*2
valid_indices[630] = compute_complement_indices(list(train_indices[630]) + list(train_indices_max[630]), 630)[::2]
valid_indices_max[630] = valid_indices[630]

train_indices[576] = [5, 58, 99, 148, 203, 241, 296, 342, 384, 441, 482, 535] #Yea!

# train_indices[576]  = [58, 99, 148, 203, 241, 296, 305, 342, 384, 441, 482, 535] # !@#$


#train_indices[576]  = [5, 58, 99, 148, 203, 241, 200, 342, 384, 441, 482, 535] # dataset 2




train_indices_max[576] = np.arange(288)*2
valid_indices[576] = compute_complement_indices(list(train_indices[576]) + list(train_indices_max[576]), 576)[::2]
valid_indices_max[576] = valid_indices[576]

train_indices[276] = np.array([0, 23, 46, 69, 92+12, 115, 138, 161, 184, 207, 230, 253])
train_indices_max[276] = np.arange(138)*2
valid_indices[276] = compute_complement_indices(list(train_indices[276]) + list(train_indices_max[276]),276)[::2]
valid_indices_max[276] = valid_indices[276]

train_indices[252] = np.array([  0,  23,  46,  69,  92+12, 115, 138, 161, 184, 207, 230, 241])
train_indices_max[252] = np.arange(126)*2
valid_indices[252] = compute_complement_indices(list(train_indices[252]) + list(train_indices_max[252]),252)[::2]
valid_indices_max[252] = valid_indices[252]

#For smaller datasets, the validation data is different.
train_indices[120] = np.append((np.arange(11)*11), 109)
train_indices_max[120] = np.arange(60)*2
valid_indices[120] = compute_complement_indices(train_indices[120], 120)[::2]
valid_indices_max[120] = compute_complement_indices(train_indices_max[120], 120)[::2]

train_indices[72] = [5, 10, 15, 16, 35, 25, 44, 42, 48, 57, 62, 67]
train_indices_max[72] = np.arange(36)*2
valid_indices[72] = compute_complement_indices(train_indices[72], 72)[::2]
valid_indices_max[72] = compute_complement_indices(train_indices_max[72], 72)[::2]

#Complex Dataset
train_indices[408] =[5,  47,  82, 117, 145, 187, 220, 255, 290, 330+12, 360+12, 404]
train_indices_max[408] = np.arange(204)*2
valid_indices[408] = compute_complement_indices(train_indices[408], 408)[::2]
valid_indices_max[408] = compute_complement_indices(train_indices_max[408], 408)[::2]

#Complex Rotation/Translation
train_indices[132] = [11,  13,  29,  43,  54,  60,  82,  87, 100, 116, 122, 141-12]
train_indices_max[132] = np.arange(66)*2
valid_indices[132] = compute_complement_indices(train_indices[132], 132)[::2]
valid_indices_max[132] = compute_complement_indices(train_indices_max[132], 132)[::2]



"""
Dataset Class

"""
class Dataset:
    def __init__(self, load_dir, speaker_xyz, all_surfaces, speed_of_sound, default_binaural_listener_forward, default_binaural_listener_left, parallel_surface_pairs):

        #Stuff from load_dir
        self.load_dir = load_dir
        self.xyzs = np.load(os.path.join(load_dir, "xyzs.npy"))
        self.RIRs = np.load(os.path.join(load_dir, "RIRs.npy"))
        self.music = np.load(os.path.join(load_dir, "music.npy"), mmap_mode='r')
        self.music_dls = np.load(os.path.join(load_dir, "music_dls.npy"), mmap_mode='r')
        self.bin_music_dls = np.load(os.path.join(load_dir, "bin_music_dls.npy"), mmap_mode='r') #!@#$
        self.bin_xyzs = np.load(os.path.join(load_dir, "bin_xyzs.npy"), mmap_mode='r')
        self.bin_RIRs = np.load(os.path.join(load_dir, "bin_RIRs.npy"), mmap_mode='r')
        self.bin_music = np.load(os.path.join(load_dir, "bin_music.npy"), mmap_mode='r')
        self.mic_numbers = np.load(os.path.join(load_dir, "raw/mic_numbers.npy"))

        #More stuff
        self.speaker_xyz = speaker_xyz
        self.all_surfaces = all_surfaces
        self.speed_of_sound = speed_of_sound
        self.default_binaural_listener_forward = default_binaural_listener_forward
        self.default_binaural_listener_left = default_binaural_listener_left
        self.parallel_surface_pairs = parallel_surface_pairs


        #indices
        self.train_indices = train_indices[self.xyzs.shape[0]]
        self.train_indices_max = train_indices_max[self.xyzs.shape[0]]

        self.valid_indices = valid_indices[self.xyzs.shape[0]]
        self.valid_indices_max = valid_indices_max[self.xyzs.shape[0]]

        self.test_indices = compute_complement_indices( list(self.train_indices)+list(self.valid_indices), self.xyzs.shape[0])
        self.test_indices_max = compute_complement_indices( list(self.train_indices_max)+list(self.valid_indices_max), self.xyzs.shape[0])


    def print_mic_numbers(self):
        print("Train Mic Numbers:")
        print((self.mic_numbers[self.train_indices]))
        print(len(set(self.mic_numbers[self.train_indices])))
        print(self.train_indices)
        print((self.xyzs[self.train_indices]))

        print("Valid Mic Numbers")
        print(self.mic_numbers[self.valid_indices])
        print(len(set(self.mic_numbers[self.valid_indices])))

        print("Valid Max Mic Numbers")
        print(self.mic_numbers[self.valid_indices])
        print(len(set(self.mic_numbers[self.valid_indices_max])))

        plt.scatter(self.xyzs[:,0], self.xyzs[:,1])
        plt.scatter(self.xyzs[self.train_indices,0], self.xyzs[self.train_indices,1])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.xyzs[:,0], self.xyzs[:,1], self.xyzs[:,2], depthshade=False, alpha=0.1)
        ax.scatter(self.xyzs[self.train_indices,0], self.xyzs[self.train_indices,1], self.xyzs[self.train_indices,2], color='red', s=36, alpha=0.2, depthshade=False)
        plt.show()

        plt.plot(self.xyzs[self.train_indices,2])

        sys.path.insert(1, "/viscam/projects/audio_nerf/masonstuff/")
        import evaluate
        evaluate.plot_2d(np.arange(self.xyzs.shape[0]), self.xyzs, self.xyzs[self.train_indices], size=25)




all_datasets = ["classroomBase", "dampenedBase", "dampenedRotation", "dampenedTranslation", "dampenedPanel", "hallwayBase", "hallwayRotation", "hallwayTranslation","hallwayPanel1","hallwayPanel2","hallwayPanel3","complexBase","complexRotation","complexTranslation"]
base_datasets = ["classroomBase", "dampenedBase", "hallwayBase", "complexBase"]


def dataLoader(name):
    #Classroom Dataset
    if name[:9] == "classroom":
        import classroom as classroom
        if name=="classroomBase":
            return classroom.BaseDataset
        else:
            raise ValueError

    #Dampened Room Datasets
    elif name[:8] == "dampened":
        import rooms.dampened as dampened
        if name =="dampenedBase":
            return dampened.BaseDataset
        elif name =="dampenedRotation":
            return dampened.RotationDataset
        elif name =="dampenedTranslation":
            return dampened.TranslationDataset
        elif name == "dampenedPanel":
            return dampened.PanelDataset
        else:
            raise ValueError
    #Hallway Datasets
    elif name[:7] == "hallway":
        import rooms.hallway as hallway
        if name == "hallwayBase":
            return hallway.BaseDataset
        elif name == "hallwayRotation":
            return hallway.RotationDataset
        elif name == "hallwayTranslation":
            return hallway.TranslationDataset
        elif name == "hallwayPanel1":
            return hallway.PanelDataset1
        elif name == "hallwayPanel2":
            return hallway.PanelDataset2
        elif name == "hallwayPanel3":
            return hallway.PanelDataset3
        else:
            raise ValueError
    elif name[:7] == "complex":
        import rooms.complex as complex
        if name == "complexBase":
            return complex.BaseDataset
        elif name == "complexRotation":
            return complex.RotationDataset
        elif name == "complexTranslation":
            return complex.TranslationDataset
        else:
            raise ValueError

    else:
        raise ValueError