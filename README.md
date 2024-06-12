# Hearing Anything Anywhere - CVPR 2024

### [Project Page](https://masonlwang.com/hearinganythinganywhere) | [Video](https://www.youtube.com/watch?v=Cv9oOFVXem4) | [Paper](https://arxiv.org/pdf/2406.07532) | [Data](https://zenodo.org/records/11195833)

Code for the DIFFRIR model presented in Hearing Anything Anywhere. Please contact Mason Wang at masonlwang32 at gmail dot com for any inquiries or issues.

[Mason Wang<sup>1</sup>](https://masonlwang.com/) | [Ryosuke Sawata<sup>1,2</sup>](https://www.linkedin.com/in/rsawata/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=jp) | [Samuel Clarke<sup>1</sup>](https://samuelpclarke.com/) | [Ruohan Gao<sup>1,3</sup>](https://ruohangao.github.io/) | [Elliott Wu<sup>1</sup>](https://elliottwu.com) |  [Jiajun Wu<sup>1</sup>](https://jiajunwu.com)

<sup>1</sup>Stanford, <sup>2</sup>SONY AI, <sup>3</sup>University of Maryland, College Park



## Organization

```HRIRs``` - the SADIE dataset of Head-Related Room Impulse Responses, which are used to render binaural audio.

```example_trajectories``` - 3 notebooks used for generating example trajectories using trajectory.py, which are on the website. Includes a hallway, dampened room, and virtual speaker rotation example. Also contains audio files you can simulate in the room.

```models``` - weights for pretrained models in each of the four base subdatasets.

```precomputed``` - folder of precomputed reflection paths for all datasets, computed up to their default order

```rooms``` - information on the geometry of each room, also contains ```dataset.py```, which is used for loading data.

```binauralize.py``` - tools used for binaural rendering

```config.py``` - used to link the dataset

```evaluate.py``` - tools used to evaluate renderings and render music

```metrics.py``` - loss functions and evaluation metrics

```render.py``` - the DIFFRIR renderer, used to render RIRs.

```train.py``` - Training script, will train a DIFFRIR renderer on the specified dataset, save its outputs, and evaluate it.

```trajectory.py``` - Used for rendering trajectories, e.g., simulating walking through a room while audio is playing

## Downloading our Dataset
The dataset can be downloaded from zenodo: https://zenodo.org/records/11195833


## Linking the Dataset

```config.py``` contains a list of paths to the data directories for different subdatasets. Each data directory should contain ```RIRs.npy```, ```xyzs.npy```, and so on.

Before using DIFFRIR, you will need to edit ```config.py``` so that these paths point to the correct directories on your machine.


## Rendering Trajectories
There are three example notebooks in the example_trajectories directory that show you how to generate realistic, immersive audio in a room.


## Training and Evaluation
The three necessary arguments to the training script ```train.py``` are:
1. The path where the model's weights and renderings should be saved.
2. The name of the dataset (e.g. ```"classroomBase"```) as specified in ```rooms/dataset.py```.
3. The path to the directory of pretraced reflection paths (these are included as part of this github repo), which should be ```precomputed/<dataset_name>```

For example, to train and evaluate DIFFRIR on the Classroom Base dataset, simply run:
```
python train.py models/classroomBase classroomBase precomputed/classroomBase
```

In the above example:
1. The weights and training losses of the model will be saved in ```models/classroomBase```,
2. In ```models/classroomBase/predictions```, the predicted RIRs for the monoaural locations in the dataset, the predicted music renderings, and the predicted binaural RIRs and music for the binaural datapoints in the dataset will be saved.
3. ```models/classroomBase/predictions``` will contain ```(N,)``` numpy arrays specifying the per-datapoint error for monoaural RIR rendering.
4. ```models/classroomBase/predictions``` will contain ```(N,K)``` numpy arrays specifying the per-datapoint, per-song error for monoaural music rendering.


## Tracing Paths
The precomputed directory contains traced paths for all of the subdatasets used, but in case you would like to retrace (perhaps to a different order), you can use trace.py:
```
python trace.py precomputed/classroomBase classroomBase
```
The above command will trace the classroomBase dataset to its default reflection order(s), and save the results in ```precomputed/classroomBase```.


## Citation
```
@InProceedings{hearinganythinganywhere2024,
  title={Hearing Anything Anywhere},
  author={Mason Wang and Ryosuke Sawata and Samuel Clarke and Ruohan Gao and Shangzhe Wu and Jiajun Wu},
  booktitle={CVPR},
  year={2024}}


```
