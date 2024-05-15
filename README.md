# Hearing Anything Anywhere - CVPR 2024

### [Project Page](https://masonlwang.com/hearinganythinganywhere) | [Video] | [Paper] | [Data]

Code for the DIFFRIR model presenting in Hearing Anything Anywhere. Please contact Mason Wang at masonlwang32 at gmail dot com for any issues.

[Mason Wang<sup>1</sup>](https://masonlwang.com/) | [Ryosuke Sawata<sup>1,2</sup>](https://www.linkedin.com/in/rsawata/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=jp) | [Samuel Clarke<sup>1</sup>](https://samuelpclarke.com/) | [Ruohan Gao<sup>1,3</sup>](https://ruohangao.github.io/) | [Elliott Wu<sup>1</sup>](https://elliottwu.com) |  [Jiajun Wu<sup>1</sup>](https://jiajunwu.com)

<sup>1</sup>Stanford, <sup>2</sup>SONY AI, <sup>3</sup>University of Maryland, College Park



## Organization
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


## Linking the Dataset

```config.py``` contains a list of paths to the data directories for different subdatasets. Each data directory should contain ```RIRs.npy```, ```xyzs.npy```, and so on.

Before using DIFFRIR, you will need to edit config.py so that these paths point to the correct datasets on your machine.


## Training and Evaluation
The three necessary arguments to the training script ```train.py``` are 
1. The path where the model's weights and renderings should be saved
2. The name of the dataset (e.g. ```"classroomBase"```) as specified in ```rooms/dataset.py```
3. The path to the directory of pretraced reflection paths (these are included in the github).

For example, to train and evaluate DIFFRIR on the Classroom Base dataset, simply run:
```
python train.py models/classroomBase classroomBase precomputed/classroomBase
```
In ```models/classroomBase```, the weights and training losses of the model will be saved. In ```models/classroomBase/predictions```, the predicted RIRs for the locations in the dataset, the predicted music renderings, and the predicted binaural RIRs and music for the binaural datapoints in the dataset are saved.

In addition, ```models/classroomBase/predictions``` contains ```(N,)``` .npy arrays specifiying the error for each datapoint for monoaural and binaural music rendering.

## Tracing Paths
The precomputed directory contains traced paths for all of the subdatasets used, but in case you would like to retrace (perhaps to a different order), you can use trace.py:
```
python trace.py precomputed/classroomBase classroomBase
```
The above command will trace the classroomBase dataset to its default reflection order(s), and save the results in ```precomputed/classroomBase```.


## Citation
```
@article{hearinganythinganywhere2024,
  title={Hearing Anything Anywhere},
  author={Mason Wang and Ryosuke Sawata and Samuel Clarke and Ruohan Gao and Elliott Wu and Jiajun Wu},
  year={2024},
  booktitle={Arkiv},
}
```
