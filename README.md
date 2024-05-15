# Hearing Anything Anywhere - CVPR 2024

### [Project Page](https://masonlwang.com/hearinganythinganywhere) | [Video] | [Paper] | [Data]

Code for the DIFFRIR model presenting in Hearing Anything Anywhere. Please contact Mason Wang at masonlwang32 at gmail dot com for any issues.

[Mason Wang<sup>1</sup>](https://masonlwang.com/) | [Ryosuke Sawata<sup>1,2</sup>](https://www.linkedin.com/in/rsawata/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=jp) | [Samuel Clarke<sup>1</sup>](https://samuelpclarke.com/) | [Ruohan Gao<sup>1,3</sup>](https://ruohangao.github.io/) | [Elliott Wu<sup>1</sup>](https://elliottwu.com) |  [Jiajun Wu<sup>1</sup>](https://jiajunwu.com)

<sup>1</sup>Stanford, <sup>2</sup>SONY AI, <sup>3</sup>University of Maryland, College Park


## Downloading our Dataset


## Linking the Dataset

config.py contains a list of paths to the data for different subdatasets. Before using DIFFRIR, you will need to edit config.py so that these paths point to the correct datasets on your machine 


## Training Model
```
python train.py /viscam/projects/audio_nerf/test_code_release/models/classroomBase classroomBase precomputed/classroomBase
```
