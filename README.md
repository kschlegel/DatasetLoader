# DatasetLoader
This is a utility project to provide a convenient and consistent access to various datasets.

In my research I am currently interested in human action recognition and localisation from skeleton data. For datasets without skeleton data I use human pose estimation to obtain a de-identified, low-dimensional representation of the human action data contained in the video sequence. As such I have also become interested in human pose estimation. I have mostly been working with CCTV footage which is not publically available as it contains personal information. Human action recognition from skeletal data has the crucial benefit of de-identification of the data.
This project was started to allow convenient experimentation with various academic datasets with relevance to my research. This includes skeleton and RGB video datasets for action recognition or localisation, but also pose estimation dataset.
More datasets will be added over time, in particular sports related datasets as recently I have become interested in applying some of the methods to sports applications. 

## Contents
1. [Usage](#usage)
2. [Datasets](#datasets)
   1. [LSP](#lsp)
   2. [JHMDB](#jhmdb)
   3. [HARPET](#harpet)
   4. [MPII](#mpii)
   5. [UCF Sports](#ucf_sports)
   6. [PKU-MMD](#pku-mmd)
3. [Requirements](#requirements)

## Usage
A DatasetLoader object will load the information contained in the dataset in its constructor. The object provides an easy but flexible interface to query any or all of this information. As a simple example:
```python
lsp_ds = LSP(PATH_TO_DATASET)

filenames = lsp_ds.get_data("image-filenames", "all")
# filenames is now a list of all filenames of images in the Leeds Sport Pose dataset

it = lsp_ds.get_iterator(("image-filenames","keypoints"), "train")
for filename, keypoints in it:
	# This iterates over filenames and keypoints of elements of the training set of Leeds Sport Pose
	load_image(filename)
	...
```
Both the get_data and get_iterator methods in their first argument take a string or an iterable of strings describing the data components to be retrieved. The idea behind this is that in an action recognition from skeleton data context one will want to use keypoint and action class data, whereas in a pose estimation context one needs the filenames of the images/video and the keypoints. An end-to-end solution consuming image/video data, estimating the pose of people and then classifying their action is going to need all three, filenames, keypoints and action classes. 

The interface is loosly inspired by the dataset modules of [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html). A crucial difference is that torchvision will in video datasets provide video sequences of fixed length. Using a signature based approach I can handle variable length time sequences and thus want the original sequence length.

The get_iterator method can be used to easily create [path-signature feature datasets](https://github.com/kschlegel/psfdataset)
```python
dataset = PSFDataset()
dataset.from_iterator(dataset.get_iterator(("keypoints", "actions"), "train"))
```

## Datasets
This sections lists and briefly describes the supported Datasets and any special properties.

### Leeds Sports Pose
Leeds Sports Pose ([LSP](https://sam.johnson.io/research/lsp.html))
Leeds Sports Pose Extended ([LSP Extended](https://sam.johnson.io/research/lspet.html))
Leeds Sports Pose Extended re-annotated by [Pishchulin et al](https://pose.mpi-inf.mpg.de/contents/pishchulin16cvpr.pdf) ([LSP Extended, re-annotadet](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip))

Provide:
* image-filenames
* keypoints
* train/test split for LSP

```python
ds = LSP(PATH_TO_DATASET)
ds = LSPExtended(PATH_TO_DATASET)
ds = LSPExtended(PATH_TO_DATASET, improved=True)
```

### JHMDB
Joint-annotated Human Motion Data Base ([JHMDB](http://jhmdb.is.tue.mpg.de/))

Provides:
* video-filenames
* keypoints
* actions
* scales
* dataset splits 1,2,3
* dataset subplits 1,2,3 of videos with full body visible

```python
ds = JHMDB(PATH_TO_DATASET)
ds = JHMDB(PATH_TO_DATASET, full_body_split=True)
```

### HARPET
Hockey Action Recognition and Pose Estimation in Temporal Space ([HARPET](https://uwaterloo.ca/vision-image-processing-lab/research-demos/vip-harpet-dataset))

Provides:
* image-filenames (sequences of three images)
* keypoints
* actions
* train/validation/test split

```python
ds = HARPET(PATH_TO_DATASET)
```

### MPII
MPII Human Pose ([MPII](http://human-pose.mpi-inf.mpg.de/))

Provides:
* image-filenames
* keypoints
* scales (of people w.r.t. 200px)
* centres (of people)
* head_bboxes
* subset of sufficiently seperated persons (default)
* train/test split

```python
ds = MPII(PATH_TO_DATASET)
ds = MPII(PATH_TO_DATASET, single_person=False)
```

### UCF Sports
UCF Sports Action Dataset ([UCF Sports](https://www.crcv.ucf.edu/data/UCF_Sports_Action.php))

Provides:
* video-filenames
* image-filenames (sequence of the frames of the video)
* bboxes
* actions
* viewpoints (where specified)
* An extra script to fix some of the issues of the dataset. This script adds image files of frames and videos to elements which only have a video file or only have images of the individual frames respectively

```python
ds = UCFSports(PATH_TO_DATASET)
```

### PKU-MDD
Peking University Multi-Modality Dataset ([PKU-MDD](http://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html))

Provides:
* video-filenames
* skeleton-filenames
* ir-filenames
* depth-filenames
* keypoints
* actions
* cross-subject and cross-view splits
* Single person subset

Loading of skeletons can be deferred as this datasets has almost 7GB worth of skeleton data. Filtering for the single person subset happens on read of the skeleton files. If the loading of skeletons is deferred, the load_keypointfile method provided will return None for any keypoint file containing at least one frame with more than one person. 

```python
ds = PKUMMD(PATH_TO_DATASET)
ds = PKUMMD(PATH_TO_DATASET, single_person=True)
ds = PKUMMD(PATH_TO_DATASET, load_skeletons=False)
for skeleton_file in pku.get_iterator("skeleton-filenames"):
	keypoints = pku.load_keypointfile(skeleton_file)
```

## Requirements
* numpy
* tqdm
* scipy
* h5py
