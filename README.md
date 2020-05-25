# DatasetLoader
This is a utility project to provide a convenient and consistent access to various datasets.

In my research I am currently interested in human action recognition and localisation from skeleton data. I use human pose estimation to obtain a de-identified, low-dimensional representation of the human action data contained in the video sequence. As such I have also become interested in human pose estimation. 
I mostly work with CCTV footage which is not publically available as it contains personal information. Human action recognition from skeletal data has the crucial benefit of de-identification of the data. This project was started to allow convenient experimentation with various academic datasets with relevance to my research.
More datasets will be added over time, in particular sports related datasets as recently I have become interested in applying some of the methods to sports applications. 

## Contents
1. [Usage](#usage)
2. [Datasets]("datasets")
3. [Requirements](#requirements)

## Usage
A DatasetLoader object will load the information contained in the dataset in its constructor. The object provides an easy but flexible interface to query any or all of this information. As a simple example:
```python
lsp_ds = LSP(PATH_TO_DATASET)

filenames = lsp_ds.get_data("filenames", "all")[0]
# filenames is now a list of all filenames of images in the Leeds Sport Pose dataset

it = lsp_ds.get_iterator(("filenames","keypoints"), "train")
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
The datasets currently supported are:
* [LSP](https://sam.johnson.io/research/lsp.html) - Leeds Sports Pose 
* [JHMDB](http://jhmdb.is.tue.mpg.de/) - Joint-annotated Human Motion Data Base
* [HARPET](https://uwaterloo.ca/vision-image-processing-lab/research-demos/vip-harpet-dataset) - Hockey Action Recognition and Pose Estimation in Temporal Space
* [MPII](http://human-pose.mpi-inf.mpg.de/) - MPII Human Pose

## Requirements
* numpy
* tqdm
* scipy
* h5py
