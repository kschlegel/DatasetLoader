# DatasetLoader
This is a utility project to provide a convenient and consistent access to various datasets.

## Contents
1. [Usage](#usage)
2. [Datasets](#datasets)
   1. [NTU RGBD](#ntu_rgbd)
   2. [Skeletics152](#skeletics152)
   3. [ChaLearn2013](#chalearn2013)
   4. [BerkeleyMHAD](#berkeleymhad)
   5. [LSP](#lsp)
   6. [JHMDB](#jhmdb)
   7. [HARPET](#harpet)
   8. [MPII](#mpii)
   9. [UCF Sports](#ucf_sports)
   10. [PKU-MMD](#pku-mmd)
3. [Requirements](#requirements)

## Usage
By default most datasets will be loaded lazily by the DatasetLoader object. The constructor will load the file structure and any data where there is no clear benefit in delaying loading (e.g. all data contained in a single file). Data that is distributed accross separate files for individual dataset elements is held as filenames and loaded on request. By passing no_lazy_loading=True to the constructor this can be changed to loading all data at once. Some of the small datasets which have all their data in single files the dataset will never load lazily.
The object provides an easy but flexible interface to query any or all of this information. As a simple example:
```python
lsp_ds = LSP(PATH_TO_DATASET)

lsp.select_col("image-filename")
sample = lsp[5]
#sample["image-filename"] is now the filename of the 6th item of the dataset

lsp.set_cols("image-filename", "keypoints2D")
for filename, keypoints in lsp.iterate(return_tuple=True):
	# This iterates over filenames and keypoints of elements of Leeds Sport Pose
	load_image(filename)
	...
```
In particular, to select the parts of the data you want from the dataset, use the `select_col(col)` and `deselect_col(col)` methods to add or remove single data columns from the DatasetLoader object, or `set_cols(col1, col2, ...)` to directly set the selection to a given set of columns. Following this subscripting of the DatasetLoader object will return a dictionary with the column names as keys and the data of the indexed sample as values.

The `iterate([split_name],[split],[return_tuple])` method provides iterable access to the dataset. Using split_name you can select a particular pre-defined split of the dataset and the split argument picks between train/val/test part. If return_tuple=False (the default) the iterator returns dictionaries as obtained from subscripting. if return_tuple=True the data is returned as a tuple with the elements ordered in the same order the columns were selected.
The iterato method can be used to easily create [path-signature feature datasets](https://github.com/kschlegel/psfdataset)

Using `set_split(split_name)` you can select a split to be used which can then be accessed using the `trainingset`,`validationset` and `testset` properties. These properties support subscripting and implement \_\_len\_\_. The subset selection can also be done at time of initialisation, by passing the name of the split to use as `split` argument to the constructor.

Lastly, all DatasetLoader classes provide a `add_argparse_args` method to add command line arguments for arguments applying to all datasets (such as the path to the data) and potential arguments specific to a given dataset. If you are using more than one dataset it is safe to call all their `add_argparse_args` methods. The resulting command line args can be passed into the datasetloader obejct as an unpacked dictionary.
```
args = vars(parser.parse(args))
dataset = NTURGBD(**args)
```

## Datasets
This sections lists and briefly describes the supported Datasets and any special properties.

### NTU RGBD
NTU RGB+D Action Recognition Dataset [NTU RGB+D](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)

Currently only keypoint data is included.
Expected filestructure:
./
|- NTU_RGBD_samples_with_missing_skeletons.txt
|- NTU_RGBD120_samples_with_missing_skeletons.txt
|- nturgb+d_skeletons
|  |- ...keypoint data files

    
### Skeletics152
[Skeletics152](https://github.com/skelemoa/quovadis/tree/master/skeletics-152)

Expected filestructure:
./
|- training
|  |- action_class
|  |  |- keypoint data files
|  |-...
|
|- validation
|  |- action_class
|  |  |- keypoint data files
|  |-...

### ChaLearn2013
ChaLearn Looking at People - Gesture Challenge [ChaLearn2013](https://gesture.chalearn.org/2013-multi-modal-challenge/data-2013-challenge)

Currently only video and keypoint data is included.
Expected filestructure:
./
|- trainingdata
|  |- SampleXXXXX
|  |  |- SampleXXXXX_color.mp4
|  |  |- SampleXXXXX_data.mp4
|  |-...
|
|- validationdata
|  |- SampleXXXXX
|  |  |- SampleXXXXX_color.mp4
|  |  |- SampleXXXXX_data.mp4
|  |-...
    
### Berkeley MHAD
Berkeley Multimodal Human Action Database (MHAD) [BerkeleyMHAD](https://tele-immersion.citris-uc.org/berkeley_mhad)

Expected filestructure:
./
|- Mocap
|  |- SkeletalData
|  |  |- csv
|  |  |  |- skl_sSS_aAA_rRR_pos.csv
|  |  |  |- ...
Here the csv files have been converted from the original BVH files.


### Leeds Sports Pose
Leeds Sports Pose ([LSP](https://sam.johnson.io/research/lsp.html))
Leeds Sports Pose Extended ([LSP Extended](https://sam.johnson.io/research/lspet.html))
Leeds Sports Pose Extended re-annotated by [Pishchulin et al](https://pose.mpi-inf.mpg.de/contents/pishchulin16cvpr.pdf) ([LSP Extended, re-annotated](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip))

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

### PKU-MMD
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

```

## Requirements
* numpy
* tqdm
* scipy
* h5py
