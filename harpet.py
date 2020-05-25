import os.path
import numpy as np
import h5py
from tqdm import trange

from .datasetloader import DatasetLoader


class HARPET(DatasetLoader):
    """
    HARPET Dataset - Hockey Action Recognition and Pose Estimation in Temporal Space
    https://uwaterloo.ca/vision-image-processing-lab/research-demos/vip-harpet-dataset
    """

    classes = ["Backward", "Forward", "Passing", "Shooting"]
    landmarks = [
        "right ankle", "right knee", "right hip", "left hip", "left knee",
        "left ankle", "pelvis", "thorax", "neck", "head top", "right wrist",
        "right elbow", "right shoulder", "left shoulder", "left elbow",
        "left wrist", "stick top", "stick end"
    ]

    def __init__(self, base_folder):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        # Elements of filenames here are 3-tuples of the 3 frames forming one
        # sequence
        self._data = {"filenames": [], "keypoints": [], "actions": []}
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._trainingset = []
        self._validationset = []
        self._testset = []

        self._parse_h5_file(base_folder, "train")
        set_len = len(self._data["actions"])
        self._trainingset = [i for i in range(set_len)]
        self._parse_h5_file(base_folder, "valid")
        self._validationset = [
            i for i in range(set_len, len(self._data["actions"]))
        ]
        set_len = len(self._data["actions"])
        self._parse_h5_file(base_folder, "test")
        self._testset = [i for i in range(set_len, len(self._data["actions"]))]
        self._length = len(self._data["actions"])

        for key in self._data.keys():
            self._data[key] = np.array(self._data[key])

    def _parse_h5_file(self, base_folder, split):
        """
        Parses one of the .h5 files for the training, validation or testset.
        This seems to be a very inefficient way to get the data but the only
        way that team to work to parse these files?
        """
        h5_file = h5py.File(
            os.path.join(base_folder, "annot_" + split + ".h5"), "r")
        for seq in trange(0,
                          len(h5_file["imgname"]),
                          3,
                          desc="Loading " + split + " file"):
            filenames = [""] * 3
            for i in range(3):
                for j in range(len(h5_file["imgname"][seq + i])):
                    if h5_file['imgname'][seq + i][j] != 0:
                        filenames[i] += chr(int(h5_file["imgname"][seq +
                                                                   i][j]))
                action = filenames[i][14:filenames[i].find("_")]
                filenames[i] = os.path.join(base_folder, "images_" + split,
                                            filenames[i])
            self._data["filenames"].append(tuple(filenames))
            self._data["keypoints"].append(h5_file["part"][seq:seq + 3])
            self._data["actions"].append(HARPET.classes.index(action))
