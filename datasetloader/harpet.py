import os.path
import numpy as np
import h5py
from tqdm import trange

from .datasetloader import DatasetLoader


class HARPET(DatasetLoader):
    """
    HARPET Dataset - Hockey Action Recognition and Pose Estimation in Temporal
    Space
    https://uwaterloo.ca/vision-image-processing-lab/research-demos/vip-harpet-dataset
    """

    actions = ["Backward", "Forward", "Passing", "Shooting"]
    landmarks = [
        "right ankle", "right knee", "right hip", "left hip", "left knee",
        "left ankle", "pelvis", "thorax", "neck", "head top", "right wrist",
        "right elbow", "right shoulder", "left shoulder", "left elbow",
        "left wrist", "stick top", "stick end"
    ]
    splits = ["default"]

    def __init__(self, data_path, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        """
        # Elements of filenames here are 3-tuples of the 3 frames forming one
        # sequence
        self._data_cols = ["image-filenames", "keypoints", "actions"]
        self._data = {"image-filenames": [], "keypoints": [], "actions": []}
        self._splits = {
            split: {
                "train": [],
                "valid": [],
                "test": []
            }
            for split in HARPET.splits
        }
        self._default_split = "default"

        kwargs["no_lazy_loading"] = True
        super().__init__(**kwargs)

        # load training set
        self._parse_h5_file(data_path, "train")
        set_len = len(self._data["actions"])
        self._splits[self._default_split]["train"] = [
            i for i in range(set_len)
        ]
        # load validation set
        self._parse_h5_file(data_path, "valid")
        self._splits[self._default_split]["valid"] = [
            i for i in range(set_len, len(self._data["actions"]))
        ]
        set_len = len(self._data["actions"])
        # load test set
        self._parse_h5_file(data_path, "test")
        self._splits[self._default_split]["test"] = [
            i for i in range(set_len, len(self._data["actions"]))
        ]
        self._length = len(self._data["actions"])

        for key in self._data.keys():
            self._data[key] = np.array(self._data[key])

    def _parse_h5_file(self, data_path, split):
        """
        Parses one of the .h5 files for the training, validation or testset.
        This seems to be a very inefficient way to get the data but the only
        way that team to work to parse these files?
        """
        h5_file = h5py.File(os.path.join(data_path, "annot_" + split + ".h5"),
                            "r")
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
                filenames[i] = os.path.join(data_path, "images_" + split,
                                            filenames[i])
            self._data["image-filenames"].append(tuple(filenames))
            self._data["keypoints"].append(h5_file["part"][seq:seq + 3])
            self._data["actions"].append(HARPET.actions.index(action))
