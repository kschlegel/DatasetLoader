import os.path
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from .datasetloader import DatasetLoader


class JHMDB(DatasetLoader):
    """
    JHMDB Dataset - Joint-annotated Human Motion Data Base
    http://jhmdb.is.tue.mpg.de/
    """

    classes = [
        "brush_hair", "catch", "clap", "climb_stairs", "golf", "jump",
        "kick_ball", "pick", "pour", "pullup", "push", "run", "shoot_ball",
        "shoot_bow", "shoot_gun", "sit", "stand", "swing_baseball", "throw",
        "walk", "wave"
    ]
    landmarks = [
        "neck", "belly", "nose", "right shoulder", "left shoulder",
        "right hip", "left hip", "right elbow", "left elbow", "right knee",
        "left knee", "right wrist", "left wrist", "right ankle", "left ankle"
    ]

    def __init__(self, base_folder, split=1, full_body_split=False):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        split : int, optional (default is 1)
            One of {1,2,3} - Load split n
        full_body_split : bool, optional (default is False)
            Load only subset with full body visible if True
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        self._data = {
            "filenames": [],
            "keypoints": [],
            "actions": [],
            "scales": []
        }
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._trainingset = []
        self._testset = []

        if full_body_split:
            split_filename = "_test_split_"
            split_folder = "sub_splits"
        else:
            split_filename = "_test_split"
            split_folder = "splits"
        split_filename += str(split) + ".txt"
        for cls_id, cls in tqdm(enumerate(JHMDB.classes)):
            with open(
                    os.path.join(base_folder, split_folder,
                                 cls + split_filename), 'r') as split_file:
                for line in split_file:
                    line = line.strip()
                    if line[-1] == "1":
                        self._trainingset.append(self._length)
                    else:
                        self._testset.append(self._length)
                    seq_name = line[:line.find(".avi")]
                    self._data["filenames"].append(
                        os.path.join(base_folder, "videos", cls,
                                     seq_name + ".avi"))
                    mat = loadmat(
                        os.path.join(base_folder, "joint_positions", cls,
                                     seq_name, "joint_positions.mat"))
                    self._data["keypoints"].append(np.transpose(
                        mat["pos_img"]))
                    self._data["actions"].append(cls_id)
                    self._data["scales"].append(mat["scale"][0])
                    self._length += 1
        for key in self._data.keys():
            self._data[key] = np.array(self._data[key])
