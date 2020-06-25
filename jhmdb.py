import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from .datasetloader import DatasetLoader


class JHMDB(DatasetLoader):
    """
    JHMDB Dataset - Joint-annotated Human Motion Data Base
    http://jhmdb.is.tue.mpg.de/
    """

    actions = [
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

    # Imagine the person standing on a compass, facing south. The viewpoint
    # string describes where the camera is located.
    viewpoints = [
        "E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW", "S", "SE", "SSE",
        "SSW", "SW", "W", "WNW", "WSW"
    ]

    def __init__(self, base_folder, full_body_split=False):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        full_body_split : bool, optional (default is False)
            Load only subset with full body visible if True
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        self._data = {
            "video-filenames": [],
            "viewpoints": [],
            "keypoints": [],
            "actions": [],
            "scales": []
        }
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {i: {"train": [], "test": []} for i in range(1, 4)}
        self._default_split = 1

        if full_body_split:
            split_filename = "_test_split_"
            split_folder = "sub_splits"
        else:
            split_filename = "_test_split"
            split_folder = "splits"
        for cls_id, cls in tqdm(enumerate(JHMDB.actions)):
            # load dat for this class
            for filename in os.listdir(os.path.join(base_folder, "videos",
                                                    cls)):
                if filename.endswith(".avi"):
                    self._data["video-filenames"].append(
                        os.path.join(base_folder, "videos", cls, filename))
                    mat = loadmat(
                        os.path.join(base_folder, "joint_positions", cls,
                                     filename[:-4], "joint_positions.mat"))
                    self._data["viewpoints"].append(
                        JHMDB.viewpoints.index(mat["viewpoint"][0]))
                    self._data["keypoints"].append(np.transpose(
                        mat["pos_img"]))
                    self._data["actions"].append(cls_id)
                    self._data["scales"].append(mat["scale"][0])
                    self._length += 1
            # load splits  information for this class
            for split in self._splits.keys():
                split_file = os.path.join(
                    base_folder, split_folder,
                    cls + split_filename + str(split) + ".txt")
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            seq_name = line[:line.find(".avi") + 4]
                            for i, filename in enumerate(
                                    self._data["video-filenames"]):
                                if filename.endswith(os.path.sep + seq_name):
                                    if line[-1] == "1":
                                        self._splits[split]["train"].append(i)
                                    else:
                                        self._splits[split]["test"].append(i)

        for key in self._data.keys():
            self._data[key] = np.array(self._data[key])
