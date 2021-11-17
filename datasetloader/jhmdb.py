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

    splits = [str(i) for i in range(1, 4)]

    @classmethod
    def add_argparse_args(cls, parser, default_split=None):
        super().add_argparse_args(parser, default_split)
        child_parser = parser.add_argument_group("JHMDB specific arguments")
        child_parser.add_argument(
            "--full_body_split",
            action="store_true",
            help="Load only the subset with the full body visible")
        return parser

    def __init__(self, data_path, full_body_split=False, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        full_body_split : bool, optional (default is False)
            Load only subset with full body visible if True
        """
        self._data_cols = [
            "video-filename", "data-filename", "viewpoint", "keypoints2D",
            "action", "scales"
        ]
        self._data = {
            "video-filename": [],
            "data-filename": [],
            "action": [],
        }
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in JHMDB.splits
        }

        self._length = 0
        if full_body_split:
            split_filename = "_test_split_"
            split_folder = "sub_splits"
        else:
            split_filename = "_test_split"
            split_folder = "splits"
        for cls_id, cls in tqdm(enumerate(JHMDB.actions)):
            # load dat for this class
            for filename in os.listdir(os.path.join(data_path, "videos", cls)):
                if filename.endswith(".avi"):
                    self._data["video-filename"].append(
                        os.path.join(data_path, "videos", cls, filename))
                    self._data["data-filename"].append(
                        os.path.join(data_path, "joint_positions", cls,
                                     filename[:-4], "joint_positions.mat"))
                    self._data["action"].append(cls_id)
                    self._length += 1
            # load splits  information for this class
            for split in self._splits.keys():
                split_file = os.path.join(
                    data_path, split_folder,
                    cls + split_filename + str(split) + ".txt")
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            seq_name = line[:line.find(".avi") + 4]
                            for i, filename in enumerate(
                                    self._data["video-filename"]):
                                if filename.endswith(os.path.sep + seq_name):
                                    if line[-1] == "1":
                                        self._splits[split]["train"].append(i)
                                    else:
                                        self._splits[split]["test"].append(i)
        super().__init__(**kwargs)

    def load_datafile(self, filename):
        """
        Load the complex data of the dataset.

        Loads all that is currently selected of skeletons, viewpoint and
        scales.

        Parameters
        ----------
        filename : string
            Filename of the file containing the data.
        """
        mat = loadmat(filename)
        data = {}
        if "keypoints2D" in self._selected_cols:
            data["keypoints2D"] = np.transpose(mat["pos_img"])
        if "viewpoint" in self._selected_cols:
            data["viewpoint"] = JHMDB.viewpoints.index(mat["viewpoint"][0])
        if "scales" in self._selected_cols:
            data["scales"] = mat["scale"][0]
        return data

    def __getitem__(self, index):
        """
        Indexing access to the dataset.

        Returns a dictionary of all currently selected data columns of the
        selected item.
        """
        data = super().__getitem__(index)
        # super() provides all non-lazy access, only need to do more for data
        # that hasn't been loaded previously
        if len(self._selected_cols - data.keys()) > 0:
            lazy_data = self.load_datafile(self._data["data-filename"][index])
            for col, val in lazy_data.items():
                data[col] = val
        return data
