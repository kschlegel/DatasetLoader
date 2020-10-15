import os
import numpy as np
from scipy.io import loadmat

from .datasetloader import DatasetLoader


class ChaLearn2013(DatasetLoader):
    """
    ChaLearn Looking at People - Gesture Challenge
    https://gesture.chalearn.org/2013-multi-modal-challenge/data-2013-challenge
    """
    landmarks = [
        "pelvis", "belly", "neck", "head top", "left shoulder", "left elbow",
        "left wrist", "left hand", "right shoulder", "right elbow",
        "right wrist", "right hand", "left hip", "left knee", "left ankle",
        "left foot", "right hip", "right knee", "right ankle", "right foot"
    ]

    actions = [
        "vattene", "vieniqui", "perfetto", "furbo", "cheduepalle", "chevuoi",
        "daccordo", "seipazzo", "combinato", "freganiente", "ok",
        "cosatifarei", "basta", "prendere", "noncenepiu", "fame", "tantotempo",
        "buonissimo", "messidaccordo", "sonostufo"
    ]

    def __init__(self, base_dir, lazy_loading=True):
        """
        Parameters
        ----------
        base_dir : string
            folder with dataset on disk
        lazy_loading : bool, optional (default is True)
            Only load individual data items when queried
        """
        self._data_cols = [
            "video-filename",
            "data-filename",
            "keypoints2D",
            "keypoints3D",
            "actions",
            # The dataset also contains these, to be implemented if/when needed
            # "rotations",
            # "depth-filenames",
            # "audio-filenames",
            # "mask-filenames",
        ]
        self._data = {
            "video-filename": [],
            "data-filename": [],
            # The dataset also contains these, to be implemented if/when needed
            # "depth-filenames": [],
            # "audio-filenames": [],
            # "mask-filenames": [],
        }
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {"default": {"train": [], "valid": [], "test": []}}

        # with the different datasets having diferent properties its worthwhile
        # having a var to keep track of the length
        self._length = 0
        #self._load_data_subset(base_dir, "train")
        self._load_data_subset(base_dir, "valid")
        super().__init__(lazy_loading)

    def _load_data_subset(self, base_dir, subset):
        """
        subset = [train, valid, test]
        """
        if subset == "train":
            subset_long = "training"
        elif subset == "valid":
            subset_long = "validation"
        else:
            subset_long = subset

        subset_dir = os.path.join(base_dir, subset_long + "data")
        for sample in os.listdir(subset_dir):
            sample_dir = os.path.join(subset_dir, sample)
            sample_id = int(sample[6:])  # sample = "SampleXXXXX"

            # From dataset homepage: "Training sequences 223, 225 and 228 and
            # Validation sequences from 629 to 639" contain too many samples
            # without skeleton information, please do not use them if your
            # proposed method is based on the skeleton information.
            if (sample_id in (223, 225, 228)
                    or sample_id in (i for i in range(629, 640))):
                continue

            self._data["video-filename"].append(
                os.path.join(sample_dir, sample + "_color.mp4"))
            self._data["data-filename"].append(
                os.path.join(sample_dir, sample + "_data.mat"))
            # self._data["depth-filenames"].append(
            #     os.path.join(sample_dir, sample + "_depth.mp4"))
            # self._data["audio-filenames"].append(
            #     os.path.join(sample_dir, sample + "_audio.wav"))
            # self._data["mask-filenames"].append(
            #     os.path.join(sample_dir, sample + "_user.mp4"))

            self._splits["default"][subset].append(self._length)
            self._length += 1

    def load_datafile(self, filename):
        """
        Load the complex data of the dataset.

        Loads all that is currently selected of 2D and 3D skeletons and gesture
        data with timestamps.

        Parameters
        ----------
        filename : string
            Filename of the file containing the data.
        """
        data = {col: [] for col in self._selected_cols}
        sample_data = loadmat(filename)
        sample_data = sample_data["Video"][0, 0]
        for frame in range(sample_data["NumFrames"][0, 0]):
            frame_data = sample_data["Frames"][0, frame]["Skeleton"][0, 0]
            # # the first few frames can be just zeros, skip
            # if isinstance(frame_data["JointType"][0, 0][0], str):
            if "keypoints2D" in self._selected_cols:
                data["keypoints2D"] += [frame_data["PixelPosition"]]
            if "keypoints3D" in self._selected_cols:
                data["keypoints3D"] += [frame_data["WorldPosition"]]
        if "actions" in self._selected_cols:
            for gesture in sample_data["Labels"][0]:
                data["actions"] += [
                    (ChaLearn2013.actions.index(gesture["Name"][0]),
                     gesture["Begin"][0, 0], gesture["End"][0, 0])
                ]
        return {col: np.array(val) for col, val in data.items()}

    def __getitem__(self, index):
        """
        Indexing access to the dataset.

        Returns a dictionary of all currently selected data columns of the
        selected item.
        """
        data = super().__getitem__(index)
        # super() provides all non-lazy access, only need to do more for data
        # that hasn't been loaded previously
        missing_cols = self._selected_cols - data.keys()
        if len(missing_cols) > 0:
            lazy_data = self.load_datafile(self._data["data-filename"][index])
            for col in missing_cols:
                data[col] = lazy_data[col]
        return data
