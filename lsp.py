import os.path
import numpy as np
from scipy.io import loadmat
from tqdm import trange

from .datasetloader import DatasetLoader


class LSP(DatasetLoader):
    """
    Leeds Sport Pose
    https://sam.johnson.io/research/lsp.html
    """
    landmarks = [
        "right ankle", "right knee", "right hip", "left hip", "left knee",
        "left ankle", "right wrist", "right elbow", "right shoulder",
        "left shoulder", "left elbow", "left wrist", "neck", "head top"
    ]

    def __init__(self, base_folder, small=False):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        small: bool, optional (default is False)
            If true load the small, readily cropped images and corresponding
            keypoints.
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        self._data = {"filenames": [], "keypoints": []}
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._trainingset = [i for i in range(1000)]
        self._testset = [i for i in range(1000, 2000)]

        filename_tail = ""
        if small:
            filename_tail = "_small"
        raw_data = loadmat(
            os.path.join(base_folder, "joints" + filename_tail + ".mat"))
        self._data["keypoints"] = np.transpose(raw_data['joints'])
        self._filenames = []
        for i in trange(0, 2000):
            self._data["filenames"].append(
                os.path.join(
                    base_folder, "images" + filename_tail, "im" +
                    ("0" * (4 - len(str(i + 1)))) + str(i + 1) + ".jpg"))
        for key in self._data.keys():
            self._data[key] = np.array(self._data[key])
