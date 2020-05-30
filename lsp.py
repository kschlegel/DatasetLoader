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

    def __init__(self, base_folder):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        self._data = {"image-filenames": [], "keypoints": []}
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            "default": {
                "train": [i for i in range(1000)],
                "test": [i for i in range(1000, 2000)]
            }
        }
        self._default_split = "default"
        self._length = 2000

        raw_data = loadmat(os.path.join(base_folder, "joints.mat"))
        self._data["keypoints"] = np.transpose(raw_data['joints'])
        for i in trange(0, 2000):
            self._data["image-filenames"].append(
                os.path.join(
                    base_folder, "images", "im" +
                    ("0" * (4 - len(str(i + 1)))) + str(i + 1) + ".jpg"))

        self._data["image-filenames"] = np.array(self._data["image-filenames"])
