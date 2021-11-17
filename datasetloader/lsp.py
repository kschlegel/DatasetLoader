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
    splits = ["default"]

    def __init__(self, data_path, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        """
        self._data_cols = ["image-filename", "keypoints2D"]
        self._data = {"image-filename": [], "keypoints2D": []}
        self._splits = {
            "default": {
                "train": [i for i in range(1000)],
                "test": [i for i in range(1000, 2000)]
            }
        }
        self._length = 2000

        kwargs["no_lazy_loading"] = True
        super().__init__(**kwargs)

        raw_data = loadmat(os.path.join(data_path, "joints.mat"))
        self._data["keypoints2D"] = np.transpose(raw_data['joints'])
        for i in trange(0, 2000):
            self._data["image-filename"].append(
                os.path.join(
                    data_path, "images", "im" + ("0" * (4 - len(str(i + 1)))) +
                    str(i + 1) + ".jpg"))

        self._data["image-filename"] = np.array(self._data["image-filename"])
