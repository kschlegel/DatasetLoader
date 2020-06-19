import os.path
import numpy as np
from scipy.io import loadmat
from tqdm import trange

from .datasetloader import DatasetLoader


class LSPExtended(DatasetLoader):
    """
    Leeds Sport Pose
    https://sam.johnson.io/research/lspet.html
    Re-annotated version by Pishchulin et al.
    http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip
    """
    landmarks = [
        "right ankle", "right knee", "right hip", "left hip", "left knee",
        "left ankle", "right wrist", "right elbow", "right shoulder",
        "left shoulder", "left elbow", "left wrist", "neck", "head top"
    ]

    def __init__(self, base_folder, improved=False):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        improved : bool, optional (default is False)
            LSP extended has be re-annotated with higher quality labels and the
            original large images but misses a small fraction of the original
            lsp extended dataset. If True load this new version of lsp
            extended.
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        self._data = {"image-filenames": [], "keypoints": []}
        # lsp extended doesn't have a split, it was only used to increase
        # the trainingset size of lsp
        self._splits = None
        if improved:
            self._length = 9428
        else:
            self._length = 10000

        raw_data = loadmat(os.path.join(base_folder, "joints.mat"))
        self._data["keypoints"] = np.transpose(raw_data['joints'], (2, 0, 1))
        for i in trange(0, 10000):
            filename = os.path.join(
                base_folder, "images",
                "im" + ("0" * (5 - len(str(i + 1)))) + str(i + 1))
            if improved:
                filename += ".png"
                # the improved version misses a few images, in this case skip
                # numbers that don't exist
                if not os.path.exists(filename):
                    continue
            else:
                filename += ".jpg"
            self._data["image-filenames"].append(filename)

        self._data["image-filenames"] = np.array(self._data["image-filenames"])
