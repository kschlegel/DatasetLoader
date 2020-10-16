import os
from scipy.io import loadmat

from .datasetloader import DatasetLoader


class MPI3DHP(DatasetLoader):
    """
    MPI3DHP - Monocular 3D Human Pose Estimation In The Wild
    http://gvv.mpi-inf.mpg.de/3dhp-dataset/

    TODO:
        - Select camera views
        - Number of frames existing in all modalities
        - Other data
    """
    landmarks = [
        'thorax', 'mid shoulder', 'mid torso', 'belly', 'pelvis', 'neck',
        'head', 'head top', 'left clavicle', 'left shoulder', 'left elbow',
        'left wrist', 'left hand', 'right clavicle', 'right shoulder',
        'right elbow', 'right wrist', 'right hand', 'left hip', 'left knee',
        'left ankle', 'left foot', 'left toe', 'right hip', 'right knee',
        'right ankle', 'right foot', 'right toe'
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
            "keypoint-filename",
            "keypoints2D",
            "keypoints3D",
            "keypoints3D-normalised",
            # The dataset also contains these, to be implemented if/when needed
            # "actions",
        ]
        self._data = {
            "keypoint-filename": [],
            # The dataset also contains these, to be implemented if/when needed
            # "actions": [],
        }
        self._splits = {"default": {"train": [], "test": []}}

        self._length = 0

        for subject_id in range(1, 9):
            if os.path.exists(os.path.join(base_dir, "S" + str(subject_id))):
                for sequence_id in range(1, 3):
                    sequence_path = os.path.join(base_dir,
                                                 "S" + str(subject_id),
                                                 "Seq" + str(sequence_id))
                    self._data["keypoint-filename"].append(
                        os.path.join(sequence_path, "annot.mat"))

    def load_skeletonfile(self, filename):
        """
        Load the skeleton data of the dataset.

        Loads all that is currently selected of 2D and 3D skeletons and
        normalised 3D skeletons.

        Parameters
        ----------
        filename : string
            Filename of the file containing the data.
        """
        data = {}
        sample_data = loadmat(filename)
        if "keypoints2D" in self._selected_cols:
            data["keypoints2D"] = [
                sample_data["annot2"][i, 0].reshape(
                    sample_data["annot2"][i, 0].shape[0], -1, 3)
                for i in data["cameras"][0]
            ]
        if "keypoints3D" in self._selected_cols:
            data["keypoints3D"] = [
                sample_data["annot3"][i, 0].reshape(
                    sample_data["annot3"][i, 0].shape[0], -1, 3)
                for i in data["cameras"][0]
            ]
        if "keypoints3D-normalised" in self._selected_cols:
            data["keypoints3D-normalised"] = [
                sample_data["univ_annot3"][i, 0].reshape(
                    sample_data["univ_annot3"][i, 0].shape[0], -1, 3)
                for i in data["cameras"][0]
            ]
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
        missing_cols = self._selected_cols - data.keys()
        if len(missing_cols) > 0:
            lazy_data = self.load_skeletonfile(
                self._data["skeleton-filename"][index])
            for col in missing_cols:
                data[col] = lazy_data[col]
        return data
