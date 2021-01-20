import os
import numpy as np
from scipy.io import loadmat

from .datasetloader import DatasetLoader


class MPI3DHP(DatasetLoader):
    """
    MPI3DHP - Monocular 3D Human Pose Estimation In The Wild
    http://gvv.mpi-inf.mpg.de/3dhp-dataset/
    """
    landmarks = [
        'thorax', 'shoulder centre', 'mid torso', 'belly', 'pelvis', 'neck',
        'head', 'head top', 'left clavicle', 'left shoulder', 'left elbow',
        'left wrist', 'left hand', 'right clavicle', 'right shoulder',
        'right elbow', 'right wrist', 'right hand', 'left hip', 'left knee',
        'left ankle', 'left foot', 'left toe', 'right hip', 'right knee',
        'right ankle', 'right foot', 'right toe'
    ]
    splits = ["default"]

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
            "video-filenames",
            "keypoints2D",
            "keypoints3D",
            "keypoints3D-normalised",
            # The dataset also contains these, to be implemented if/when needed
            # "camera-calibration-filename",
            # "fgmask-filenames",
            # "chairmask-filenames",
        ]
        self._data = {
            "keypoint-filename": [],
            "video-filenames": [],
            "num-frames": []
            # The dataset also contains these, to be implemented if/when needed
            # "camera-calibration-filename": [],
            # "fgmask-filenames": [],
            # "chairmask-filenames": [],
        }
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in MPI3DHP.splits
        }

        self._length = 0

        # some sequences have more frames with keypoints than video frames. These
        # are the numbers of frames for which both exist.
        num_frames = [(6416, 12430), (6502, 6081), (12488, 12283),
                      (6171, 6675), (12820, 12312), (6188, 6145), (6239, 6320),
                      (6468, 6054)]
        for subject_id in range(1, 9):
            if os.path.exists(os.path.join(base_dir, "S" + str(subject_id))):
                for sequence_id in range(1, 3):
                    sequence_path = os.path.join(base_dir,
                                                 "S" + str(subject_id),
                                                 "Seq" + str(sequence_id))
                    self._data["keypoint-filename"].append(
                        os.path.join(sequence_path, "annot.mat"))
                    video_filelist = [''] * 14
                    video_path = os.path.join(sequence_path, "imageSequence")
                    for filename in sorted(os.listdir(video_path)):
                        # Skip hidden files such as .DS_Store on mac
                        if filename.startswith("."):
                            continue
                        cam_id = int(filename[filename.find("_") + 1:-4])
                        video_filelist[cam_id] = os.path.join(
                            video_path, filename)
                    self._data["video-filenames"].append(video_filelist)
                    self._data['num-frames'].append(
                        num_frames[subject_id - 1][sequence_id - 1])
                    self._length += 1
        # Select the set of cams for which we have keypoints video
        self.select_cameraset("vnect")
        super().__init__(lazy_loading=lazy_loading)

    def select_cameraset(self, camset_key='vnect'):
        """
        Select which set of camera views to include.

        Uses the same identifiers as in the original 'mpii_get_cmera_set.m'
        file.
        Parameters
        ----------
        camset_key : string (default is vnect)
            Valid choices are:
             - 'regular': all cameras (14 cams)
             - 'relevant': all cameras, except ceiling mounted (11 cams)
             - 'ceiling': all ceiling mounted cameras (3 cams)
             - 'vnect': chest (5), knee (1) height and cams angled down (2)
             - 'mm3d_chest': chest high cameras (5)
        """
        if camset_key == 'regular':
            self._camera_selection = list(range(14))
        elif camset_key == 'relevant':
            self._camera_selection = list(range(11))
        elif camset_key == 'ceiling':
            self._camera_selection = list(range(11, 14))
        elif camset_key == 'vnect':
            self._camera_selection = [0, 1, 2, 4, 5, 6, 7, 8]
        elif camset_key == 'mm3d_chest':
            self._camera_selection = [0, 2, 4, 7, 8]
        else:
            raise Exception("'" + camset_key + "' is not a valid camera set!")

    def load_keypointfile(self, filename, num_frames=None):
        """
        Load the skeleton data of the dataset.

        Loads all that is currently selected of 2D and 3D skeletons and
        normalised 3D skeletons.

        Parameters
        ----------
        filename : string
            Filename of the file containing the data.
        num_frames : int, optional (default is None)
            If set only loads skeletons up to given frame. Some sequences may
            contain keypoint data than video frames, this allows to only load
            the keypoints for which there are video frames.
        """
        data = {}
        sample_data = loadmat(filename)
        if "keypoints2D" in self._selected_cols:
            data["keypoints2D"] = np.array([
                sample_data["annot2"][i, 0][:num_frames].reshape(
                    sample_data["annot2"][i, 0].shape[0], -1, 2)
                for i in self._camera_selection
            ])
        if "keypoints3D" in self._selected_cols:
            data["keypoints3D"] = []
            for i in self._camera_selection:
                keypoints = sample_data["annot3"][i, 0][:num_frames].reshape(
                    sample_data["annot3"][i, 0].shape[0], -1, 3)
                # For some reason keypoints are upside down by default
                keypoints[:, :, 1] *= -1
                data["keypoints3D"].append(keypoints)
            data["keypoints3D"] = np.array(data["keypoints3D"])
        if "keypoints3D-normalised" in self._selected_cols:
            data["keypoints3D-normalised"] = []
            for i in self._camera_selection:
                keypoints = sample_data["univ_annot3"][
                    i, 0][:num_frames].reshape(
                        sample_data["univ_annot3"][i, 0].shape[0], -1, 3)
                # For some reason keypoints are upside down by default
                keypoints[:, :, 1] *= -1
                data["keypoints3D-normalised"].append(keypoints)
            data["keypoints3D-normalised"] = np.array(
                data["keypoints3D-normalised"])
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
            lazy_data = self.load_keypointfile(
                self._data["keypoint-filename"][index],
                self._data["num-frames"][index])
            for col in missing_cols:
                data[col] = lazy_data[col]
        return data
