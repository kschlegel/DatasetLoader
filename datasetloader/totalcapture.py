import os
import numpy as np

from .datasetloader import DatasetLoader


class TotalCapture(DatasetLoader):
    """
    TotalCapture Dataset
    https://cvssp.org/data/totalcapture/data/
    """
    landmarks = [
        'pelvis', 'belly', 'mid torso', 'thorax', 'shoulder centre', 'neck',
        'head top', 'right clavicle', 'right shoulder', 'right elbow',
        'right wrist', 'left clavicle', 'left shoulder', 'left elbow',
        'left wrist', 'right hip', 'right knee', 'right ankle', 'left hip',
        'left knee', 'left ankle'
    ]

    actions = ["rom", "walking", "freestyle", "acting"]

    splits = ["default"]

    def __init__(self, data_path, **kwargs):
        """
        Parameters
        ----------
        data_path: string
            folder with dataset on disk
        lazy_loading : bool, optional (default is True)
            Only load individual data items when queried
        """

        # TODO: the dataset got more information, should include more here
        self._data_cols = [
            "video-filenames",
            "keypoint-filename",
            "keypoints3D",
            "action",
            # The dataset also contains these, to be implemented if/when needed
            # "imu-filename",
            # "rotations-filename",
            # "rotations",
            # "mask-filename"
        ]
        self._data = {
            "video-filenames": [],
            "keypoint-filename": [],
            "action": [],
            # The dataset also contains these, to be implemented if/when needed
            # "imu-filename",
            # "rotations-filename": [],
            # "rotations": [],
            # "mask-filename": [],
        }
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in TotalCapture.splits
        }

        self._length = 0
        for subject_id in range(1, 6):
            for action in TotalCapture.actions:
                for sequence_id in range(1, 4):
                    mocap_path = os.path.join(data_path "S" + str(subject_id),
                                              "mocap_csv",
                                              action + str(sequence_id))
                    if os.path.exists(mocap_path):
                        video_dir = os.path.join(data_path
                                                 "S" + str(subject_id),
                                                 "video",
                                                 action + str(sequence_id))
                        video_base_name = "TC_S" + str(
                            subject_id) + "_" + action + str(
                                sequence_id) + "_cam"
                        self._data["video-filenames"].append([
                            os.path.join(video_dir,
                                         video_base_name + str(i) + ".mp4")
                            for i in range(1, 9)
                        ])
                        self._data["keypoint-filename"].append(
                            os.path.join(mocap_path, 'gt_skel_gbl_pos.txt'))
                        self._data["action"].append(
                            TotalCapture.actions.index(action))
                        if ((action == "walking" and sequence_id == 2)
                                or (action == "freestyle" and sequence_id == 3)
                                or (action == "acting" and sequence_id == 3)):
                            self._splits["default"]["test"].append(
                                self._length)
                        else:
                            self._splits["default"]["train"].append(
                                self._length)
                        self._length += 1
        super().__init__(**kwargs)

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.

        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        frames = []
        with open(filename, newline='\n') as f:
            f.readline()  # first line are just the column names
            for line in f:
                frame = np.array(line.split()).astype(np.float)
                frame = frame.reshape(-1, 3)
                frames.append(frame)
        return np.array(frames)

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
            if "keypoints3D" in self._selected_cols:
                data["keypoints3D"] = self.load_keypointfile(
                    self._data["keypoint-filename"][index])
        return data
