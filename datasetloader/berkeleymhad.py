import os

import numpy as np

from .datasetloader import DatasetLoader


class BerkeleyMHAD(DatasetLoader):
    """
    BerkeleyMHAD - Berkeley Multimodal Human Action Database
    https://tele-immersion.citris-uc.org/berkeley_mhad
    """
    actions = [
        "Jumping in place", "Jumping jacks",
        "Bending - hands up all the way down", "Punching (boxing",
        "Waving - two hands", "Waving - one hand (right", "Clapping hands",
        "Throwing a ball", "Sit down then stand up", "Sit down", "Stand up"
    ]

    landmarks = [
        "pelvis", "belly", "mid torso", "thorax", "neck", "head",
        "right shoulder", "right elbow", "right wrist", "left shoulder",
        "left elbow", "left wrist", "right hip", "right knee", "right ankle",
        "right foot", "left hip", "left knee", "left ankle", "left foot"
    ]

    # Order and correspondence of landmarks in the csv file (elements from the
    # csv file in () represent points which don't correspond to a joint but
    # were measured in the mocap system to determin bone rotation. These are
    # dropped using the _landmark_mask):
    # Hips=pelvis, spine=belly, spine1=mid torso, spine2=thorax, Neck=neck,
    # Head=head, (RightShoulder), RightArm=right shoulder, (RightArmRoll),
    # RightForeArm=right elbow, (RightForeArmRoll), RightHand=right wrist,
    # (LeftShoulder), LeftArm=left shoulder, (LeftArmRoll), LeftForeArm=left
    # elbow, (LeftForeArmRoll), LeftHand=left wrist, RightUpLeg=left hip,
    # (RightUpLegRoll), RightLeg=left knee, (RightLegRoll), RightFoot=left
    # ankle, RightToeBase=left foot, LeftUpLeg=right hip, (LeftUpLegRoll),
    # LeftLeg=right knee, (LeftLegRoll), LeftFoot=right ankle,
    # LeftToeBase=right foot
    _landmark_mask = [
        0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 18, 20, 22, 23, 24, 26, 28, 29
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
            "keypoints3D",
            "action",
        ]
        self._data = {"keypoint-filename": [], "action": []}

        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in BerkeleyMHAD.splits
        }

        self._length = 0
        for subject in range(1, 13):
            for action in range(1, 12):
                for recording in range(1, 6):
                    if subject == 4 and action == 8 and recording == 5:
                        # This sequence is missing in the dataset
                        continue
                    self._data["keypoint-filename"].append(
                        os.path.join(
                            base_dir, "Mocap", "SkeletalData", "csv",
                            "skl_s{:02d}_a{:02d}_r{:02d}_pos.csv".format(
                                subject, action, recording)))
                    self._data["action"].append(action - 1)
                    if subject < 8:
                        self._splits["default"]["train"].append(self._length)
                    else:
                        self._splits["default"]["test"].append(self._length)
                    self._length += 1

        super().__init__(lazy_loading)

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.

        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        keypoints = []
        with open(filename, "r") as csv_file:
            csv_file.readline()  # header
            for row in csv_file:
                coords = row.split(",")
                coords = list(map(float, coords[1:]))
                coords = np.array(coords).reshape((-1, 3))
                keypoints.append(coords[self._landmark_mask])
        return np.array(keypoints)

    def __getitem__(self, index):
        """
        Indexing access to the dataset.

        Returns a dictionary of all currently selected data columns of the
        selected item.
        """
        data = super().__getitem__(index)
        # super() provides all non-lazy access, only need to do more for data
        # that hasn't been loaded previously
        if "keypoints3D" in self._selected_cols:
            data["keypoints3D"] = self.load_keypointfile(
                self._data["keypoint-filename"][index])
        return data
