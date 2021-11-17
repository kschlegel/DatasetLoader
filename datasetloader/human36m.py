import os

import numpy as np
import cdflib

from .datasetloader import DatasetLoader


class Human36M(DatasetLoader):
    """
    Human 3.6M Dataset
    http://vision.imar.ro/human3.6m/description.php
    """
    landmarks = [
        "not used",  # 0 pelvis - near identical with 11 - NOT USED
        "left hip",  # 1
        "left knee",  # 2
        "left ankle",  # 3
        "left foot",  # 4
        "left small toe",  # 5
        "right hip",  # 6
        "right knee",  # 7
        "right ankle",  # 8
        "right foot",  # 9
        "right small toe",  # 10
        "pelvis",  # 11
        "mid torso",  # 12
        "neck",  # 13
        "nose",  # 14
        "head top",  # 15
        "not used",  # 16 neck - identical with 13 & 24 - NOT USED
        "left shoulder",  # 17
        "left elbow",  # 18
        "left wrist",  # 19
        "not used",  # 20 left wrist - identical with 19 - NOT USED
        "left thumb",  # 21
        "left handtip",  # 22
        "not used",  # 23 left hand tip - identical with 22 - NOT USED
        "not used",  # 24 - neck - identical with 13 & 16 - NOT USED
        "right shoulder",  # 25
        "right elbow",  # 26
        "right wrist",  # 27
        "not used",  # 28 right wrist - identical with 27 - NOT USED
        "right thumb",  # 29
        "right handtip",  # 30
        "not used",  # 31 right hand tip - identical with 30 - NOT USED
    ]

    actions = [
        "directions", "discussion", "eating", "greeting", "phoning", "posing",
        "purchases", "sitting", "sittingdown", "smoking", "photo", "waiting",
        "walking", "walkdog", "walktogether"
    ]

    splits = ["default"]

    def __init__(self, data_path, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        """

        self._data_cols = [
            "video-filenames",
            "keypoint2D-filenames",
            "keypoint3D-filename",
            "keypoint3D-mono-filenames",
            "keypoint3D-mono-universal-filenames",
            "keypoints2D",
            "keypoints3D",
            "keypoints3D-mono",
            "keypoints3D-mono-universal",
            "action",
            # The dataset also contains other data, to be implemented if/when needed
        ]
        self._data = {
            "video-filenames": [],
            "keypoint2D-filenames": [],
            "keypoint3D-filename": [],
            "keypoint3D-mono-filenames": [],
            "keypoint3D-mono-universal-filenames": [],
            "action": [],
            # The dataset also contains other data, to be implemented if/when needed
        }
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in Human36M.splits
        }

        self._length = 0
        for subject_id in range(1, 11):
            if os.path.exists(os.path.join(data_path, "S" + str(subject_id))):
                keypoint_folder = os.path.join(data_path,
                                               "S" + str(subject_id),
                                               "MyPoseFeatures")
                for filename in os.listdir(
                        os.path.join(keypoint_folder, 'D3_Positions')):
                    if filename.startswith("."):
                        continue
                    # chop of file ending, leave dot on
                    # find will give location of " " if exists ("Action X.cfd")
                    # otw ("Action.cdf") find will return -1 and we chop off the dot we left
                    # on previously
                    action = filename[:-3]
                    action = action[:action.find(" ")].lower()
                    if subject_id == 11 and action == 'directions':
                        continue  # Discard corrupted video
                    self._data["keypoint3D-filename"].append(
                        os.path.join(keypoint_folder, "D3_positions",
                                     filename))
                    # some actions are named inconsitently, fix names
                    if action == "takingphoto":
                        action = "photo"
                    elif action == "walkingdog":
                        action = "walkdog"
                    self._data["action"].append(Human36M.actions.index(action))
                    base_filename = filename[:-4]

                    video_filenames = []
                    keypoint2D_filenames = []
                    keypoint3D_mono_filenames = []
                    keypoint3D_mono_universal_filenames = []
                    for d2_filename in os.listdir(
                            os.path.join(keypoint_folder, 'D2_Positions')):
                        # skip hidden files and files of oother actions
                        if (d2_filename.startswith(".")
                                or not d2_filename.startswith(base_filename)):
                            continue
                        # filenames can b of the form "Action.cam_name.cdf" or"Action X.cam_name.cdf"
                        # need to avoid false matches of Action.cam_name for action X.cam_name
                        if d2_filename[len(base_filename):len(base_filename) +
                                       1] != ".":
                            continue
                        # filenames here are of the structure "base_filename.cam_name.cdf"
                        # the same is true for other corresponding files so add all of them
                        # to the respective lists
                        cam_name = d2_filename[len(base_filename) + 1:-4]
                        keypoint2D_filenames.append(
                            os.path.join(keypoint_folder, "D2_Positions",
                                         d2_filename))
                        keypoint3D_mono_filenames.append(
                            os.path.join(
                                keypoint_folder, "D3_Positions_mono",
                                base_filename + "." + cam_name + ".cdf"))
                        keypoint3D_mono_universal_filenames.append(
                            os.path.join(
                                keypoint_folder, "D3_Positions_mono_universal",
                                base_filename + "." + cam_name + ".cdf"))
                        video_filenames.append(
                            os.path.join(
                                data_path, "S" + str(subject_id), "Videos",
                                base_filename + "." + cam_name + ".mp4"))

                    self._data["video-filenames"].append(video_filenames)
                    self._data["keypoint2D-filenames"].append(
                        keypoint2D_filenames)
                    self._data["keypoint3D-mono-filenames"].append(
                        keypoint3D_mono_filenames)
                    self._data["keypoint3D-mono-universal-filenames"].append(
                        keypoint3D_mono_universal_filenames)
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
        # print(filename)
        cdf_file = cdflib.CDF(filename)
        keypoints = cdf_file.varget("Pose", expand=False)[0]
        if keypoints.shape[-1] == 64:  # 2D
            keypoints = keypoints.reshape(-1, 32, 2)
        elif keypoints.shape[-1] == 96:  # 3D
            keypoints = keypoints.reshape(-1, 32, 3)
            if filename.find("mono") > 0:
                keypoints[:, :, 1] *= -1
            else:
                keypoints = keypoints[:, :, (0, 2, 1)]
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
        missing_cols = self._selected_cols - data.keys()
        if len(missing_cols) > 0:
            for col in missing_cols:
                if col.startswith("keypoints"):
                    if col == "keypoints3D":
                        data["keypoints3D"] = self.load_keypointfile(
                            self._data["keypoint3D-filename"][index])
                    else:
                        filename_index = "keypoint" + col[9:] + "-filenames"
                        data[col] = []
                        for filename in self._data[filename_index][index]:
                            data[col].append(self.load_keypointfile(filename))
        return data
