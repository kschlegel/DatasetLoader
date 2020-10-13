import os
import numpy as np

from .datasetloader import DatasetLoader


class PKUMMD(DatasetLoader):
    """
    PKU-MMD - Peking University Multi-Modality Dataset
    http://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html
    """
    actions = [
        "bow", "brushing hair", "brushing teeth", "check time (from watch)",
        "cheer up", "clapping", "cross hands in front (say stop)",
        "drink water", "drop", "eat meal/snack", "falling",
        "giving something to other person", "hand waving", "handshaking",
        "hopping (one foot jumping)", "hugging other person", "jump up",
        "kicking other person", "kicking something",
        "make a phone call/answer phone", "pat on back of other person",
        "pickup", "playing with phone/tablet",
        "point finger at the other person",
        "pointing to something with finger", "punching/slapping other person",
        "pushing other person", "put on a hat/cap",
        "put something inside pocket", "reading", "rub two hands together",
        "salute", "sitting down", "standing up", "take off a hat/cap",
        "take off glasses", "take off jacket",
        "take out something from pocket", "taking a selfie", "tear up paper",
        "throw", "touch back (backache)",
        "touch chest (stomachache/heart pain)", "touch head (headache)",
        "touch neck (neckache)", "typing on a keyboard",
        "use a fan (with hand or paper)/feeling warm", "wear jacket",
        "wear on glasses", "wipe face", "writing"
    ]
    interactions = [
        "giving something to other person", "handshaking",
        "hugging other person", "kicking other person",
        "pat on back of other person", "point finger at the other person",
        "punching/slapping other person", "pushing other person"
    ]
    landmarks = [
        "pelvis", "spine mid", "neck", "head top", "left shoulder",
        "left elbow", "left wrist", "left hand", "right shoulder",
        "right elbow", "right wrist", "right hand", "left hip", "left knee",
        "left ankle", "left foot", "right hip", "right knee", "right ankle",
        "right foot", "spine shoulder", "left handtip", "left thumb",
        "right handtip", "right thumb"
    ]

    def __init__(self,
                 base_dir,
                 lazy_loading=True,
                 single_person=False,
                 include_missing=True):
        """
        Parameters
        ----------
        base_dir : string
            folder with dataset on disk
        lazy_loading : bool, optional (default is True)
            Only load individual data items when queried
        single_person : bool, optional (default is False)
            If true only load sequences with a single actor (either at init
            time if load_skeletons is True or at access time when calling
            load_keypointfile if load_skeletons is False)
        include_missing : bool, optional (default is True)
            If True missing skeletons are returned as zero-vectors of the same
            shape as a skeleton. If False missing skeletons are returned as an
            empty list.
        """
        self._data_cols = [
            "video-filename",
            "keypoint-filename",
            "action-filename",
            "keypoints3D",
            "actions",
            # The dataset also contains these, to be implemented if/when needed
            # "ir-filenames",
            # "depth-filenames",
        ]
        self._data = {
            "video-filename": [],
            "keypoint-filename": [],
            "action-filename": [],
            # The dataset also contains these, to be implemented if/when needed
            # "ir-filenames": [],
            # "depth-filenames": [],
        }
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            "cross-subject": {
                "train": [],
                "test": []
            },
            "cross-view": {
                "train": [],
                "test": []
            }
        }

        self._single_person = single_person
        self._include_missing = include_missing

        self._length = 0
        filename_list = []
        if single_person:
            interaction_ids = set([
                PKUMMD.actions.index(interaction)
                for interaction in PKUMMD.interactions
            ])
        for filename in os.listdir(os.path.join(base_dir, "Label")):
            filename = filename[:-4]

            # the label files are the easiest ones, use these to check at init
            # time which of the sequences are single person if the dataset is
            # to be restricted to single person
            if single_person:
                with open(os.path.join(base_dir, "Label", filename + ".txt"),
                          "r") as f:
                    action_ids = []
                    for l in f:
                        # Action class ids are one-based in the file
                        action_ids.append(int(l[:l.find(",")]) - 1)
                    if len(interaction_ids.intersection(action_ids)) == len(
                            action_ids):
                        # this is a sequence with 2 persons, skip
                        # TODO: action 17 and 20 occasionally occur as actions
                        # in single person sequences, should maybe fix that
                        continue

            # store short filename for easier identification for split info
            filename_list.append(filename)
            self._data["keypoint-filename"].append(
                os.path.join(base_dir, "Data", "SKELETON_VIDEO",
                             filename + ".txt"))
            self._data["action-filename"].append(
                os.path.join(base_dir, "Label", filename + ".txt"))
            self._data["video-filename"].append(
                os.path.join(base_dir, "Data", "RGB_VIDEO", filename + ".avi"))
            # self._data["ir-filenames"].append(
            #     os.path.join(base_dir, "Data", "IR_VIDEO",
            #                  filename + "-infrared.avi"))
            # self._data["depth-filenames"].append(
            #     os.path.join(base_dir, "Data", "DEPTH_VIDEO",
            #                  filename + "-depth.avi"))
            self._length += 1

        # Load splits information
        for split in self._splits.keys():
            with open(os.path.join(base_dir, "Split", split + ".txt"),
                      "r") as f:
                f.readline()  # Dump the "Training videos:" headline
                line = f.readline()
                trainingset = line[:line.rfind(",")].split(", ")
                for filename in trainingset:
                    for i, sample_file in enumerate(filename_list):
                        if filename == sample_file:
                            self._splits[split]["train"].append(i)
                            break
                f.readline()  # Dump the "Validation videos:" headline
                line = f.readline()
                testset = line[:line.rfind(",")].split(", ")
                for filename in testset:
                    if filename == sample_file:
                        self._splits[split]["test"].append(i)
                        break

        super().__init__(lazy_loading)

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.

        If the dataset is set to single_person this will return None when
        attempting to load a file which contains at least one frame with two
        skeletons.

        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        with open(filename, "r") as f:
            keypoints = []
            for l in f:
                raw_kp = np.array(list(map(float, l.strip().split(" "))))
                frame = []
                if self._single_person:
                    num_people = 0
                for i in range(2):
                    # single person videos are zero buffered, and sometimes
                    # there are no skeletons at all in a frame
                    has_data = np.count_nonzero(
                        raw_kp[i * 75:(i + 1) * 75]) > 0
                    if self._include_missing or has_data:
                        frame.append(raw_kp[i * 75:(i + 1) * 75].reshape(
                            25, 3))
                        if self._single_person and has_data:
                            num_people += 1
                if self._single_person:
                    if num_people > 1:
                        return None
                    elif self._include_missing:
                        frame = frame[0]
                keypoints.append(np.array(frame))
        return np.array(keypoints)

    def load_actionfile(self, filename):
        """
        Load the actions with timestamps from the given file.

        Parameters
        ----------
        filename : string
            Filename of the file containing the action data.
        """
        with open(filename, "r") as f:
            actions = []
            for l in f:
                action_data = list(map(int, l.split(",")[0:3]))
                # Action class ids are one-based in the file
                action_data[0] -= 1
                # Correct occasional order errors in the label file
                if action_data[1] > action_data[2]:
                    action_data[1], action_data[2] = action_data[
                        2], action_data[1]
                actions.append(action_data)
        return actions

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
            if "actions" in self._selected_cols:
                data["actions"] = self.load_actionfile(
                    self._data["action-filename"][index])
        return data
