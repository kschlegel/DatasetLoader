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
        "right foot", "shoulder centre", "left handtip", "left thumb",
        "right handtip", "right thumb"
    ]
    splits = ["cross-subject", "cross-view"]

    @classmethod
    def add_argparse_args(cls, parser, default_split=None):
        super().add_argparse_args(parser, default_split)
        child_parser = parser.add_argument_group("PKU-MMD specific arguments")
        child_parser.add_argument(
            "--single_person",
            action="store_true",
            help="Only use sequences with a single actor, no interactions.")
        child_parser.add_argument(
            "--exclude_missing",
            action="store_true",
            help="If given missing skeletons are returned as an empty list "
            "instead of a zero vector in skeleton shape")
        return parser

    def __init__(self,
                 data_path,
                 single_person=False,
                 exclude_missing=False,
                 **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        single_person : bool, optional (default is False)
            If true only load sequences with a single actor (either at init
            time if load_skeletons is True or at access time when calling
            load_keypointfile if load_skeletons is False)
        exclude_missing : bool, optional (default is False)
            If False missing skeletons are returned as zero-vectors of the same
            shape as a skeleton. If True missing skeletons are returned as an
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
            split: {
                "train": [],
                "test": []
            }
            for split in PKUMMD.splits
        }

        self._single_person = single_person
        self._exclude_missing = exclude_missing

        self._length = 0
        filename_list = []
        if single_person:
            interaction_ids = set([
                PKUMMD.actions.index(interaction)
                for interaction in PKUMMD.interactions
            ])
        filelist = sorted(os.listdir(os.path.join(data_path, "Label")))
        for filename in filelist:
            filename = filename[:-4]

            # the label files are the easiest ones, use these to check at init
            # time which of the sequences are single person if the dataset is
            # to be restricted to single person
            if single_person:
                with open(os.path.join(data_path, "Label", filename + ".txt"),
                          "r") as f:
                    action_ids = []
                    for l in f:
                        # Action class ids are one-based in the file
                        action_ids.append(int(l[:l.find(",")]) - 1)
                    if len(interaction_ids.intersection(action_ids)) == len(
                            set(action_ids)):
                        # this is a sequence with 2 persons, skip
                        # TODO: This misses a few sequences which have data for
                        # two skeletons. Are thos true single person with extra
                        # skeleton or true two person sequences?
                        # TODO: action 17 and 20 occasionally occur as actions
                        # in single person sequences, should maybe fix that
                        continue

            # store short filename for easier identification for split info
            filename_list.append(filename)
            self._data["keypoint-filename"].append(
                os.path.join(data_path, "Data", "SKELETON_VIDEO",
                             filename + ".txt"))
            self._data["action-filename"].append(
                os.path.join(data_path, "Label", filename + ".txt"))
            self._data["video-filename"].append(
                os.path.join(data_path, "Data", "RGB_VIDEO",
                             filename + ".avi"))
            # self._data["ir-filenames"].append(
            #     os.path.join(data_path, "Data", "IR_VIDEO",
            #                  filename + "-infrared.avi"))
            # self._data["depth-filenames"].append(
            #     os.path.join(data_path, "Data", "DEPTH_VIDEO",
            #                  filename + "-depth.avi"))
            self._length += 1

        # Load splits information
        for split in self._splits.keys():
            with open(os.path.join(data_path, "Split", split + ".txt"),
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
                    for i, sample_file in enumerate(filename_list):
                        if filename == sample_file:
                            self._splits[split]["test"].append(i)
                            break

        super().__init__(**kwargs)

    def get_single_action_id(self, action_id):
        """
        Remaps the label action id into a set of purely single person actions
        for pure single person tasks.
        """
        if not hasattr(self, "_single_action_ids"):
            counter = 0
            self._single_action_ids = []
            for ac in self.actions:
                if ac in self.interactions:
                    self._single_action_ids.append(None)
                else:
                    self._single_action_ids.append(counter)
                    counter += 1
        return self._single_action_ids[action_id]

    def get_interaction_id(self, action_id):
        """
        Remaps the label action_ids into a set of purely interaction for pure
        interaction tasks.
        """
        if not hasattr(self, "_interaction_ids"):
            counter = 0
            self._interaction_ids = []
            for ac in self.actions:
                if ac in self.interactions:
                    self._interaction_ids.append(counter)
                    counter += 1
                else:
                    self._interaction_ids.append(None)
        return self._interaction_ids[action_id]

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
                    if not self._exclude_missing or has_data:
                        frame.append(raw_kp[i * 75:(i + 1) * 75].reshape(
                            25, 3))
                        if self._single_person and has_data:
                            num_people += 1
                if self._single_person:
                    if num_people > 1:
                        return None
                    elif not self._exclude_missing:
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
                    action_data[1], action_data[2] = (action_data[2],
                                                      action_data[1])
                actions.append(action_data)
        actions.sort(key=lambda t: t[1])
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
