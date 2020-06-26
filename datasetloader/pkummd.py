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
    landmarks = []

    def __init__(self,
                 base_dir,
                 load_skeletons=True,
                 single_person=False,
                 include_missing=True):
        """
        Parameters
        ----------
        base_folder : string
            folder with dataset on disk
        load_skeletons : bool, optional (default is True)
            This dataset is large so that on machines with not a lot of memory
            it might not be feasible to hold all skeletons in memory. Set t his
            to false to no load all skeleton sequences immediately.
        single_person : bool, optional (default is False)
            If true only load sequences with a single actor (either at init
            time if load_skeletons is True or at access time when calling
            load_keypointfile if load_skeletons is False)
        """
        super().__init__()
        # lists to hold all information contained in the dataset
        # TODO: the dataset got more information, include more?
        self._data = {
            "video-filenames": [],
            "skeleton-filenames": [],
            "ir-filenames": [],
            "depth-filenames": [],
            "keypoints": [],
            "actions": []
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
        self._default_split = "cross-subject"

        self._single_person = single_person
        self._include_missing = include_missing

        filename_list = []
        for filename in os.listdir(os.path.join(base_dir, "Label")):
            filename = filename[:-4]
            # store the short sequence names here to easily match for ids when
            # loading the splits later

            # load skeleton data if requested
            if load_skeletons:
                keypoints = self.load_keypointfile(
                    os.path.join(base_dir, "Data", "SKELETON_VIDEO",
                                 filename + ".txt"))
                # load_keypointfile returns None if single_person is True and
                # the given file contains at least one frame with more than one
                # person => skip this one
                if keypoints is None:
                    continue
                self._data["keypoints"].append(keypoints)
            # Add the skeleton filename (whether we just loaded them or not)
            self._data["skeleton-filenames"].append(
                os.path.join(base_dir, "Data", "SKELETON_VIDEO",
                             filename + ".txt"))

            # At this point we know if an item is skipped or not, set all
            # general info vars
            filename_list.append(filename)
            self._length += 1

            # load action data
            with open(os.path.join(base_dir, "Label", filename + ".txt"),
                      "r") as f:
                actions = []
                for l in f:
                    data = list(map(int, l.split(",")[0:3]))
                    # Action classes are 1-labelled in the file, 0-labelled in
                    # the array => subtract 1
                    data[0] -= 1
                    # Correct order errors in the label file
                    if data[1] > data[2]:
                        data[1], data[2] = data[2], data[1]
                    actions.append(data)
                self._data["actions"].append(np.array(actions))

            # fill in filenames for all types of video data
            self._data["video-filenames"].append(
                os.path.join(base_dir, "Data", "RGB_VIDEO", filename + ".avi"))
            self._data["ir-filenames"].append(
                os.path.join(base_dir, "Data", "IR_VIDEO",
                             filename + "-infrared.avi"))
            self._data["depth-filenames"].append(
                os.path.join(base_dir, "Data", "DEPTH_VIDEO",
                             filename + "-depth.avi"))

        # Load splits information
        for split in self._splits.keys():
            with open(os.path.join(base_dir, "Split", split + ".txt"),
                      "r") as f:
                f.readline()  # Dump the "Training videos:" headline
                line = f.readline()
                trainingset = line[:line.rfind(",")].split(", ")
                for filename in trainingset:
                    i = filename_list.index(filename)
                    self._splits[split]["train"].append(i)
                f.readline()  # Dump the "Validation videos:" headline
                line = f.readline()
                testset = line[:line.rfind(",")].split(", ")
                for filename in testset:
                    i = filename_list.index(filename)
                    self._splits[split]["test"].append(i)

        for key in self._data.keys():
            self._data[key] = np.array(self._data[key], dtype=object)

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
