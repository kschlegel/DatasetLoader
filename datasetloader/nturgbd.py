"""
This class is currently rather incomplete, only contains the basic
functionality to get the skeletons.
"""

import os

import numpy as np

from .datasetloader import DatasetLoader


class NTURGBD(DatasetLoader):
    """
    NTU RGB+D Action Recognition Dataset
    http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
    """
    actions = [
        "drink water", "eat meal/snack", "brushing teeth", "brushing hair",
        "drop", "pickup", "throw", "sitting down",
        "standing up (from sitting position)", "clapping", "reading",
        "writing", "tear up paper", "wear jacket", "take off jacket",
        "wear a shoe", "take off a shoe", "wear on glasses",
        "take off glasses", "put on a hat/cap", "take off a hat/cap",
        "cheer up", "hand waving", "kicking something", "reach into pocket",
        "hopping (one foot jumping)", "jump up",
        "make a phone call/answer phone", "playing with phone/tablet",
        "typing on a keyboard", "pointing to something with finger",
        "taking a selfie", "check time (from watch)", "rub two hands together",
        "nod head/bow", "shake head", "wipe face", "salute",
        "put the palms together", "cross hands in front (say stop)",
        "sneeze/cough", "staggering", "falling", "touch head (headache)",
        "touch chest (stomachache/heart pain)", "touch back (backache)",
        "touch neck (neckache)", "nausea or vomiting condition",
        "use a fan (with hand or paper)/feeling warm",
        "punching/slapping other person", "kicking other person",
        "pushing other person", "pat on back of other person",
        "point finger at the other person", "hugging other person",
        "giving something to other person", "touch other person's pocket",
        "handshaking", "walking towards each other",
        "walking apart from each other", "put on headphone",
        "take off headphone", "shoot at the basket", "bounce ball",
        "tennis bat swing", "juggling table tennis balls", "hush (quite)",
        "flick hair", "thumb up", "thumb down", "make ok sign",
        "make victory sign", "staple book", "counting money", "cutting nails",
        "cutting paper (using scissors)", "snapping fingers", "open bottle",
        "sniff (smell)", "squat down", "toss a coin", "fold paper",
        "ball up paper", "play magic cube", "apply cream on face",
        "apply cream on hand back", "put on bag", "take off bag",
        "put something into a bag", "take something out of a bag",
        "open a box", "move heavy objects", "shake fist", "throw up cap/hat",
        "hands up (both hands)", "cross arms", "arm circles", "arm swings",
        "running on the spot", "butt kicks (kick backward)", "cross toe touch",
        "side kick", "yawn", "stretch oneself", "blow nose",
        "hit other person with something", "wield knife towards other person",
        "knock over other person (hit with body)", "grab other person’s stuff",
        "shoot at other person with a gun", "step on foot", "high-five",
        "cheers and drink", "carry something with other person",
        "take a photo of other person", "follow other person",
        "whisper in other person’s ear", "exchange things with other person",
        "support somebody with hand",
        "finger-guessing game (playing rock-paper-scissors)"
    ]

    landmarks = [
        "pelvis", "mid torso", "neck", "head top", "left shoulder",
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
        child_parser = parser.add_argument_group(
            "NTU RGB+D specific arguments")
        child_parser.add_argument(
            "--ntu120",
            action="store_true",
            help="Load all 120 instead of first 60 classes of NTU RGB+D")
        child_parser.add_argument(
            "--include_missing_skeletons",
            action="store_true",
            help="Also include the samples with missing skeletons")
        return parser

    def __init__(self,
                 data_path,
                 ntu120=False,
                 include_missing_skeletons=False,
                 **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        ntu120 : bool, optional (default is False)
            Load all 120 instead of first 60 classes of NTU RGB+D
        include_missing_skeletons : bool, optional (default is False)
            If True also include all samples that have missing skeletons
        """
        self._data_cols = [
            "keypoint-filename",
            "keypoints3D",
            "keypoints2D",
            "keypoints_depth",
            "action",
            # The dataset also contains these, to be implemented if/when needed
            # "video-filename",
            # "depth-filenames",
        ]
        self._data = {
            "keypoint-filename": [],
            "action": []
            # The dataset also contains these, to be implemented if/when needed
            # "video-filename": [],
            # "depth-filenames": [],
        }

        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in NTURGBD.splits
        }

        self._length = 0

        # Load list of of samples to ignore
        missing_skeletons = []
        if not include_missing_skeletons:
            filenames = ["NTU_RGBD_samples_with_missing_skeletons.txt"]
            if ntu120:
                filenames.append(
                    "NTU_RGBD120_samples_with_missing_skeletons.txt")
            for filename in filenames:
                with open(os.path.join(data_path, filename), "r") as f:
                    for i in range(3):
                        f.readline()
                    for line in f:
                        missing_skeletons.append(line.strip())

        skeleton_dir = os.path.join(data_path, "nturgb+d_skeletons")
        for filename in os.listdir(skeleton_dir):
            subject_id = int(filename[1:4])
            if ntu120 or subject_id <= 17:
                if filename[:-9] in missing_skeletons:
                    continue
                self._data["keypoint-filename"].append(
                    os.path.join(skeleton_dir, filename))
                action_id = int(filename[17:20]) - 1
                self._data["action"].append(action_id)
                if subject_id in (1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
                                  25, 27, 28, 31, 34, 35, 38):
                    self._splits["cross-subject"]["train"].append(self._length)
                else:
                    self._splits["cross-subject"]["test"].append(self._length)
                camera_id = int(filename[5:8])
                if camera_id != 1:
                    self._splits["cross-view"]["train"].append(self._length)
                else:
                    self._splits["cross-view"]["train"].append(self._length)
                self._length += 1

        super().__init__(**kwargs)

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.

        For reference, the format of the skeleton-file is:
        num_frames
        num_bodies (in frame 0)
        body_0_info: ID, clipped_edges, hand_left_confidence, hand_left_state,
                     hand_right_confidence, hand_right_state, is_restricted,
                     lean_x, lean_y, tracking_state
        num_joints (of person 0 in frame 0, constant 25)
        joint_0_info: x,y,z,depth_x,depth_y,rgb_x,rgb_y,orientation_w,
                      orientation_x,orientation_y,orientation_z,tracking_state
        . . .
        body_1_info...
        . . .
        num_bodies (in frame 1)
        . . .

        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        with open(filename, "r") as skel_file:
            data = skel_file.readlines()
        num_frames = int(data[0][:-1])
        if "keypoints3D" in self._selected_cols:
            persons3d = np.zeros((0, num_frames, 25, 3))
        if "keypoints2D" in self._selected_cols:
            persons2d = np.zeros((0, num_frames, 25, 2))
        if "keypoints_depth" in self._selected_cols:
            persons_depth = np.zeros((0, num_frames, 25, 2))
        existing_persons = 0
        data_index = 0
        for frame_id in range(num_frames):
            data_index += 1
            person_count = int(data[data_index][:-1])
            if existing_persons > 0 and person_count != existing_persons:
                # print("INCONSISTENT PERSON COUNT", existing_persons,
                #       person_count)
                # Why do these occur? What do they mean/do they matter?
                pass
            if existing_persons < person_count:
                add_persons = person_count - existing_persons
                if "keypoints3D" in self._selected_cols:
                    persons3d = np.append(persons3d,
                                          np.zeros((add_persons, num_frames,
                                                    25, 3)),
                                          axis=0)
                if "keypoints2D" in self._selected_cols:
                    persons2d = np.append(persons2d,
                                          np.zeros((add_persons, num_frames,
                                                    25, 2)),
                                          axis=0)
                if "keypoints_depth" in self._selected_cols:
                    persons_depth = np.append(persons_depth,
                                              np.zeros((add_persons,
                                                        num_frames, 25, 2)),
                                              axis=0)
                existing_persons += add_persons
            for person_id in range(person_count):
                data_index += 2
                num_joints = int(data[data_index][:-1])
                if num_joints != len(self.landmarks):
                    print("WRONG JOINT COUNT!", num_joints)
                for joint_id in range(num_joints):
                    data_index += 1
                    jointinfo = data[data_index][:-1].split(' ')
                    jointinfo = np.array(list(map(float, jointinfo)))
                    if "keypoints3D" in self._selected_cols:
                        persons3d[person_id][frame_id,
                                             joint_id] = jointinfo[:3]
                    if "keypoints2D" in self._selected_cols:
                        persons2d[person_id][frame_id,
                                             joint_id] = jointinfo[5:7]
                    if "keypoints_depth" in self._selected_cols:
                        persons_depth[person_id][frame_id,
                                                 joint_id] = jointinfo[3:5]
        persons = []
        if "keypoints3D" in self._selected_cols:
            if persons3d.shape[0] == 0:
                print("Empty person array", filename)
            persons.append(persons3d)
        if "keypoints2D" in self._selected_cols:
            persons.append(persons2d)
        if "keypoints_depth" in self._selected_cols:
            persons.append(persons_depth)
        return persons

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
            if ("keypoints3D" in self._selected_cols
                    or "keypoints2D" in self._selected_cols
                    or "keypoints_depth" in self._selected_cols):
                keypoints = self.load_keypointfile(
                    self._data["keypoint-filename"][index])
                if "keypoints_depth" in self._selected_cols:
                    data["keypoints_depth"] = keypoints.pop()
                if "keypoints2D" in self._selected_cols:
                    data["keypoints2D"] = keypoints.pop()
                if "keypoints3D" in self._selected_cols:
                    data["keypoints3D"] = keypoints.pop()
        return data
