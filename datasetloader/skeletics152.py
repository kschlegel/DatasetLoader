import os
import json
import re

import numpy as np

from .datasetloader import DatasetLoader


class Skeletics152(DatasetLoader):
    """
    Skeletics152
    https://github.com/skelemoa/quovadis/tree/master/skeletics-152
    """
    actions = [
        "tai chi", "walking with crutches", "juggling soccer ball",
        "dribbling basketball", "catching or throwing softball",
        "playing organ", "catching or throwing baseball", "taking photo",
        "robot dancing", "playing cello", "training dog", "tiptoeing",
        "pumping fist", "contact juggling", "skipping stone",
        "playing piccolo", "front raises", "dancing gangnam style",
        "head stand", "side kick", "push up", "riding a bike",
        "stretching arm", "bench pressing", "casting fishing line",
        "disc golfing", "playing accordion", "playing ping pong",
        "air drumming", "chopping wood", "country line dancing",
        "tightrope walking", "catching or throwing frisbee",
        "exercising with an exercise ball", "riding mechanical bull",
        "rope pushdown", "digging", "situp", "shearing sheep",
        "breaking boards", "playing guitar", "golf driving", "playing drums",
        "riding mule", "playing ukulele", "hula hooping", "javelin throw",
        "singing", "clapping", "snatch weight lifting", "snorkeling",
        "golf chipping", "clean and jerk", "sweeping floor",
        "passing soccer ball", "long jump", "playing harp", "tackling",
        "skiing slalom", "playing field hockey", "golf putting",
        "playing harmonica", "swinging on something", "longboarding",
        "jumping into pool", "snowboarding", "bowling", "shoot dance",
        "saluting", "deadlifting", "jumping jacks",
        "bouncing on bouncy castle", "playing squash or racquetball",
        "shot put", "yoga", "falling off chair", "playing tennis",
        "swinging baseball bat", "moon walking", "archery", "stretching leg",
        "checking watch", "climbing a rope", "pull ups", "tobogganing",
        "hitting baseball", "pushing cart", "paragliding",
        "running on treadmill", "high fiving", "cumbia",
        "battle rope training", "playing oboe", "sword swallowing",
        "biking through snow", "playing violin", "jogging", "opening door",
        "chiseling stone", "abseiling", "hammer throw", "standing on hands",
        "sword fighting", "dunking basketball", "reading book",
        "jumpstyle dancing", "cutting cake", "belly dancing", "exercising arm",
        "pushing wheelbarrow", "pushing car", "lunge", "hopscotch",
        "ski ballet", "combing hair", "playing hand clapping games",
        "playing trombone", "clam digging", "slacklining", "climbing tree",
        "walking on stilts", "eating watermelon", "squat",
        "skiing crosscountry", "ice skating", "falling off bike", "krumping",
        "punching bag", "directing traffic", "playing badminton",
        "backflip (human)", "bouncing ball (not juggling)", "fencing (sport)",
        "hugging (not baby)", "mountain climber (exercise)",
        "passing American football (not in game)", "playing saxophone",
        "pulling rope (game)", "punching person (boxing)", "riding camel",
        "riding unicycle", "rock climbing", "roller skating", "salsa dancing",
        "shooting goal (soccer)", "tango dancing", "tap dancing",
        "tapping guitar", "trimming shrubs", "using a sledge hammer",
        "wrestling", "zumba"
    ]
    landmarks = [
        "nose",
        "neck",
        "right shoulder",
        "right elbow",
        "right wrist",
        "left shoulder",
        "left elbow",
        "left wrist",
        "pelvis",
        "right hip",
        "right knee",
        "right ankle",
        "left hip",
        "left knee",
        "left ankle",
        "right eye",
        "left eye",
        "right ear",
        "left ear",
        "left big toe",
        "left small toe",
        "left heel",
        "right big toe",
        "right small toe",
        "right heel",  # Up to here OpenPose landmarks
        'rankle',  # 25
        'rknee',  # 26
        'rhip',  # 27
        'lhip',  # 28
        'lknee',  # 29
        'lankle',  # 30
        'rwrist',  # 31
        'relbow',  # 32
        'rshoulder',  # 33
        'lshoulder',  # 34
        'lelbow',  # 35
        'lwrist',  # 36
        'neck',  # 37
        'headtop',  # 38
        'hip',  # 39 'Pelvis (MPII)', # 39
        'thorax',  # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',  # 41
        'Jaw (H36M)',  # 42
        'Head (H36M)',  # 43
        'nose',  # 44
        'leye',  # 45 'Left Eye', # 45
        'reye',  # 46 'Right Eye', # 46
        'lear',  # 47 'Left Ear', # 47
        'rear',  # 48 'Right Ear', # 48
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
            "keypoint-filename", "keypoints3D", "keypoints2D", "action",
            "frame_ids", "youtube_id", "youtube-timerange", "pred_cams",
            "bboxes"
        ]
        self._data = {
            "keypoint-filename": [],
            "action": [],
            "youtube_id": [],
            "youtube-timerange": []
        }

        self._splits = {"default": {"train": [], "test": []}}

        self._length = 0
        youtube_regex = re.compile(r"(.*)_(\d{6})_(\d{6}).json")
        for subset, split in (("training", "train"), ("validation", "test")):
            for action in self.actions:
                data_path = os.path.join(base_dir, subset, action)
                for filename in os.listdir(data_path):
                    self._data["keypoint-filename"].append(
                        os.path.join(data_path, filename))
                    self._data["action"].append(action)

                    m = youtube_regex.match(filename)
                    self._data["youtube_id"].append(m.group(1))
                    self._data["youtube-timerange"].append(
                        (int(m.group(2)), int(m.group(3))))

                    self._splits["default"][split].append(self._length)
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
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            print("Json decoder error:", filename)
            return False, None
        if len(data) == 0:
            # print("No person?", filename)
            return None, None
        keypoints = []
        frame_ids = []
        pred_cams = []
        bboxes = []
        for key, val in data.items():
            keypoints.append(np.array(val["joints3d"]))
            frame_ids.append(np.array(val["frame_ids"]))
            pred_cams.append(np.array(val["pred_cam"]))
            bboxes.append(np.array(val["bboxes"]))
        args = {}
        if len(keypoints) > 1:
            prev_len = len(keypoints[0])
            for i in range(1, len(keypoints)):
                if len(keypoints[i]) != prev_len:
                    args["dtype"] = object
                    break
        keypoints = np.array(keypoints, **args)
        frame_ids = np.array(frame_ids, **args)
        pred_cams = np.array(pred_cams, **args)
        bboxes = np.array(bboxes, **args)
        return keypoints, frame_ids, pred_cams, bboxes

    def _project_keypoints(self, keypoints3D, pred_cams):
        keypoints2D = []
        for person_id in range(keypoints3D.shape[0]):
            keypoints2D.append(
                Skeletics152._projection(keypoints3D[person_id],
                                         pred_cams[person_id]))
        args = {}
        if len(keypoints2D) > 1:
            prev_len = len(keypoints2D[0])
            for i in range(1, len(keypoints2D)):
                if len(keypoints2D[i]) != prev_len:
                    args["dtype"] = object
                    break
        keypoints2D = np.array(keypoints2D, **args)
        return keypoints2D

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
            if any(item in self._selected_cols
                   for item in ("keypoints3D", "keypoints2D", "frame_ids",
                                "pred_cams", "bboxes")):
                keypoints, frame_ids, pred_cams, bboxes = self.load_keypointfile(
                    self._data["keypoint-filename"][index])
                if "keypoints3D" in self._selected_cols:
                    data["keypoints3D"] = keypoints
                if "frame_ids" in self._selected_cols:
                    data["frame_ids"] = frame_ids
                if "pred_cams" in self._selected_cols:
                    data["pred_cams"] = pred_cams
                if "bboxes" in self._selected_cols:
                    data["bboxes"] = bboxes
                if "keypoints2D" in self._selected_cols:
                    data["keypoints2D"] = self._project_keypoints(
                        keypoints, pred_cams)
        return data

    ###########################################################################
    # Projection of 3d keypoints onto 2d image plane (with respect to the bbox)
    # using the weak perspective camera approximation.
    # This implementation is a direct copy, simply replacing every PyTorch
    # function by the equivalent numpy function of the original PyTorch
    # implementation in the VIBE repository, here:
    # https://github.com/mkocabas/VIBE/blob/945fd109eaace037b38c56e22ec235a9d3c5100a/lib/models/spin.py#L426-L439

    @staticmethod
    def _projection(pred_joints, pred_camera):
        pred_cam_t = np.stack([
            pred_camera[:, 1], pred_camera[:, 2], 2 * 5000. /
            (224. * pred_camera[:, 0] + 1e-9)
        ],
                              axis=-1)

        batch_size = pred_joints.shape[0]
        camera_center = np.zeros((batch_size, 2))
        pred_keypoints_2d = Skeletics152._perspective_projection(
            pred_joints,
            rotation=np.repeat(np.expand_dims(np.eye(3), 0), batch_size, 0),
            translation=pred_cam_t,
            focal_length=5000.,
            camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)

        return pred_keypoints_2d

    @staticmethod
    def _perspective_projection(points, rotation, translation, focal_length,
                                camera_center):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]
        K = np.zeros([batch_size, 3, 3])
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = np.einsum('bij,bkj->bki', rotation, points)
        points = points + np.expand_dims(translation, 1)

        # Apply perspective distortion
        projected_points = points / np.expand_dims(points[:, :, -1], -1)

        # Apply camera intrinsics
        projected_points = np.einsum('bij,bkj->bki', K, projected_points)

        return projected_points[:, :, :-1]
