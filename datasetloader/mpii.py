import os.path
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat, matlab

from .datasetloader import DatasetLoader


class MPII(DatasetLoader):
    """
    MPII Human Pose Dataset
    http://human-pose.mpi-inf.mpg.de/#overview
    """
    landmarks = [
        "right ankle", "right knee", "right hip", "left hip", "left knee",
        "left ankle", "pelvis", "thorax", "neck", "head top", "right wrist",
        "right elbow", "right shoulder", "left shoulder", "left elbow",
        "left wrist"
    ]
    splits = ["default"]
    reference_scale = 200

    @classmethod
    def add_argparse_args(cls, parser, default_split=None):
        super().add_argparse_args(parser, default_split)
        child_parser = parser.add_argument_group("MPII specific arguments")
        child_parser.add_argument("--single_person",
                                  action="store_true",
                                  help="Load the small cropped images.")
        return parser

    def __init__(self, data_path, single_person=False, **kwargs):
        """
        Parameters
        ----------
        data_path : string
            folder with dataset on disk
        single_person : bool, optional (default is False)
            If true load the small, readily cropped images and corresponding
            keypoints.
        """
        self._data_cols = [
            "image-filename", "keypoints2D", "scale", "centre", "head_bbox"
        ]
        self._data = {
            "image-filename": [],
            "keypoints2D": [],
            "scale": [],
            "centre": [],
            "head_bbox": []
        }
        self._splits = {
            split: {
                "train": [],
                "test": []
            }
            for split in MPII.splits
        }

        kwargs["no_lazy_loading"] = True
        super().__init__(**kwargs)

        print("Loading the data file. This may take a while...")
        self._length = 0
        raw_data = loadmat(os.path.join(data_path,
                                        "mpii_human_pose_v1_u12_1.mat"),
                           struct_as_record=False,
                           squeeze_me=True)
        for img_id, is_training in enumerate(
                tqdm(raw_data["RELEASE"].img_train)):
            if single_person:
                # single_person ids are 1-indexed, arrays are 0-indexed
                # => need to subtract one
                person_ids = raw_data["RELEASE"].single_person[img_id] - 1
                if isinstance(person_ids, int):
                    person_ids = [person_ids]
            else:
                # annorect can be an array of objects or an object of a single
                # person. I am going to wrap this into a single element list
                # below
                try:
                    person_ids = list(
                        range(
                            len(raw_data["RELEASE"].annolist[img_id].annorect))
                    )
                except TypeError:
                    person_ids = [0]

            if isinstance(raw_data["RELEASE"].annolist[img_id].annorect,
                          np.ndarray):
                annotations = raw_data["RELEASE"].annolist[img_id].annorect
            else:
                # if annorect is not an array it is an object of a single
                # person. Wrap in a list so I can always access in a loop
                annotations = [raw_data["RELEASE"].annolist[img_id].annorect]

            keypoints = []
            centres = []
            scales = []
            head_bboxes = []
            for person_id in person_ids:
                # There are many person entries with mistakes. Check each
                # property for correctness, read it out if correct, fix/ignore
                # what can be fixed/ignored and skip entries that aren't useful
                if "objpos" in annotations[person_id]._fieldnames:
                    if isinstance(annotations[person_id].objpos, np.ndarray):
                        # If this is an array, not a struct its always = []
                        centre = None
                    else:
                        centre = np.array([
                            annotations[person_id].objpos.x,
                            annotations[person_id].objpos.y
                        ])
                else:
                    centre = None
                if "scale" in annotations[person_id]._fieldnames:
                    scale = 1 / annotations[person_id].scale
                else:
                    scale = None
                if "x1" in annotations[person_id]._fieldnames:
                    head_bbox = [
                        annotations[person_id].x1, annotations[person_id].y1,
                        annotations[person_id].x2, annotations[person_id].y2
                    ]
                else:
                    head_bbox = None
                if is_training == 0:
                    # testset instances don't have keypoint data
                    data_pt = None
                    if single_person and (centre is None or scale is None):
                        # in single person I need centre and scale to know
                        # where to look
                        continue
                else:
                    if "annopoints" in annotations[person_id]._fieldnames:
                        if isinstance(annotations[person_id].annopoints,
                                      np.ndarray):
                            # if this is an array, not a struct its always = []
                            # this is useless
                            continue
                        elif isinstance(
                                annotations[person_id].annopoints.point,
                                matlab.mio5_params.mat_struct):
                            # This seems to be occur once in the dataset, the
                            # struct then contains a single 'head top' point,
                            # which appears to be garbage though
                            continue
                        else:
                            # Here I have a proper point
                            data_pt = [[0, 0, 0]] * len(MPII.landmarks)
                            for pt in annotations[person_id].annopoints.point:
                                if isinstance(pt.is_visible, np.ndarray):
                                    # TODO: I don't understand what is_visible
                                    # being a list means
                                    pt.is_visible = -1
                                data_pt[pt.id] = [pt.x, pt.y, pt.is_visible]
                            data_pt = np.array(data_pt, dtype=np.float64)
                    else:
                        # if this is a training image and I don't have keypoint
                        # data then it is useless
                        continue
                keypoints.append(data_pt)
                centres.append(centre)
                scales.append(scale)
                head_bboxes.append(head_bbox)

            # Save split information
            if is_training == 0:
                if len(keypoints) == 0:
                    continue
                self._splits["default"]["test"].append(self._length)
            else:
                if len(keypoints) == 0:
                    # sometimes for some reason there are images with not a
                    # single skeleton
                    continue
                self._splits["default"]["train"].append(self._length)

            self._data["image-filename"].append(
                os.path.join(data_path, "images",
                             raw_data["RELEASE"].annolist[img_id].image.name))
            self._data["keypoints2D"].append(np.array(keypoints))
            self._data["centre"].append(np.array(centres))
            self._data["scale"].append(np.array(scales))
            self._data["head_bbox"].append(np.array(head_bboxes))
            self._length += 1
        for key in self._data.keys():
            self._data[key] = np.array(self._data[key], dtype=object)
