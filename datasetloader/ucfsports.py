import os
import numpy as np
from tqdm import tqdm

from .datasetloader import DatasetLoader


class UCFSports(DatasetLoader):
    """
    UCF Sports Action Dataset
    https://www.crcv.ucf.edu/data/UCF_Sports_Action.php
    """

    classes = [
        "Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse", "Run",
        "SkateBoarding", "Swing-Bench", "Swing-Side", "Walk"
    ]

    def __init__(self, base_dir):
        """
        Parameters
        ----------
        base_dir : string
            folder with dataset on disk
        """
        self._data_cols = [
            "video-filename", "image-filenames", "bboxes", "action",
            "viewpoint"
        ]
        self._data = {
            "video-filename": [],
            "image-filenames": [],
            "bboxes": [],
            "action": [],
            "viewpoint": []
        }
        # Leave-One-Out cross validation recommended for action recognition
        # Add Action localisation split here for completeness?
        self._splits = None

        super().__init__(lazy_loading=False)

        self._length = 0
        viewpoints = ("", "-Front", "-Side", "-Back", "Angle")
        for cls_id, cls in tqdm(enumerate(UCFSports.classes)):
            for vp in viewpoints:
                cls_folder = os.path.join(base_dir, "ucf action", cls + vp)
                if os.path.exists(cls_folder):
                    video_id = "001"
                    while os.path.exists(os.path.join(cls_folder, video_id)):
                        self._length += 1
                        cur_id = self._length - 1
                        # set filename to blank here as in a few instances it
                        # doesn't exist and should be set to blank in those
                        # cases (this can be fixed with a script from this
                        # package)
                        self._data["video-filename"].append("")
                        self._data["action"].append(cls_id)
                        self._data["image-filenames"].append([])
                        self._data["bboxes"].append([])
                        if len(vp) > 0 and vp[0] == "-":
                            self._data["viewpoint"].append(vp)
                        else:
                            self._data["viewpoint"].append("")
                        filelist = sorted(
                            os.listdir(os.path.join(cls_folder, video_id)))
                        for filename in filelist:
                            if filename.endswith(".avi"):
                                self._data["video-filename"][
                                    cur_id] = os.path.join(
                                        cls_folder, video_id, filename)
                            elif filename.endswith(".jpg"):
                                self._data["image-filenames"][cur_id].append(
                                    os.path.join(cls_folder, video_id,
                                                 filename))
                            elif filename == "gt" or filename == "gt2":
                                if (len(self._data["bboxes"][cur_id]) <
                                        len(filename) - 1):
                                    # gt is read before gt2 so string len
                                    # corresponds to instances read
                                    self._data["bboxes"][self._length -
                                                         1].append([])
                                gt_folder = os.path.join(
                                    cls_folder, video_id, "gt")
                                gt_files = sorted(os.listdir(gt_folder))
                                for gt_file in gt_files:
                                    if gt_file.endswith(".txt"):
                                        with open(
                                                os.path.join(
                                                    gt_folder, gt_file),
                                                "r") as f:
                                            data = f.read()
                                            data = data.split("\t")
                                            self._data["bboxes"][cur_id][
                                                len(filename) - 2].append(
                                                    data[0:4])

                        self._data["bboxes"][cur_id] = np.array(
                            self._data["bboxes"][cur_id])
                        self._data["image-filenames"][cur_id].sort()
                        video_id = int(video_id) + 1
                        video_id = "0" * (3 -
                                          len(str(video_id))) + str(video_id)

        for key in self._data.keys():
            self._data[key] = np.array(self._data[key], dtype=object)
