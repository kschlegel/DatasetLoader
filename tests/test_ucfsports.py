import os.path
import numpy as np

from .. import UCFSports

from . import DS_PATH


class TestUCFSports():
    # These tests assume the dataset issues that can easily be fixed (using the
    # script in this project) have been fixed
    def test_UCFSports(self):
        ucf = UCFSports(os.path.join(DS_PATH, "ucf_sports_actions"))
        assert ucf.get_data("viewpoints").shape == (150, )

        expected_counts = [14, 18, 20, 6, 12, 13, 12, 20, 13, 22]
        counts = np.bincount(ucf.get_data("actions"))
        for cls_id in range(10):
            assert expected_counts[cls_id] == counts[cls_id]

        image_names, bboxes = ucf.get_data(("image-filenames", "bboxes"))
        no_bboxes = 0
        for i in range(150):
            assert len(image_names[i]) > 0
            # All golf swing back and golf swing front 2 don't come with jpgs,
            # just the avi by default. Trhis can be fixed with the script
            # included in this project
            if bboxes[i].shape[0] == 0:
                # these are the 6 lifting videos and walking 5,19,20,21
                no_bboxes += 1
            else:
                # if bboxes are given the number of bbox entries should equal
                # the number of frames
                assert bboxes[i].shape[1] == len(image_names[i])
        assert no_bboxes == 10

        for filename in ucf.get_iterator("video-filenames"):
            assert len(filename) > 0
            # Diving 8-14 don't come with a video file by default
            # This can be fixed with the script included in this project
