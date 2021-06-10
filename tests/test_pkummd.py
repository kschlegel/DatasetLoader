import os.path

from datasetloader import PKUMMD

from . import DS_PATH


class TestPKUMMD():
    def test_PKUMMD(self):
        pku = PKUMMD(os.path.join(DS_PATH, "pku-mmd"), load_skeletons=False)
        assert pku.get_data("video-filenames").shape == (1076, )
        assert pku.get_data("skeleton-filenames", "train",
                            "cross-subject").shape == (944, )
        assert pku.get_data("actions", "test",
                            "cross-subject").shape == (132, )
        assert pku.get_data("video-filenames", "train",
                            "cross-view").shape == (717, )
        assert pku.get_data("skeleton-filenames", "test",
                            "cross-view").shape == (359, )

        it = pku.get_iterator(("skeleton-filenames", "actions"))
        skeleton_file, action = next(it)
        assert action.shape[1:] == (3, )
        keypoints = pku.load_keypointfile(skeleton_file)
        assert keypoints[0].shape[1:] == (25, 3)
