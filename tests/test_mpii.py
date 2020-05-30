import os.path

from .. import MPII

from . import DS_PATH


class TestMPII():
    def test_MPII(self):
        mpii = MPII(os.path.join(DS_PATH, "mpii_human_pose"))
        # check dataset sizes and get_data accessors on different elements
        assert mpii.get_data("image-filenames").shape == (22155, )
        # keypoint shape is not known here as some images contain several
        # people
        assert mpii.get_data("keypoints", "train").shape == (15247, )
        assert mpii.get_data("scales", "test").shape == (6908, )

        d = mpii.get_data(("centres", "head_bboxes"), "test")
        assert d[0].shape == (6908, )
        assert d[1].shape == (6908, )
        # test iterator access
        it = mpii.get_iterator(("image-filenames", "keypoints", "scales",
                                "centres", "head_bboxes"), "train")
        filename, keypoints, scale, centre, head_bbox = next(it)
        # check we got the correct first element
        assert isinstance(filename, str)
        assert len(keypoints.shape) == 3
        assert keypoints[0].shape == (16, 3)
        assert scale.shape == (keypoints.shape[0], )
        assert len(centre.shape) == 2
        assert centre.shape == (keypoints.shape[0], 2)
        assert head_bbox.shape == (keypoints.shape[0], 4)
