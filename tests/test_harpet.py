import os.path
import numpy as np
import pytest

from datasetloader import HARPET

from . import DS_PATH


class TestHARPET():
    def test_HARPET(self):
        harpet = HARPET(os.path.join(DS_PATH, "harpet"))
        # check dataset sizes
        assert harpet.get_data("image-filenames").shape == (424, 3)
        assert harpet.get_data("keypoints", "train").shape == (297, 3, 18, 2)
        assert harpet.get_data("actions", "valid").shape == (64, )
        # checked all single accessors above, now check accessing two
        # elements at once
        d = harpet.get_data(("keypoints", "actions"), "test")
        assert d[0].shape == (63, 3, 18, 2)
        assert d[1].shape == (63, )
        # check iterator access on non-existing element raises an exception
        with pytest.raises(Exception):
            it = harpet.get_iterator(
                ("image-filenames", "keypoints", "scales", "actions"))
            filename, keypoints, scales, action = next(it)
        # check iterator access on subset
        it = harpet.get_iterator(("image-filenames", "keypoints", "actions"),
                                 "train")
        filename, keypoints, action = next(it)
        # check we got the correct first element
        assert len(filename) == 3
        assert isinstance(filename[0], str)
        assert keypoints.shape == (3, 18, 2)
        assert isinstance(action, np.int64)
        assert action >= 0
        assert action < 4
