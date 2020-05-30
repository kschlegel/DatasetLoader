import os.path
import numpy as np

from .. import JHMDB

from . import DS_PATH


class TestJHMDB():
    def test_JHMDB(self):
        jhmdb = JHMDB(os.path.join(DS_PATH, "jhmdb"))
        # test full dataset splits
        assert jhmdb.get_data("video-filenames").shape == (928, )
        assert jhmdb.get_data("keypoints", "train", split=1).shape == (660, )
        assert jhmdb.get_data("scales", "test", split=1).shape == (268, )
        assert jhmdb.get_data("actions", "train", split=2).shape == (658, )
        assert jhmdb.get_data("video-filenames", "test",
                              split=2).shape == (270, )
        assert jhmdb.get_data("keypoints", "train", split=3).shape == (663, )
        assert jhmdb.get_data("scales", "test", split=3).shape == (265, )

        # test iterator access
        it = jhmdb.get_iterator(
            ("video-filenames", "keypoints", "scales", "actions"))
        filename, keypoints, scale, action = next(it)
        # check we got the correct first element
        assert keypoints.shape[1:] == (15, 2)
        assert len(scale.shape) == 1
        assert isinstance(action, np.int64)
        assert action >= 0
        assert action < 21

        # test full body subsplits
        jhmdb = JHMDB(os.path.join(DS_PATH, "jhmdb"), full_body_split=True)
        assert jhmdb.get_data("keypoints", "train", split=1).shape == (227, )
        assert jhmdb.get_data("scales", "test", split=1).shape == (89, )
        assert jhmdb.get_data("actions", "train", split=2).shape == (236, )
        assert jhmdb.get_data("video-filenames", "test",
                              split=2).shape == (80, )
        assert jhmdb.get_data("keypoints", "train", split=3).shape == (224, )
        assert jhmdb.get_data("actions", "test", split=3).shape == (92, )
