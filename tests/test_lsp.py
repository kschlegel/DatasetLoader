import os.path
import pytest

from .. import LSP
from .. import LSPExtended

from . import DS_PATH


class TestLSP():
    def test_LSP(self):
        # test loading both, full sized and small images
        folders = ("lsp", "lsp_small")
        for ds in folders:
            lsp = LSP(os.path.join(DS_PATH, ds))
            # check dataset sizes and get_data accessors on different elements
            assert lsp.get_data("image-filenames").shape == (2000, )
            assert lsp.get_data("keypoints", "train").shape == (1000, 14, 3)
            d = lsp.get_data(("image-filenames", "keypoints"), "test")
            assert d[0].shape == (1000, )
            assert d[1].shape == (1000, 14, 3)
            # test iterator access
            it = lsp.get_iterator(("image-filenames", "keypoints"), "train")
            filename, keypoints = next(it)
            # check we got the correct first element
            assert isinstance(filename, str)
            assert keypoints.shape == (14, 3)

    def test_LSPExtended(self):
        lsp = LSPExtended(os.path.join(DS_PATH, "lsp_extended"))
        # check dataset sizes and get_data accessors on different elements
        assert lsp.get_data("image-filenames").shape == (10000, )
        assert lsp.get_data("keypoints").shape == (10000, 14, 3)
        with pytest.raises(Exception):
            lsp.get_data("image-filenames", "train")

        lsp = LSPExtended(os.path.join(DS_PATH, "lsp_extended_improved"),
                          improved=True)
        # check dataset sizes and get_data accessors on different elements
        assert lsp.get_data("image-filenames").shape == (9428, )
        assert lsp.get_data("keypoints").shape == (9428, 14, 3)
