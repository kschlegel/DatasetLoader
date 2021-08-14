from abc import ABC

from .datasubset import DataSubset


class DatasetLoader(ABC):
    """
    Base class for all dataset loaders to provide a common interface for
    retrieving the data out of the dataset object.
    """
    def __init__(self, lazy_loading):
        self._selected_cols = []
        self._lazy = lazy_loading
        self._cur_split = None
        if not self._lazy:
            self._load_all()

    def __len__(self):
        return self._length

    @classmethod
    def add_args(cls, parser, default_split=None):
        parser.add_argument('-p',
                            '--path',
                            type=str,
                            required=True,
                            help="Path to the dataset")

        if cls.splits is not None:
            if default_split is None:
                default_split = cls.splits[0]
            parser.add_argument(
                '-s',
                '--split',
                type=str,
                default=default_split,
                choices=cls.splits,
                help="Dataset split to use (Default is {})".format(
                    default_split))

    def set_split(self, split_name):
        if split_name not in self._splits:
            raise KeyError("This dataset has no split '" + split_name + "'!")
        self._cur_split = split_name

    @property
    def trainingset(self):
        return self._datasubset("train")

    @property
    def validationset(self):
        return self._datasubset("valid")

    @property
    def testset(self):
        return self._datasubset("test")

    def _datasubset(self, subset):
        if self._cur_split is None:
            raise KeyError("A split must be selected using '.set_split' "
                           "before accessing a subset")
        elif self._cur_split not in self._splits:
            raise KeyError("This dataset has no split '" + self._cur_split +
                           "'!")
        if subset not in self._splits[self._cur_split]:
            raise KeyError("The split '" + self._cur_split +
                           "' doesn't have a subset " + subset)
        return DataSubset(self, self._cur_split, subset)

    def set_cols(self, *args):
        """
        Sets the data columns to be returned on query.

        Overwrites any previous selection.

        Parameters
        ----------
        strings of data columns to be used.
        """
        for data_key in args:
            if data_key not in self._data_cols:
                raise KeyError("This dataset does not have '" + data_key +
                               "'information.")
        self._selected_cols = list(args)

    def select_col(self, col):
        """
        Add the given column to the list of data returned on query.

        Parameters
        ----------
        col : string
            Name of the data column to be selected
        """
        if col not in self._data_cols:
            raise KeyError("This dataset does not have '" + col +
                           "'information.")
        if col not in self._selected_cols:
            self._selected_cols.append(col)

    def deselect_col(self, col):
        """
        Remove the given column from the list of data returned on query.

        Parameters
        ----------
        col : string
            Name of the data column to be removed from the selection.
        """
        if col in self._selected_cols:
            self._selected_cols.remove(col)

    def has_col(self, col):
        """
        Check whether the dataset has the given type of data.

        Pass in a string key to check its validity for use as key to query
        data.

        Parameters
        ----------
        col : string
            Name of the data column to be checked
        """
        return (col in self._data_cols)

    def __getitem__(self, index):
        """
        Indexing access to the dataset.

        Provides the non-lazy access only. Any dataset to offer lazy access
        must implement the lazy access for any lazy parts manually.
        """
        return {
            data_key: self._data[data_key][index]
            for data_key in self._selected_cols if data_key in self._data
        }

    def iterate(self, split_name=None, split=None, return_tuple=False):
        """
        Iterate over the dataset or a subset of it.

        Parameters
        ----------
        split_name : string, optional
            If given and split is given iterate over the specified data subset
            (if it exists). If None, iterate over the whole dataset.
        split : string, optional
            One of {train, valid, test} If given and split_name is given
            iterate over the specified data subset (if it exists). If None,
            iterate over the whole dataset.
        return_tuple : bool, optional (default is False)
            If True return the data elements as tuples instead of dicts as
            __getitem__does
        """
        if split_name is not None and split is not None:
            index_list = self.get_split(split_name, split)
        else:
            index_list = range(len(self))
        for i in index_list:
            if return_tuple:
                sample = self[i]
                yield tuple(sample[col] for col in self._selected_cols)
            else:
                yield self[i]

    def get_split(self, split_name, split):
        """
        Get indices of elements belonging to a given dataset split.

        Parameters
        ----------
        split_name : string
            Name identifying the dataset split to be returned.
        split_name : string
            One of {train, valid, test}. The datasubset of the given split to
            be returned.
        """
        if split_name not in self._splits:
            raise KeyError("This dataset has no split '" + split_name + "'!")
        if split not in self._splits[split_name]:
            raise KeyError("The split '" + split_name +
                           "' doesn't have a subset " + split)
        return self._splits[split_name][split]

    def _load_all(self):
        """
        Helper for easy non-lazy loading of datasets which do offer lazy
        loading.
        """
        select_cols = self._selected_cols
        self._selected_cols = []
        data = {}
        for col in self._data_cols:
            if col not in self._data.keys():
                self._selected_cols.append(col)
                data[col] = []
        if len(self._selected_cols) > 0:
            for i in range(len(self)):
                sample = self[i]
                for col in self._selected_cols:
                    data[col].append(sample[col])
        for key, val in data.items():
            self._data[key] = val
        self._selected_cols = select_cols
