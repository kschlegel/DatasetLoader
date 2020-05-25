class DatasetLoader():
    """
    Base class for all dataset loaders to provide a common interface for
    retrieving the data out of the dataset object.
    """
    def __init__(self):
        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._trainingset = None
        self._validationset = None
        self._testset = None
        # Different datasets will have different elements so safer to keep
        # track of length rather than checking the length of any given list
        self._length = 0

    def has_attribute(self, data_key):
        """
        Check whether the dataset has the given type of data.

        Pass in a string key to check its validity for use as key in the
        get_data and get_iterator methods.

        Parameters
        ----------
        data_key : string
            key to be checked
        """
        return (data_key in self._data)

    def get_data(self, data_keys, subset="all"):
        """
        Returns a list of the data.

        Returns a list of the data of either the full dataset or the requested
        subset) of the requested data keys.

        Parameters
        ----------
        data_keys : string or tuple/list of strings
            either a string or tuple/list of keys into the data_dict to
            be returned.
        subset : string, optional (default is 'all')
            One of {‘train’, ‘valid’, ‘test’, ‘all’}. Some
            datasets may not have a pre-defined split, an exception will be
            raised when trying to get a subset set which doesn't exist.
        """
        data_keys = self._check_keys(data_keys)
        if subset == "all":
            return [self._data[key] for key in data_keys]
        elif subset == "train":
            if self._trainingset is not None:
                index_set = self._trainingset
            else:
                raise Exception("This dataset doesn't have a training set.")
        elif subset == "valid":
            if self._validationset is not None:
                index_set = self._validationset
            else:
                raise Exception("This dataset doesn't have a validation set.")
        elif subset == "test":
            if self._testset is not None:
                index_set = self._testset
            else:
                raise Exception("This dataset doesn't have a test set.")
        else:
            raise Exception(
                "subset must be one of {‘train’, ‘valid’, ‘test’, ‘all’}.")
        return [self._data[key][index_set] for key in data_keys]

    def get_iterator(self, data_keys, subset="all"):
        """
        Returns an iterator over the data.

        Returns an iterator over either the full dataset or the requested
        subset) of the requested data keys.

        Parameters
        ----------
        data_keys : string or tuple/list of strings
            either a string or tuple/list of keys into the data_dict to
            be returned.
        subset : string, optional (default is 'all')
            One of {‘train’, ‘valid’, ‘test’, ‘all’}. Some
            datasets may not have a pre-defined split, an exception will be
            raised when trying to get a subset set which doesn't exist.
        """
        data_keys = self._check_keys(data_keys)

        if subset == "all":
            index_set = [i for i in range(self._length)]
        elif subset == "train":
            index_set = self._trainingset
        elif subset == "valid":
            if self._validationset is not None:
                index_set = self._validationset
            else:
                raise Exception("This dataset doesn't have a validation set.")
        elif subset == "test":
            index_set = self._testset
        else:
            raise Exception(
                "subset must be one of {‘train’, ‘valid’, ‘test’, ‘all’}.")
        for i in index_set:
            yield [self._data[key][i] for key in data_keys]

    def _check_keys(self, data_keys):
        if not isinstance(data_keys, (tuple, list)):
            data_keys = (data_keys, )
        for key in data_keys:
            if key not in self._data:
                raise Exception("The dataset '" + str(self.__class__) +
                                "' does not contain '" + key +
                                "' information!")
        return data_keys
