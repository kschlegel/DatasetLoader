class DatasetLoader():
    """
    Base class for all dataset loaders to provide a common interface for
    retrieving the data out of the dataset object.
    """
    def __init__(self):
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

    def get_traintest_split(self, split):
        """
        Returns a tuple with the lists of the training and testset ids.

        Parameters
        ----------
        split : string, optional (default is 'default')
            if the dataset definition includes splits into training and test
            set query the train and testset ids of this split. Valid split 
            names depend on the dataset.
        """
        train_ids = self._get_index_set("train", split)
        test_ids = self._get_index_set("test", split)
        return (train_ids, test_ids)
    
    def get_data(self, data_keys, subset="all", split="default"):
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
        split : string, optional (default is 'default')
            if the dataset definition includes splits into training, test and
            possibly validation set, the split can be chosen using this
            parameter. Valid split names depend on the dataset.
        """
        data_keys = self._check_keys(data_keys)
        if subset == "all":
            if len(data_keys) == 1:
                return self._data[data_keys[0]]
            else:
                return [self._data[key] for key in data_keys]
        else:
            index_set = self._get_index_set(subset, split)

        if len(data_keys) == 1:
            return self._data[data_keys[0]][index_set]
        else:
            return [self._data[key][index_set] for key in data_keys]

    def get_iterator(self, data_keys, subset="all", split="default"):
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
        split : string, optional (default is 'default')
            if the dataset definition includes splits into training, test and
            possibly validation set, the split can be chosen using this
            parameter. Valid split names depend on the dataset.
        """
        data_keys = self._check_keys(data_keys)

        if subset == "all":
            index_set = [i for i in range(self._length)]
        else:
            index_set = self._get_index_set(subset, split)

        for i in index_set:
            if len(data_keys) == 1:
                yield self._data[data_keys[0]][i]
            else:
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

    def _get_index_set(self, subset, split):
        if self._splits is not None:
            if split == "default":
                split = self._default_split
            if split not in self._splits:
                raise Exception(split +
                                " is not a valid split name for this dataset.")

            if subset in self._splits[split]:
                return self._splits[split][subset]
            else:
                raise Exception(" This dataset has no " + subset + " subset.")
        else:
            raise Exception("This dataset doesn't have any splits.")
