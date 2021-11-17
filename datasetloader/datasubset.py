class DataSubset:
    """
    Provides a Sequence for a given subset of a DatasetLoader object.

    Sequence object provide __len__ and __getitem__ and so can be directly
    passed into a PyTorch DatasetLoader.
    """
    def __init__(self, dataset_loader, split_name, subset):

        self._dataset_loader = dataset_loader
        self._samples = self._dataset_loader.get_split(split_name, subset)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        sample = self._dataset_loader[self._samples[index]]
        return tuple(sample[col]
                     for col in self._dataset_loader._selected_cols)
