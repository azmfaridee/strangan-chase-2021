from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    """
    Infinite data loader so that mismatched (by number of samples) dataset from source and target are not a problem anymore
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

        # keep track how many epochs have passed
        # index starts at 1
        self.epoch = 1

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            self.epoch += 1
            batch = next(self.dataset_iterator)
        return batch
