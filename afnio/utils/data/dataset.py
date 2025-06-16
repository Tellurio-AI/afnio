from typing import Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key and :meth:`__len__`, which is expected to return
    the size of the dataset by the default options of
    :class:`~afnio.utils.data.DataLoader`. Subclasses could also optionally
    implement :meth:`__getitems__`, for speedup batched samples loading. This method
    accepts list of indices of samples of batch and returns list of samples.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self):
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher
