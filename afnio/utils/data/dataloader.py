from typing import Any, Generic, Iterable, Optional, TypeVar, Union

from afnio.utils.data.dataset import Dataset
from afnio.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

T_co = TypeVar("T_co", covariant=True)


class DataLoader(Generic[T_co]):
    r"""
    Data loader combines a dataset and a sampler, and provides an iterable over the
    given dataset.

    The :class:`~afnio.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single-process loading, customizing loading order
    and optional automatic batching (collation) and memory pinning.

    See :py:mod:`afnio.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        seed (int, optional): If not ``None``, this seed will be used by RandomSampler
            to generate random indexes. (default: ``None``)
    """

    dataset: Dataset[T_co]
    batch_size: Optional[int]
    drop_last: bool
    sampler: Union[Sampler, Iterable]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        sampler: Union[Sampler, Iterable, None] = None,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle not in {True, False}:
            raise ValueError(
                f"DataLoader with IterableDataset: "
                f"expected unspecified shuffle option, but got shuffle={shuffle}"
            )

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset, seed=seed)
            else:
                sampler = SequentialSampler(dataset)

        self.index_sampler = sampler
        self._sampler_iter = iter(self.index_sampler)
        self.__initialized = True

    def __iter__(self) -> Iterable[Any]:
        self._sampler_iter = iter(self.index_sampler)  # Ensure new iterator every time
        return self

    def _next_index(self):
        return next(self._sampler_iter)

    # TODO: Batches must return Variables with list `.data` field,
    #       rather than tuples of Variables
    def __next__(self) -> Any:
        batch = []
        for _ in range(self.batch_size):
            try:
                index = self._next_index()
                batch.append(self.dataset[index])
            except StopIteration:
                if not batch or self.drop_last:
                    raise
                break
        return batch

    def __len__(self) -> int:
        length = len(self.dataset)
        if self.batch_size is not None:
            from math import ceil

            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length
