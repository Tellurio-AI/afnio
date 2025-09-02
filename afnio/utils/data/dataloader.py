from typing import Any, Generic, Iterable, Optional, TypeVar, Union

from afnio._variable import Variable
from afnio.tellurio._variable_registry import suppress_variable_notifications
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

    def __next__(self) -> Any:
        """
        Returns the next batch from the dataset, collated according to the structure
        of the dataset's __getitem__ output.
        Batching logic:
        - If the dataset returns a dictionary, this method aggregates each key across
          the batch into a list of values. For example, if each sample is
          {'a': 'foo', 'b': 'bar'}, the batch will be {'a': [...], 'b': [...]}.
        - If the dataset returns a tuple (e.g., (X, y)), this method recursively
          collates each position in the tuple using `collate_tuple`, preserving
          nested tuple structure and batching Variables as described below.
        - If the dataset returns Variables directly, this method batches them into
          a single Variable whose `.data` is a list of the original `.data` fields,
          and whose `role` and `requires_grad` are taken from the first Variable.
        - Otherwise, returns the batch as a list.
        """
        # Suppress notifications for individual Variables
        with suppress_variable_notifications():
            batch = []
            for _ in range(self.batch_size):
                try:
                    index = self._next_index()
                    batch.append(self.dataset[index])
                except StopIteration:
                    if not batch or self.drop_last:
                        raise
                    break

        # If dataset returns a dictionary, we aggregate each key across the batch
        if (
            batch
            and isinstance(batch[0], dict)  # noqa: W503
            and all(isinstance(item, dict) for item in batch)  # noqa: W503
        ):
            keys = batch[0].keys()
            collated = {}
            for key in keys:
                values = [item[key] for item in batch]
                collated[key] = values
            return collated
        # If dataset returns a tuple, we recursively collate each position in the tuple
        if (
            batch
            and isinstance(batch[0], tuple)  # noqa: W503
            and all(isinstance(item, tuple) for item in batch)  # noqa: W503
        ):
            return collate_tuple(batch)

        # If dataset returns Variables, we batch them into a single Variable
        if (
            batch
            and isinstance(batch[0], Variable)  # noqa: W503
            and all(isinstance(item, Variable) for item in batch)  # noqa: W503
        ):
            first = batch[0]
            return Variable(
                data=[item.data for item in batch],
                role=first.role,
                requires_grad=first.requires_grad,
            )

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


def collate_tuple(items):
    """
    Recursively collates a batch of tuples, preserving nested structure.

    This function should only be called when processing batches where each element
    is a tuple (i.e., when the dataset's __getitem__ returns tuples).

    The function first transposes the batch, so that each position in the tuple is
    grouped together. For each group:
      - If all elements are Variables, returns a single Variable whose `.data` is a list
        of the original `.data` fields, and whose `role` and `requires_grad` are taken
        from the first Variable.
      - If all elements are tuples, recursively collates them to preserve nested
        structure.
      - If some elements are tuples and some are not, recursively collates the tuples
        and leaves other elements as is, preserving their position.
      - Otherwise, returns a list of the grouped items.

    This enables flexible batching for datasets that return tuples of Variables,
    nested tuples, or mixed structures.
    """
    transposed = list(zip(*items))
    collated = []
    for group in transposed:
        # If all are Variables, batch as Variable
        if all(isinstance(x, Variable) for x in group):
            first = group[0]
            collated.append(
                Variable(
                    data=[x.data for x in group],
                    role=first.role,
                    requires_grad=first.requires_grad,
                )
            )
        # If all are tuples, recurse
        elif all(isinstance(x, tuple) for x in group):
            collated.append(collate_tuple(group))
        # If some are tuples and some are not, handle each element
        else:
            collated.append(
                [collate_tuple([x]) if isinstance(x, tuple) else x for x in group]
            )
    return tuple(collated)
