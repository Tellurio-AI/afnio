import random
from typing import Generic, Iterator, Optional, Sequence, Sized, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned
    iterators.
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement,
    then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``,
            default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        seed (int): A number to set the seed for the random draws.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, "
                f"but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, "
                f"but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def _is_valid_random_state(self, state) -> bool:
        return isinstance(state, tuple) and len(state) > 0

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        random.seed(self.seed)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from random.choices(range(n), k=32)
            yield from random.choices(range(n), k=self.num_samples % 32)
        else:
            for _ in range(self.num_samples // n):
                yield from random.sample(range(n), n)
            yield from random.sample(range(n), self.num_samples % n)

    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence): a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        seed (int): A number to set the seed for the random draws.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """  # noqa: E501

    weights: Sequence[float]
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, "
                f"but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, "
                f"but got replacement={replacement}"
            )

        if len(weights) == 0 or not all(isinstance(w, (float, int)) for w in weights):
            raise ValueError("Weights must be a non-empty sequence of numbers.")

        if not replacement and num_samples > len(weights):
            raise ValueError(
                f"num_samples ({num_samples}) cannot be greater than "
                f"the population size ({len(weights)}) when replacement is False."
            )

        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        random.seed(self.seed)

        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]

        if self.replacement:
            yield from random.choices(
                population=range(len(self.weights)),
                weights=probabilities,
                k=self.num_samples,
            )
        else:
            # Sample without replacement
            yield from random.sample(range(len(self.weights)), k=self.num_samples)

    def __len__(self) -> int:
        return self.num_samples
