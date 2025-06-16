from typing import List, Optional, Union

from afnio._variable import Variable
from afnio.autodiff.basic_ops import Split as SplitOp

from .module import Module


class Split(Module):
    """
    Splits a single input Variable into multiple output Variables.

    This module utilizes the `Split` operation from `afnio.autodiff.basic_ops`.
    It supports string data types, splitting the string data of the input Variable
    based on a specified delimiter and an optional maximum number of splits.

    Note:
        This module does not have any trainable parameters.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> class Splitter(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.split = cog.Split()
        >>>     def forward(self, x):
        ...         return self.split(x, " ", 1)
        >>> input = hf.Variable(data="Afnio is great!", role="sentence")
        >>> splitter = Splitter()
        >>> result = splitter(input)
        >>> print([r.data for r in result])
        ['Afnio', 'is great!']
        >>> print([r.role for r in result])
        ['split part 0 of sentence', 'split part 1 of sentence']

    Raises:
        TypeError: If the input is not an instance of `Variable`.
        TypeError: If the input Variable's data is not a string.

    See Also:
        :class:`afnio.autodiff.basic_ops.Split` for the underlying operation.
    """

    sep: Optional[Union[str, Variable]]
    maxsplit: Optional[Union[int, Variable]]

    def __init__(self):
        super().__init__()

        self.register_buffer("sep", None)
        self.register_buffer("maxsplit", None)

    def forward(
        self,
        x: Variable,
        sep: Optional[Union[str, Variable]] = None,
        maxsplit: Optional[Union[int, Variable]] = -1,
    ) -> List[Variable]:
        self.sep = (
            None
            if sep is None
            else (sep if isinstance(sep, Variable) else Variable(sep))
        )
        self.maxsplit = (
            None
            if maxsplit is None
            else (maxsplit if isinstance(maxsplit, Variable) else Variable(maxsplit))
        )
        return SplitOp.apply(x, self.sep, self.maxsplit)
