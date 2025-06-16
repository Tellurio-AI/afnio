from typing import List

from afnio._variable import Variable
from afnio.autodiff.basic_ops import Sum as SumOp

from .module import Module


class Sum(Module):
    """
    Aggregates a list of input Variables into a single output Variable.

    This module utilizes the `Sum` operation from `afnio.autodiff.basic_ops`.
    It supports both numerical (int, float) and string data types. For numerical data,
    it computes the sum. For string data, it concatenates the values and wraps
    each in `<ITEM></ITEM>` tags.

    Note:
        This module does not have any trainable parameters.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> class Summation(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.sum = cog.Sum()
        >>>     def forward(self, x):
        ...         return self.sum(x)
        >>> input1 = hf.Variable(data="abc", role="input1")
        >>> input2 = hf.Variable(data="def", role="input2")
        >>> input3 = hf.Variable(data="ghi", role="input3")
        >>> summation = Summation()
        >>> result = summation([input1, input2, input3])
        >>> print(result.data)
        '<ITEM>abc</ITEM><ITEM>def</ITEM><ITEM>ghi</ITEM>'
        >>> print(result.role)
        'input1 and input2 and input3'

    Raises:
        TypeError: If any input is not an instance of `Variable`.

    See Also:
        :class:`afnio.autodiff.basic_ops.Sum` for the underlying operation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: List[Variable]) -> Variable:
        if not isinstance(x, list) and all(isinstance(y, Variable) for y in x):
            raise TypeError("All inputs must be instances of 'Variable'.")
        return SumOp.apply(x)
