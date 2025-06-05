from afnio._variable import Variable
from afnio.autodiff.basic_ops import Add as AddOp

from .module import Module


class Add(Module):
    """
    Performs element-wise addition of two input Variables.

    This module utilizes the `Add` operation from `afnio.autodiff.basic_ops`.
    The inputs must be instances of the `Variable` class. The `forward` method
    applies the addition operation to the `.data` field of the inputs and returns
    the resulting `Variable`.

    Note:
        This module does not have any trainable parameters.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> class Addition(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.add = cog.Add()
        >>>     def forward(self, x, y):
        ...         return self.add(x, y)
        >>> input1 = hf.Variable(data="abc", role="input1")
        >>> input2 = hf.Variable(data="def", role="input2")
        >>> addition = Addition()
        >>> result = addition(input1, input2)
        >>> print(result)
        'abcdef'
        >>> print(result.role)
        'input1 and input2'

    Raises:
        TypeError: If either input is not an instance of `Variable`.
        TypeError: If addition between the input types is not allowed.
        ValueError: If a scalar `.data` is added to a list `.data`.
        ValueError: If list `.data` fields have mismatched lengths.

    See Also:
        :class:`afnio.autodiff.basic_ops.Add` for the underlying operation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Variable, y: Variable) -> Variable:
        if not isinstance(x, Variable) or not isinstance(y, Variable):
            raise TypeError("Both inputs must be of type 'Variable'.")
        return AddOp.apply(x, y)
