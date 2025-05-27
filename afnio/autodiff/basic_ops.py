from typing import List, Optional, Tuple, Union

from afnio._variable import Variable
from afnio.autodiff.function import Function


class Add(Function):
    r"""Implements an addition operation for `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

    This class inherits from `autodiff.Function` and requires both the `forward`
    and `backward` methods to be defined.

    The `Add` function supports both scalar and list `.data` fields:
    - **Scalars**: Adds numerical values (`int`, `float`) or concatenates strings.
    - **Lists**: Performs element-wise addition of corresponding elements from the lists.
      Lists must be of the same length.

    It automatically handles type-based operations:
    - For numerical data (`int`, `float`), it performs arithmetic addition.
    - For strings, it concatenates the values.
    - Mixed types (e.g., string and number) are converted appropriately before performing
      the addition.

    This operation also tracks `Variable` dependencies, enabling automatic gradient
    computation through backpropagation.

    Example with scalar inputs:
        >>> x = Variable(data="abc", role="first input", requires_grad=True)
        >>> y = Variable(data="def", role="second input", requires_grad=False)
        >>> result = Add.apply(x, y)
        >>> result.data
        'abcdef'
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
        >>> g = Variable(data="MY_FEEDBACK", role="add gradient")
        >>> result.backward(g)
        >>> x.grad[0].data
        'Here is the combined feedback we got for this specific first input and other variables: MY_FEEDBACK'
        >>> x.grad[0].role
        'feedback to first input'

    Example with batched inputs:
        >>> x = Variable(data=[1, 2, 3], role="first input", requires_grad=True)
        >>> y = Variable(data=[4, 5, 6], role="second input", requires_grad=False)
        >>> result = Add.apply(x, y)
        >>> result.data
        [5, 7, 9]
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
    """  # noqa: E501

    @staticmethod
    def forward(ctx, x: Variable, y: Variable) -> Variable:
        raise NotImplementedError(
            "Add.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(
        ctx, grad_output: Variable
    ) -> Tuple[Optional[Variable], Optional[Variable]]:
        raise NotImplementedError(
            "Add.backward is implemented on the server. "
            "Client-side execution is not supported."
        )


# TODO: enable summarization of `.data`` fields in `forward()` method using a LM
#       (ideally a small and fast SLM)
class Sum(Function):
    r"""
    Implements a summation operation for a list of `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    The `Sum` function aggregates the `.data`, `.role`, and `.requires_grad` attributes
    of all input `Variable` instances into a single `Variable`. It supports both
    scalar and list `.data` fields:
    - **Scalars**: Computes the arithmetic sum for numerical data (`int`, `float`)
      or concatenates all string values, wrapping each in `<ITEM></ITEM>` tags.
    - **Lists**: Aggregates the corresponding elements of the lists. For numerical
      data, it sums the corresponding elements. For string data, it concatenates them,
      wrapping each element in `<ITEM></ITEM>` tags.

    During backpropagation, the function distributes the gradient to all input
    `Variable` instances that require gradients.

    Example with scalar inputs:
        >>> x = Variable(data="abc", role="first input", requires_grad=True)
        >>> y = Variable(data="def", role="second input", requires_grad=False)
        >>> result = Sum.apply([x, y])
        >>> result.data
        '<ITEM>abc</ITEM><ITEM>def</ITEM>'
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
        >>> g = Variable(data="MY_FEEDBACK", role="add gradient")
        >>> result.backward(g)
        >>> x.grad[0].data
        'Here is the combined feedback we got for this specific first input and other variables: MY_FEEDBACK'
        >>> x.grad[0].role
        'feedback to first input'

    Example with batched inputs:
        >>> x = Variable(data=[1, 2, 3.5], role="first input", requires_grad=True)
        >>> y = Variable(data=[4, 5, 6], role="second input", requires_grad=False)
        >>> result = Sum.apply([x, y])
        >>> result.data
        [5, 7, 9.5]
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
    """  # noqa: E501

    @staticmethod
    def forward(ctx, x: List[Variable]) -> Variable:
        raise NotImplementedError(
            "Sum.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(ctx, grad_output: Variable) -> Tuple[Optional[Variable], ...]:
        raise NotImplementedError(
            "Sum.backward is implemented on the server. "
            "Client-side execution is not supported."
        )


class Split(Function):
    r"""
    Implements a split operation for `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    The `Split` function divides the `.data` of the input `Variable` into multiple parts
    using a specified delimiter `sep`. If `maxsplit` is specified, the split operation
    is limited to a maximum number of splits. It handles both scalar and list `.data`
    fields:
    - **Scalars**: The scalar `.data` (a single string) is split into substrings
      based on the specified `sep` and `maxsplit` parameters.
    - **Lists**: Each element of the list `.data` (strings) is split individually. If
      splits of varying lengths occur, shorter splits are automatically padded with
      empty strings to ensure consistent dimensions.

    During backpropagation, feedback is collected and aggregated across all split parts.
    The combined feedback is propagated back to the original input `Variable`, allowing
    for the proper computation of gradients.

    Example with scalar inputs:
        >>> x = Variable(data="afnio is great!", role="sentence", requires_grad=True)
        >>> result = Split.apply(x, sep=" ", maxsplit=1)
        >>> [var.data for var in result]
        ['afnio', 'is great!']
        >>> result[0].role
        'split part 0 of sentence'
        >>> g_1 = Variable(data="MY_FIRST_FEEDBACK", role="gradient")
        >>> g_2 = Variable(data="MY_SECOND_FEEDBACK", role="gradient")
        >>> result[0].backward(g_1, retain_graph=True)
        >>> result[1].backward(g_2)
        >>> x.grad[0].data
        'Here is the combined feedback we got for this specific sentence and other variables: <ITEM>MY_FIRST_FEEDBACK</ITEM><ITEM></ITEM>'
        >>> x.grad[0].role
        'feedback to sentence'
        >>> x.grad[1].data
        'Here is the combined feedback we got for this specific sentence and other variables: <ITEM></ITEM><ITEM>MY_SECOND_FEEDBACK</ITEM>'
        >>> x.grad[1].role
        'feedback to sentence'

    Example with batched inputs:
        >>> x = Variable(
        ...     data=["afnio is great!", "Deep learning"],
        ...     role="sentences",
        ...     requires_grad=True
        ... )
        >>> result = Split.apply(x, sep=" ", maxsplit=2)
        >>> [var.data for var in result]
        [['afnio', 'Deep'], ['is', 'learning'], ['great!', '']]
        >>> g = Variable(data="MY_FEEDBACK", role="gradient")
        >>> result[1].backward(g)
        >>> x.grad[0].data
        'Here is the combined feedback we got for this specific sentences and other variables: <ITEM></ITEM><ITEM>MY_FEEDBACK</ITEM><ITEM></ITEM>'
        >>> x.grad[0].role
        'feedback to sentences'
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        x: Variable,
        sep: Optional[Union[str, Variable]] = None,
        maxsplit: Optional[Union[int, Variable]] = -1,
    ) -> Tuple[Variable]:
        raise NotImplementedError(
            "Split.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    # TODO: enable summarization of elements in `grad.data`` fields in `backward()`
    #       method using a LM (ideally a small and fast SLM)
    @staticmethod
    def backward(ctx, *grad_outputs: Variable) -> Tuple[Optional[Variable], None, None]:
        raise NotImplementedError(
            "Split.backward is implemented on the server. "
            "Client-side execution is not supported."
        )
