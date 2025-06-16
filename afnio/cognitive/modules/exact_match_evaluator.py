from typing import Any, Callable, List, Optional, Tuple, Union

from afnio._variable import Variable
from afnio.autodiff.evaluator import ExactMatchEvaluator as ExactMatchEvaluatorOp

from .module import Module


class ExactMatchEvaluator(Module):
    """
    Evaluates predictions using an exact match criterion.

    This module leverages the `ExactMatchEvaluator` operation from
    `afnio.autodiff.evaluator` and is a specialized version of the
    `DeterministicEvaluator` that uses an exact matching function to compare the
    `prediction` and `target`. It returns an evaluation `score` (1 for exact match,
    0 otherwise) and an `explanation` describing the evaluation result.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> from afnio import set_backward_model_client
        >>> set_backward_model_client("openai/gpt-4o")
        >>> class ExactColor(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.exact_match = cog.ExactMatchEvaluator()
        ...     def forward(self, prediction, target):
        ...         return self.exact_match(prediction, target)
        >>> prediction = hf.Variable(
        ...     data="green",
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = "red"
        >>> model = ExactColor()
        >>> score, explanation = model(prediction, target)
        >>> print(score.data)
        0
        >>> print(explanation.data)
        'The evaluation function, designed for 'exact match', compared the <DATA> fields of the predicted variable and the target variable, resulting in a score: 0.'
        >>> explanation.backward()
        >>> system.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'

    Raises:
        TypeError: If inputs are not of the correct types.

    See Also:
        :class:`afnio.autodiff.evaluator.ExactMatchEvaluator` for the underlying
        operation.
    """  # noqa: E501

    reduction_fn: Optional[Callable[[List[Any]], Any]]
    reduction_fn_purpose: Optional[Union[str, Variable]]

    def __init__(self):
        super().__init__()

        self.register_function("reduction_fn", None)
        self.register_buffer("reduction_fn_purpose", None)

    def forward(
        self,
        prediction: Variable,
        target: Union[str, List[str], Variable],
        reduction_fn: Optional[Callable[[List[Any]], Any]] = sum,
        reduction_fn_purpose: Optional[Union[str, Variable]] = "summation",
    ) -> Tuple[Variable, Variable]:
        self.reduction_fn = reduction_fn
        self.reduction_fn_purpose = (
            None
            if reduction_fn_purpose is None
            else (
                reduction_fn_purpose
                if isinstance(reduction_fn_purpose, Variable)
                else Variable(reduction_fn_purpose)
            )
        )
        return ExactMatchEvaluatorOp.apply(
            prediction, target, self.reduction_fn, self.reduction_fn_purpose
        )
