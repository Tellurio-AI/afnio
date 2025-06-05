from typing import Any, Callable, List, Optional, Tuple, Union

from afnio._variable import Variable
from afnio.autodiff.evaluator import (
    DeterministicEvaluator as DeterministicEvaluatorOp,
)

from .module import Module


class DeterministicEvaluator(Module):
    """
    Evaluates predictions deterministically using a user-defined evaluation function.

    This module utilizes the `DeterministicEvaluator` operation from
    `afnio.autodiff.evaluator` to compute evaluation scores and explanations.
    The `forward` method takes in a `prediction`, a `target`, an evaluation
    function (`eval_fn`), and its purpose description (`eval_fn_purpose`). It also
    accepts a reduction function (`reduction_fn`) and its purpose description
    (`reduction_fn_purpose`) to aggregate scores if needed. The method outputs
    an evaluation `score` and an `explanation`, both as `Variable` instances. The
    `success_fn` checks if all evaluations are successful, allowing the `backward` pass
    to skip unnecessary gradient computations. The method outputs an evaluation
    `score` and an `explanation`, both as `Variable` instances.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> from afnio import set_backward_model_client
        >>> set_backward_model_client("openai/gpt-4o")
        >>> class ExactColor(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         def exact_match_fn(pred: str, tgt: str) -> int:
        ...             return 1 if pred == tgt_data else 0
        ...         self.exact_match_fn = exact_match_fn
        ...         self.fn_purpose = "exact match"
        ...         self.reduction_fn = sum
        ...         self.reduction_fn_purpose = "summation"
        ...         self.exact_match = cog.DeterministicEvaluator()
        ...     def forward(self, prediction, target):
        ...         return self.exact_match(
        ...             prediction,
        ...             target,
        ...             self.exact_match_fn,
        ...             self.fn_purpose,
        ...             self.reduction_fn,
        ...             self.reduction_fn_purpose,
        ...         )
        >>> prediction = hf.Variable(
        ...     data=["the color is green", "blue"],
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = ["green", "blue"]
        >>> model = ExactColor()
        >>> score, explanation = model(prediction, target)
        >>> print(score.data)
        1
        >>> print(explanation.data)
        'The evaluation function, designed for 'exact match', compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch, generating individual scores for each pair. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'

    Raises:
        TypeError: If inputs are not of the correct types.

    See Also:
        :class:`afnio.autodiff.evaluator.DeterministicEvaluator` for the underlying
        operation.
    """  # noqa: E501

    eval_fn: Callable[[Variable, Union[str, Variable]], List[Any]]
    eval_fn_purpose: Union[str, Variable]
    success_fn: Optional[Callable[[List[Any]], bool]]
    reduction_fn: Optional[Callable[[List[Any]], Any]]
    reduction_fn_purpose: Optional[Union[str, Variable]]

    def __init__(self):
        super().__init__()

        self.register_function("eval_fn", None)
        self.register_buffer("eval_fn_purpose", None)
        self.register_function("success_fn", None)
        self.register_function("reduction_fn", None)
        self.register_buffer("reduction_fn_purpose", None)

    def forward(
        self,
        prediction: Variable,
        target: Union[str, List[str], Variable],
        eval_fn: Callable[[Variable, Union[str, Variable]], List[Any]],
        eval_fn_purpose: Union[str, Variable],
        success_fn: Optional[Callable[[List[Any]], bool]],
        reduction_fn: Optional[Callable[[List[Any]], Any]],
        reduction_fn_purpose: Optional[Union[str, Variable]],
    ) -> Tuple[Variable, Variable]:
        self.eval_fn = eval_fn
        self.eval_fn_purpose = (
            None
            if eval_fn_purpose is None
            else (
                eval_fn_purpose
                if isinstance(eval_fn_purpose, Variable)
                else Variable(eval_fn_purpose)
            )
        )
        self.success_fn = success_fn
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
        return DeterministicEvaluatorOp.apply(
            prediction,
            target,
            self.eval_fn,
            self.eval_fn_purpose,
            self.success_fn,
            self.reduction_fn,
            self.reduction_fn_purpose,
        )
