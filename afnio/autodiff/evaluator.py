from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from afnio._utils import (
    MultiTurnMessages,
)
from afnio._variable import Variable
from afnio.autodiff.decorators import evaluator
from afnio.autodiff.function import Function
from afnio.models import ChatCompletionModel


@evaluator
class DeterministicEvaluator(Function):
    """
    Evaluates predictions deterministically using a user-defined evaluation function
    within the `afnio` framework, supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    The `DeterministicEvaluator` function computes a `score` and an `explanation` based
    on the `prediction` and `target` inputs using a user-defined evaluation function
    (`eval_fn`). The evaluation function's purpose is described by `eval_fn_purpose`.
    Outputs include a numerical or textual score and a textual explanation, both wrapped
    as  `Variable` objects.

    The `prediction` is a `Variable`. The `target` can be a string, a list of strings,
    or a `Variable`. Each `Variable` passed as an input argument can have either
    a scalar or a list `.data` field, supporting both individual samples and batch
    processing. For batch processing, the lengths of `prediction` and `target` must
    match.

    The `success_fn` parameter is a user-defined function that returns `True` when
    all predictions evaluated by `eval_fn` are considered successful, and `False`
    otherwise. If `success_fn` returns `True`, the `backward` pass will skip gradient
    calculations and directly return an empty gradient, optimizing computational time.

    The `reduction_fn` parameter specifies the aggregation function to use for scores
    across a batch of predictions and targets. When specified, the reduction function's
    purpose is described using `reduction_fn_purpose`. If aggregation is not desired,
    set `reduction_fn` and `reduction_fn_purpose` to `None`.

    Example with scalar inputs:
        >>> prediction = Variable(
        ...     data="green",
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = "red"
        >>> def exact_match_fn(p: str, t: str) -> int:
        ...     return 1 if p == t else 0
        >>> score, explanation = DeterministicEvaluator.apply(
        ...     prediction,
        ...     target,
        ...     exact_match_fn,
        ...     "exact match",
        ... )
        >>> score.data
        0
        >>> explanation.data
        'The evaluation function, designed for 'exact match', compared the <DATA> field of the predicted variable ('green') with the <DATA> field of the target variable ('red'), resulting in a score: 0.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'

    Example with batched inputs:
        >>> prediction = Variable(
        ...     data=["green", "blue"],
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = ["red", "blue"]
        >>> def exact_match_fn(p: str, t: str) -> int:
        ...     return 1 if p == t else 0
        >>> score, explanation = DeterministicEvaluator.apply(
        ...     prediction,
        ...     target,
        ...     exact_match_fn,
        ...     "exact match",
        ...     reduction_fn=sum,
        ...     reduction_fn_purpose="summation"
        ... )
        >>> score.data
        1
        >>> explanation.data
        'The evaluation function, designed for 'exact match', compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch, generating individual scores for each pair. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        prediction: Variable,
        target: Union[str, List[str], Variable],
        eval_fn: Callable[[Variable, Union[str, Variable]], List[Any]],
        eval_fn_purpose: Union[str, Variable],
        success_fn: Optional[Callable[[List[Any]], bool]],
        reduction_fn: Optional[Callable[[List[Any]], Any]],
        reduction_fn_purpose: Optional[Union[str, Variable]],
    ) -> Tuple[Variable, Variable]:
        raise NotImplementedError(
            "DeterministicEvaluator.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(
        ctx, score_grad_output: Variable, explanation_grad_output: Variable
    ) -> Tuple[Variable, None, None, None]:
        raise NotImplementedError(
            "DeterministicEvaluator.backward is implemented on the server. "
            "Client-side execution is not supported."
        )


@evaluator
class ExactMatchEvaluator(Function):
    """
    Evaluates predictions using exact matching within the `afnio` framework,
    supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    The `ExactMatchEvaluator` function computes a `score` and an `explanation` by
    comparing the `data` fields of a `prediction` and a `target` for an exact match.
    For each sample:
    - A score of `1` is assigned for an exact match.
    - A score of `0` is assigned otherwise.

    The `prediction` is a `Variable`. The `target` can be a string, a list of strings,
    or a `Variable`. Each `Variable` passed as an input argument can have either
    a scalar or a list `.data` field, supporting both individual samples and batch
    processing. For batch processing, the lengths of `prediction` and `target` must
    match.

    If batched inputs are provided, the scores can be aggregated using an optional
    `reduction_fn`, such as `sum`. The purpose of the reduction is described using
    `reduction_fn_purpose`. If aggregation is not desired, set `reduction_fn` and
    `reduction_fn_purpose` to `None`.

    Example with scalar inputs:
        >>> prediction = Variable(
        ...     data="green",,
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = "red",
        >>> score, explanation = ExactMatchEvaluator.apply(prediction, target)
        >>> score.data
        0
        >>> explanation.data
        'The evaluation function, designed for 'exact match', compared the <DATA> field of the predicted variable ('green') with the <DATA> field of the target variable ('red'), resulting in a score: 0.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'

    Example with batched inputs:
        >>> prediction = Variable(
        ...     data=["green", "blue"],
        ...     role="color prediction",
        ...     requires_grad=True
        ... )
        >>> target = ["red", "blue"]
        >>> score, explanation = ExactMatchEvaluator.apply(prediction, target)
        >>> score.data
        1
        >>> explanation.data
        'The evaluation function, designed for 'exact match', compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch, generating individual scores for each pair. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        prediction: Variable,
        target: Union[str, List[str], Variable],
        reduction_fn: Optional[Callable[[List[Any]], Any]] = sum,
        reduction_fn_purpose: Optional[Union[str, Variable]] = "summation",
    ) -> Tuple[Variable, Variable]:
        raise NotImplementedError(
            "ExactMatchEvaluator.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(
        ctx, score_grad_output: Variable, explanation_grad_output: Variable
    ) -> Tuple[Variable, None]:
        raise NotImplementedError(
            "ExactMatchEvaluator.backward is implemented on the server. "
            "Client-side execution is not supported."
        )


@evaluator
class LMJudgeEvaluator(Function):
    r"""
    Implements an evaluation of a model prediction using a language model (LM) as the
    judge within the `afnio` framework, supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    This function returns a `score` and an `explanation`, both as `Variable` objects,
    by comparing a `prediction` against a `target` (when present) using a composite
    prompt. The prompt is constructed from a list of `messages` and optional `inputs`,
    which can dynamically populate placeholders in the message templates. The evaluation
    process leverages the specified `forward_model_client` to perform the
    LM-based assessment.

    The `prediction` is a `Variable`. The `target` can be a string, a list of strings,
    or a `Variable`. Similarly, the `inputs` dictionary can include strings, lists of
    strings, or `Variable`s. Each `Variable` passed as an input argument can have either
    a scalar or a list `.data` field, supporting both individual samples and batch
    processing. For batch processing, the lengths of `prediction`, `target`, and any
    batched `inputs` must match.

    The `success_fn` parameter is a user-defined function that returns `True` when
    all predictions evaluated by the LM as Judge are considered successful, and `False`
    otherwise. If `success_fn` returns `True`, the `backward` pass will skip gradient
    calculations and directly return an empty gradient, optimizing computational time.

    If you are processing a batch of predictions and targets, you can use the
    `reduction_fn` to aggregate individual scores (e.g., using `sum` to compute a total
    score). The `reduction_fn_purpose` parameter is a brief description of the
    aggregation’s purpose (e.g., `"summation"`). If you don’t want any aggregation, set
    both `reduction_fn` and `reduction_fn_purpose` to `None`.

    The function operates in two modes controlled by `eval_mode`:
    - **`eval_mode=True` (default)** – Computes gradients for `prediction` only. Use it
      for direct feedback on predictions.
    - **`eval_mode=False`** – Computes gradients for `messages` and `inputs`. Use it to
      optimize the evaluator or align with human evaluation datasets.

    Additional model parameters, such as temperature, max tokens, or seed values, can
    be passed through `completion_args` to customize the LLM's behavior.

    Example with scalar inputs:
        >>> task = Variable(
        ...     "Evaluate if the translation is accurate.",
        ...     role="evaluation task",
        ...     requires_grad=True
        ... )
        >>> format = Variable(
        ...     "Provide 'score' (true/false) and 'explanation' in JSON.",
        ...     role="output format"
        ... )
        >>> user = Variable(
        ...     "<PREDICTION>{prediction}</PREDICTION><TARGET>{target}</TARGET>",
        ...     role="user query"
        ... )
        >>> prediction = Variable(
        ...     "Hola Mundo",
        ...     role="translated text",
        ...     requires_grad=True
        ... )
        >>> target = Variable("Ciao Mondo", role="expected output")
        >>> messages = [
        ...     {"role": "system", "content": [task, format]},
        ...     {"role": "user", "content": [user]}
        ... ]
        >>> score, explanation = LMJudgeEvaluator.apply(
        ...     model,
        ...     messages,
        ...     prediction,
        ...     target,
        ...     temperature=0.5,
        ... )
        >>> score.data
        False
        >>> explanation.data
        'The translated text is in Spanish, but the expected is in Italian.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'The translated text should be in Italian.'

    Example with batched inputs:
        >>> task = Variable(
        ...     "Evaluate if the translation is accurate.",
        ...     role="evaluation task",
        ...     requires_grad=True
        ... )
        >>> format = Variable(
        ...     "Provide 'score' (true/false) and 'explanation' in JSON.",
        ...     role="output format"
        ... )
        >>> user = Variable(
        ...     "<PREDICTION>{prediction}</PREDICTION><TARGET>{target}</TARGET>",
        ...     role="user query"
        ... )
        >>> prediction = Variable(
        ...     data=["Hola Mundo", "Salve a tutti"],
        ...     role="translated text",
        ...     requires_grad=True,
        ... )
        >>> target = ["Ciao Mondo", "Salve a tutti"]
        >>> score, explanation = LMJudgeEvaluator.apply(
        ...     model,
        ...     messages,
        ...     prediction,
        ...     target,
        ...     reduction_fn=sum,
        ...     reduction_fn_purpose="summation",
        ... )
        >>> score.data
        1
        >>> explanation.data
        'The evaluation function, designed using an LM as the judge, compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
    """  # noqa: E501

    @staticmethod
    def forward(
        ctx,
        forward_model_client: Optional[ChatCompletionModel],
        messages: MultiTurnMessages,
        prediction: Variable,
        target: Optional[Union[str, List[str], Variable]] = None,
        inputs: Optional[Dict[str, Union[str, Variable]]] = None,
        success_fn: Optional[Callable[[List[Any]], bool]] = None,
        reduction_fn: Optional[Callable[[List[Any]], Any]] = sum,
        reduction_fn_purpose: Optional[Union[str, Variable]] = "summation",
        eval_mode: Union[bool, Variable] = True,
        **completion_args,
    ) -> Tuple[Variable, Variable]:
        raise NotImplementedError(
            "LMJudgeEvaluator.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(
        ctx, score_grad_output: Variable, explanation_grad_output: Variable
    ) -> Tuple[Optional[Variable], ...]:
        raise NotImplementedError(
            "LMJudgeEvaluator.backward is implemented on the server. "
            "Client-side execution is not supported."
        )
