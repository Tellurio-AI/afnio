from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from afnio._utils import MultiTurnMessages
from afnio._variable import Variable
from afnio.autodiff.basic_ops import Add, Split, Sum
from afnio.autodiff.evaluator import (
    DeterministicEvaluator,
    ExactMatchEvaluator,
    LMJudgeEvaluator,
)
from afnio.autodiff.lm_ops import ChatCompletion
from afnio.models import ChatCompletionModel


def add(x: Variable, y: Variable) -> Variable:
    r"""Implements an addition operation for `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

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
        >>> result = F.add(x, y)
        >>> result.data
        'abcdef'
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
        >>> g = Variable(data="MY_FEEDBACK", role="add gradient")
        >>> result.backward(g)
        >>> x.grad.data
        'Here is the combined feedback we got for this specific first input and other variables: MY_FEEDBACK'
        >>> x.grad.role
        'feedback to first input'

    Example with batched inputs:
        >>> x = Variable(data=[1, 2, 3], role="first input", requires_grad=True)
        >>> y = Variable(data=[4, 5, 6], role="second input", requires_grad=False)
        >>> result = F.add(x, y)
        >>> result.data
        [5, 7, 9]
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
    """  # noqa: E501
    return Add.apply(x, y)


def sum(x: List[Variable]) -> Variable:
    r"""
    Implements a summation operation for a list of `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

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
        >>> result = F.sum([x, y])
        >>> result.data
        '<ITEM>abc</ITEM><ITEM>def</ITEM>'
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
        >>> g = Variable(data="MY_FEEDBACK", role="add gradient")
        >>> result.backward(g)
        >>> x.grad.data
        'Here is the combined feedback we got for this specific first input and other variables: MY_FEEDBACK'
        >>> x.grad.role
        'feedback to first input'

    Example with batched inputs:
        >>> x = Variable(data=[1, 2, 3.5], role="first input", requires_grad=True)
        >>> y = Variable(data=[4, 5, 6], role="second input", requires_grad=False)
        >>> result = F.sum([x, y])
        >>> result.data
        [5, 7, 9.5]
        >>> result.role
        'first input and second input'
        >>> result.requires_grad
        True
    """  # noqa: E501
    return Sum.apply(x)


def split(
    x: Variable,
    sep: Optional[Union[str, Variable]] = None,
    maxsplit: Optional[Union[int, Variable]] = -1,
) -> List[Variable]:
    r"""
    Implements a split operation for `Variable` instances within the
    `afnio` framework, supporting automatic differentiation.

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
    return Split.apply(x, sep, maxsplit)


def chat_completion(
    forward_model_client: Optional[ChatCompletionModel],
    messages: MultiTurnMessages,
    inputs: Optional[Dict[str, Union[str, List[str], Variable]]] = None,
    **completion_args,
) -> Variable:
    r"""
    Implements a chat completion operation using the specified lanuage model within
    the `afnio` framework, supporting automatic differentiation.

    ### Features:
    - **Mini-Batching**: Processes multiple input dictionaries simultaneously to improve
        throughput.
    - **Asynchronous Execution**: Both the forward and backward passes are optimized to
        run asynchronous calls for each mini-batch, reducing latency.
    - **Gradient Computation**: Supports automatic differentiation for all `Variables`
        in `messages` and `inputs` arguments, maintaining the order of gradients.

    The `ChatCompletion` function generates a `Variable` responses by passing a
    composite prompt, built from a list of `messages` and optional `inputs`, to the
    `forward_model_client`. Each message is a dictionary with a 'role' (e.g., 'system',
    'user') and a list of `Variable` objects as 'content'. `inputs` is a dictionary
    containing strings, list of strings or `Variable`s providing dynamic values to fill
    placeholders within message templates. If `inputs` contain lists of strings or
    `Variable`s which `.data` field is a list, the response's `.data` field will be a
    list, corresponding to the batched results. Otherwise, the `.data` field will be a
    scalar string. Additional behavior, such as temperature or token limits, can be
    customized through `completion_args`.

    Example with scalar inputs:
        >>> system = Variable(
        ...     "You are a helpful assistant.",
        ...     role="system instruction",
        ...     requires_grad=True
        ... )
        >>> user = Variable("Translate 'Hello' to {language}.", role="user query")
        >>> messages = [
        ...     {"role": "system", "content": [system]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> inputs = {"language": Variable("Italian", role="language")}
        >>> response = F.chat_completion(
        ...     model_client,
        ...     messages,
        ...     inputs=inputs,
        ...     temperature=0.7
        ... )
        >>> print(response.data)
        'Ciao'
        'Hola'
        >>> feedback = Variable("Use only capital letters.", role="feedback")
        >>> response.backward(feedback)
        >>> system.grad[0].data
        'The system instruction should enforce the use of capital letters only.'

    Example with batched inputs:
        >>> system = Variable(
        ...     "You are a helpful assistant.",
        ...     role="system instruction",
        ...     requires_grad=True
        ... )
        >>> user = Variable("Translate 'Hello' to {language}.", role="user query")
        >>> messages = [
        ...     {"role": "system", "content": [system]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> inputs = {
        ...     "language": [
        ...         Variable("Italian", role="language"),
        ...         Variable("Spanish", role="language")
        ...     ]
        ... }
        >>> response = F.chat_completion(
        ...     model_client,
        ...     messages,
        ...     inputs=inputs,
        ...     temperature=0.7
        ... )
        >>> print(response.data)
        ['Ciao', 'Hola']
    """
    return ChatCompletion.apply(
        forward_model_client,
        messages,
        inputs,
        **completion_args,
    )


def lm_judge_evaluator(
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
    r"""
    Implements an evaluation of a model prediction using a language model (LM) as the
    judge within the `afnio` framework, supporting automatic differentiation.

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
        >>> score, explanation = F.lm_judge_evaluator(
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
        >>> score, explanation = F.lm_judge_evaluator(
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
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'The translated text should be in Italian.'
    """  # noqa: E501
    return LMJudgeEvaluator.apply(
        forward_model_client,
        messages,
        prediction,
        target,
        inputs,
        success_fn,
        reduction_fn,
        reduction_fn_purpose,
        eval_mode,
        **completion_args,
    )


def deterministic_evaluator(
    prediction: Variable,
    target: Union[str, List[str], Variable],
    eval_fn: Callable[[Variable, Union[str, Variable]], List[Any]],
    eval_fn_purpose: Union[str, Variable],
    success_fn: Optional[Callable[[List[Any]], bool]],
    reduction_fn: Optional[Callable[[List[Any]], Any]],
    reduction_fn_purpose: Optional[Union[str, Variable]],
) -> Tuple[Variable, Variable]:
    """
    Evaluates predictions deterministically using a user-defined evaluation function
    within the `afnio` framework, supporting automatic differentiation.

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
        >>> score, explanation = F.deterministic_evaluator(
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
        >>> score, explanation = F.deterministic_evaluator(
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
    return DeterministicEvaluator.apply(
        prediction,
        target,
        eval_fn,
        eval_fn_purpose,
        success_fn,
        reduction_fn,
        reduction_fn_purpose,
    )


def exact_match_evaluator(
    prediction: Variable,
    target: Union[str, List[str], Variable],
    reduction_fn: Optional[Callable[[List[Any]], Any]] = sum,
    reduction_fn_purpose: Optional[Union[str, Variable]] = "summation",
) -> Tuple[Variable, Variable]:
    """
    Evaluates predictions using exact matching within the `afnio` framework,
    supporting automatic differentiation.

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
        >>> score, explanation = F.exact_match_evaluator(prediction, target)
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
        >>> score, explanation = F.exact_match_evaluator(prediction, target)
        >>> score.data
        1
        >>> explanation.data
        'The evaluation function, designed for 'exact match', compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch, generating individual scores for each pair. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
        >>> explanation.backward()
        >>> prediction.grad[0].data
        'Reassess the criteria that led to the initial prediction of 'green'.'
    """  # noqa: E501
    return ExactMatchEvaluator.apply(
        prediction, target, reduction_fn, reduction_fn_purpose
    )
