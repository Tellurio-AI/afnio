from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from afnio._utils import MultiTurnMessages
from afnio._variable import Variable
from afnio.autodiff.evaluator import LMJudgeEvaluator as LMJudgeEvaluatorOp
from afnio.models import ChatCompletionModel

from .module import Module


class LMJudgeEvaluator(Module):
    """
    Evaluates predictions using a language model (LM) as the judge.

    This module leverages the `LMJudgeEvaluator` operation from
    `afnio.autodiff.evaluator` to perform model-based evaluations. The `forward`
    method accepts a list of `messages` that construct the evaluation prompt, with
    optional `inputs` to dynamically fill placeholders within message templates.
    A `prediction` is compared against a `target` (optional) to generate a `score`
    and an `explanation`.

    When processing a batch of predictions and targets, `reduction_fn` function
    aggregates individual scores (e.g., using `sum` to compute a total score). The
    `reduction_fn_purpose` parameter is a brief description of the aggregation’s purpose
    (e.g., `"summation"`). If aggregation is not desired, set `reduction_fn` and
    `reduction_fn_purpose` to `None`. The `success_fn` checks if all evaluations are
    successful, allowing the `backward` pass to skip unnecessary gradient computations.

    This module supports both evaluation (`eval_mode=True`) and optimization
    (`eval_mode=False`) modes.

    The `forward_model_client` specifies the LM responsible for evaluation, while
    `completion_args` allows customization of generation parameters like temperature,
    max tokens, and seed.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> from afnio.models.openai import OpenAI
        >>> from afnio import set_backward_model_client
        >>> fwd_model_client = OpenAI()
        >>> fwd_model_args = {"model": "gpt-4o", "temperature": 0.5}
        >>> set_backward_model_client("openai/gpt-4o")
        >>> class Evaluator(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.judge = cog.LMJudgeEvaluator()
        ...     def forward(self, fwd_model, messages, prediction, target, inputs, **completion_args):
        ...         return self.judge(fwd_model, messages, prediction, target, inputs, **completion_args)
        >>> task = Variable(
        ...     "Evaluate if the translation is {metric}.",
        ...     role="evaluation task",
        ...     requires_grad=True
        ... )
        >>> format = Variable(
        ...     "Provide 'score' (true/false) and 'explanation' in JSON.",
        ...     role="output format"
        ... )
        >>> metric = Variable(["accurate", "accurate"], role="metric")
        >>> user = Variable(
        ...     "<PREDICTION>{prediction}</PREDICTION><TARGET>{target}</TARGET>",
        ..      role="user query"
        ... )
        >>> prediction = Variable(
        ...     ["Hola Mundo", "Salve a tutti"],
        ...     role="translated text",
        ...     requires_grad=True
        ... )
        >>> target = ["Ciao Mondo", "Salve a tutti"]
        >>> messages = [
        ...     {"role": "system", "content": [task, format]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> model = Evaluator()
        >>> score, explanation = model(
        ...     fwd_model_client,
        ...     messages,
        ...     prediction,
        ...     target,
        ...     inputs={"metric": metric},
        ...     reduction_fn=sum,
        ...     reduction_fn_purpose="summation",
        ...     **fwd_model_args
        ... )
        >>> print(score.data)
        1
        >>> print(explanation.data)
        'The evaluation function, designed using an LM as the judge, compared the <DATA> fields of the predicted variable and the target variable across all samples in the batch. These scores were then aggregated using the reduction function 'summation', resulting in a final aggregated score: 1.'
        >>> explanation.backward()
        >>> system.grad[0].data
        'The translated text should be in Italian.'

    See Also:
        :class:`afnio.autodiff.evaluator.LMJudgeEvaluator` for the underlying operation.
    """  # noqa: E501

    forward_model_client: Optional[ChatCompletionModel]
    messages: MultiTurnMessages
    success_fn: Optional[Callable[[List[Any]], bool]]
    reduction_fn: Optional[Callable[[List[Any]], Any]]
    reduction_fn_purpose: Optional[Union[str, Variable]]
    eval_mode: Union[bool, Variable]
    completion_args: Dict[str, Any]

    def __init__(self):
        super().__init__()

        self.register_model("forward_model_client", None)
        self.register_chat("messages", None)
        self.register_function("success_fn", None)
        self.register_function("reduction_fn", None)
        self.register_buffer("reduction_fn_purpose", None)
        self.register_buffer("eval_mode", None)
        self.register_completion_config("completion_args", None)

    def forward(
        self,
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
        self.forward_model_client = forward_model_client
        self.messages = messages
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
        self.eval_mode = (
            eval_mode if isinstance(eval_mode, Variable) else Variable(eval_mode)
        )
        self.completion_args = completion_args
        return LMJudgeEvaluatorOp.apply(
            self.forward_model_client,
            self.messages,
            prediction,
            target,
            inputs,
            self.success_fn,
            self.reduction_fn,
            self.reduction_fn_purpose,
            self.eval_mode,
            **self.completion_args,
        )
