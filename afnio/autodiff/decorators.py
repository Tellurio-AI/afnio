def evaluator(cls):
    """
    Decorator to mark a class as an evaluator within the `afnio` framework.

    This decorator is intended to be applied to classes inheriting from `Function`.
    It sets an internal attribute `_is_evaluator` on the class to `True`, allowing
    the autodiff engine to recognize the class as an evaluator. This designation
    enables the evaluator’s `backward()` function to be called without any input
    arguments when computing gradients.

    Evaluators are responsible for assessing predictions using either deterministic
    or language model-based approaches. By using this decorator, users can build
    their own evaluator classes without worrying about internal setup details.

    Args:
        cls (type): The class being decorated, which should inherit from `Function`.

    Returns:
        type: The same class with the `_is_evaluator` attribute set to `True`.

    Example:
        >>> @evaluator
        >>> class MyCustomEvaluator(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, prediction, target):
        >>>         # Evaluation logic
        >>>         pass
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         # Backpropagation logic
        >>>         pass
    """
    cls._is_evaluator = True
    return cls
