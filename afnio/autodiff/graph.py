from typing import NamedTuple, Optional, Tuple


class Node:
    def __init__(self, next_functions: Optional[Tuple["GradientEdge"]] = None):
        self._next_functions = next_functions if next_functions else ()
        self._name = None
        self.node_id = None

    def __str__(self):
        return f"<{self.__class__.__name__} object at {hex(id(self))}>"

    def apply(self, *args):
        raise NotImplementedError("Subclasses should implement this method.")

    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import afnio
            >>> import afnio.cognitive.functional as F
            >>> a = hf.Variable("Hello,", requires_grad=True)
            >>> b = hf.Variable("world!", requires_grad=True)
            >>> c = F.sum([a, b])
            >>> assert isinstance(c.grad_fn, afnio.autodiff.graph.Node)
            >>> print(c.grad_fn.name())
            SumBackward0
        """
        return self._name

    @property
    def next_functions(self) -> Tuple["GradientEdge"]:
        return self._next_functions

    @next_functions.setter
    def next_functions(self, edges: Tuple["GradientEdge", ...]):
        self._next_functions = edges


class GradientEdge(NamedTuple):
    """Object representing a given gradient edge within the autodiff graph.

    To get the gradient edge where a given Variable gradient will be computed,
    you can do ``edge = autodiff.graph.get_gradient_edge(tensor)``.
    """

    node: Node
    output_nr: int

    def __repr__(self):
        return f"(<{self.node.__class__.__name__} object at {hex(id(self.node))}>, {self.output_nr})"  # noqa: E501

    def __str__(self):
        return f"(<{self.node.__class__.__name__} object at {hex(id(self.node))}>, {self.output_nr})"  # noqa: E501
