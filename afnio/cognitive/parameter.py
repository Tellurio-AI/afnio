from afnio._variable import Variable


class Parameter(Variable):
    """
    A subclass of Variable that represents learnable parameters (similar to
    nn.Parameter). These parameters are typically text-based, learnable weights,
    embeddings, etc.
    """

    def __init__(self, data=None, role=None, requires_grad=True):
        super().__init__(data=data, role=role, requires_grad=requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data, self.role, self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return "Parameter containing:\n  " + super().__repr__()
