import itertools
from abc import abstractmethod
from collections import OrderedDict, namedtuple
from copy import deepcopy
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import afnio as hf
from afnio._utils import (
    MultiTurnMessages,
    _is_valid_function,
    _validate_function,
    is_multi_turn_messages,
)
from afnio._variable import Variable, is_scalar_variable
from afnio.cognitive.parameter import Parameter
from afnio.models.model import BaseModel
from afnio.models.model_registry import MODEL_REGISTRY

# Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


class Module:
    r"""Base class for all LM pipeline modules.

    Your pipeline should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import afnio as hf
        import afnio.cognitive as cog
        import torch.cognitive.functional as F
        from afnio.models.openai import OpenAI
        from afnio import set_backward_model_client

        fwd_model_client = OpenAI()
        fwd_model_args = {"model": "gpt-4o", "temperature": 0.7}
        set_backward_model_client("openai/gpt-4o")

        class MedQA(cog.Module):
            def __init__(self):
                super().__init__()
                self.system_prompt = cog.Parameter(
                    data="You are a doctor. Only answer medical questions on these areas:",
                    role="system prompt",
                    requires_grad=True,
                )
                self.topics = cog.Parameter(
                    data="Dermatology and Cardiology",
                    role="medical topics",
                    requires_grad=False,
                )
                self.epilogue = hf.Variable(
                    data="\nThank you for your query.",
                    role="response preamble",
                )
                self.chat = cog.ChatCompletion()

            def forward(self, fwd_model, user_query, inputs, **completion_args):
                messages = [
                    {"role": "system", "content": [self.system_prompt, self.topics]},
                    {"role": "user", "content": [user_query]},
                ]
                response = self.chat(fwd_model, messages, inputs, **completion_args)
                return F.Add(response, self.epilogue)

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """  # noqa: E501

    _version: int = 1
    r"""This allows better backward support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Variable]]
    _non_persistent_buffers_set: Set[str]
    _chats: Dict[str, Optional[MultiTurnMessages]]
    _modules: Dict[str, Optional["Module"]]
    _models: Dict[str, Optional[BaseModel]]
    _completion_configs: Dict[str, Optional[Dict[str, Any]]]
    _functions: Dict[str, Optional[Callable]]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state.

        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, buffers, multi-turn chats, language model
        clients and completion configurations but simply calls into super().__setattr__
        for all other attributes.
        """
        super().__setattr__("training", True)
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_chats", OrderedDict())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_models", OrderedDict())
        super().__setattr__("_completion_configs", OrderedDict())
        super().__setattr__("_functions", OrderedDict())

    def forward(self, *args, **kwargs) -> Any:
        r"""Define the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            One should invoce the :class:`Module` instance (`Module.__call__` method)
            instead of directly calling `Module.forward()`. This way hooks are
            registered and run.
        """
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required "
            '"forward" function.'
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def extra_repr(self) -> str:
        r"""Set the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        chats = list(self._chats.keys())
        models = list(self._models.keys())
        completion_config = list(self._completion_configs.keys())
        functions = list(self._functions.keys())
        keys = (
            module_attrs
            + attrs
            + parameters
            + modules
            + buffers
            + chats
            + models
            + completion_config
            + functions
        )

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def register_buffer(
        self, name: str, variable: Optional[Variable], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, Prompt's ``format_type``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): Name of the buffer. The buffer can be accessed
                from this module using the given name.
            variable (Variable or None): Buffer to be registered. If ``None``, then
                operations that run on buffers are ignored. If ``None``, the buffer
                is **not** included in the module's :attr:`state_dict`.
            persistent (bool): Whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::
            >>> self.register_buffer('format_type', hf.Variable(data="Structure your answer as JSON.", role="format type"))
        """  # noqa: E501
        if "_buffers" not in self.__dict__:
            raise AttributeError("Cannot assign buffer before Module.__init__() call.")
        elif not isinstance(name, str):
            raise TypeError(
                f"Buffer name should be a string. Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Buffer name cannot contain ".".')
        elif name == "":
            raise KeyError('Buffer name cannot be empty string "".')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"Attribute '{name}' already exists.")
        elif variable is not None and not isinstance(variable, Variable):
            raise TypeError(
                f"Cannot assign '{type(variable).__name__}' object to buffer '{name}' "
                "(hf.Variable or None required)."
            )
        else:
            self._buffers[name] = variable
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): Name of the parameter. The parameter can be accessed
                from this module using the given name.
            param (Parameter or None): Parameter to be added to the module. If
                ``None``, then operations that run on parameters are ignored.
                If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "Cannot assign parameter before Module.__init__() call."
            )

        elif not isinstance(name, str):
            raise TypeError(
                f"Parameter name should be a string. Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Parameter name cannot contain ".".')
        elif name == "":
            raise KeyError('Parameter name cannot be empty string "".')
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"Attribute '{name}' already exists.")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"Cannot assign '{type(param).__name__}' object to parameter '{name}' "
                "(afnio.cognitive.Parameter or None required)."
            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Variable to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Variable, compute the value in "
                "the forward() method."
            )
        else:
            self._parameters[name] = param

    def register_chat(self, name: str, messages: Optional[MultiTurnMessages]) -> None:
        r"""Add multi-turn chat messages to the module.

        The chat can be accessed as an attribute using given name.

        Args:
            name (str): Name of the chat. The chat can be accessed
                from this module using the given name.
            messages (MultiTurnMessages or None): Chat to be added to the module.
                If ``None``, then operations that run on chats are ignored. If ``None``,
                the chat is **not** included in the module's :attr:`state_dict`.
        """
        if "_chats" not in self.__dict__:
            raise AttributeError("Cannot assign chat before Module.__init__() call.")

        elif not isinstance(name, str):
            raise TypeError(
                f"Chat name should be a string. " f"Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Chat name cannot contain ".".')
        elif name == "":
            raise KeyError('Chat name cannot be empty string "".')
        elif hasattr(self, name) and name not in self._chats:
            raise KeyError(f"Attribute '{name}' already exists.")

        if messages is None:
            self._chats[name] = None
        elif not is_multi_turn_messages(messages):
            raise TypeError(
                f"Cannot assign '{type(messages).__name__}' object to chat '{name}' "
                "(afnio.MultiTurnMessages or None required)."
            )
        else:
            self._chats[name] = messages

    def register_model(self, name: str, model: Optional[BaseModel]) -> None:
        r"""Add language model the module.

        The language model can be accessed as an attribute using given name.

        Args:
            name (str): Name of the model. The model can be accessed from this module
                using the given name.
            model (BaseModel or None): Model to be added to the module. If ``None``,
                then operations that run on models are ignored. If ``None``,
                the model is **not** included in the module's :attr:`state_dict`.
        """
        if "_models" not in self.__dict__:
            raise AttributeError("Cannot assign model before Module.__init__() call.")

        elif not isinstance(name, str):
            raise TypeError(
                f"Model name should be a string. " f"Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Model name cannot contain ".".')
        elif name == "":
            raise KeyError('Model name cannot be empty string "".')
        elif hasattr(self, name) and name not in self._models:
            raise KeyError(f"Attribute '{name}' already exists.")

        if model is None:
            self._models[name] = None
        elif not isinstance(model, BaseModel):
            raise TypeError(
                f"Cannot assign '{type(model).__name__}' object to model '{name}' "
                "(afnio.models.BaseModel or None required)."
            )
        else:
            self._models[name] = model

    def register_completion_config(
        self, name: str, args: Optional[Dict[str, Any]]
    ) -> None:
        r"""Register completion-specific arguments for text generation.

        This method allows dynamic storage of completion-related parameters
        such as `temperature`, `max_tokens`, `top_p`, etc.

        Args:
            name (str): Name of the completion argument set.
            args (dict or None): Dictionary of completion arguments. If ``None``, the
                argument is **not** included in the module's :attr:`state_dict`.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Completion config name should be a string. Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Completion config name cannot contain ".".')
        elif name == "":
            raise KeyError('Completion config name cannot be an empty string "".')
        elif hasattr(self, name) and name not in self._completion_configs:
            raise KeyError(f"Attribute '{name}' already exists.")

        if args is None:
            self._completion_configs[name] = None
        elif not isinstance(args, dict):
            raise TypeError(
                f"Cannot assign '{type(args).__name__}' object to "
                f"completion config '{name}' (dict or None required)."
            )
        else:
            self._completion_configs[name] = args

    def register_function(self, name: str, func: Optional[FunctionType]) -> None:
        r"""Add a function to the module.

        The function can be accessed as an attribute using given name.

        Args:
            name (str): Name of the function. The function can be accessed
                from this module using the given name.
            func (FunctionType or None): A standard Python function (i.e., a def-defined
                function, not a lambda or callable object) that can be pickled and
                registered for later execution. If None, the function is unregistered.
                If ``None``, the function is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_functions" not in self.__dict__:
            raise AttributeError(
                "Cannot assign function before Module.__init__() call."
            )
        elif not isinstance(name, str):
            raise TypeError(
                f"Function name should be a string. Got {type(name).__name__}."
            )
        elif "." in name:
            raise KeyError('Function name cannot contain ".".')
        elif name == "":
            raise KeyError('Function name cannot be empty string "".')
        elif hasattr(self, name) and name not in self._functions:
            raise KeyError(f"Attribute '{name}' already exists.")

        if func is None:
            self._functions[name] = None
        else:
            _validate_function(func)  # Validate the function before registering
            self._functions[name] = func

    def __getattr__(self, name: str) -> Any:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_chats" in self.__dict__:
            _chats = self.__dict__["_chats"]
            if name in _chats:
                return _chats[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        if "_models" in self.__dict__:
            _models = self.__dict__["_models"]
            if name in _models:
                return _models[name]
        if "_completion_configs" in self.__dict__:
            _completion_configs = self.__dict__["_completion_configs"]
            if name in _completion_configs:
                return _completion_configs[name]
        if "_functions" in self.__dict__:
            _functions = self.__dict__["_functions"]
            if name in _functions:
                return _functions[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(
        self, name: str, value: Union[Variable, "Module", MultiTurnMessages, BaseModel]
    ) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "Cannot assign parameters before Module.__init__() call."
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
                self._chats,
                self._models,
                self._completion_configs,
                self._functions,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"Cannot assign '{type(value).__name__}' as parameter '{name}' "
                    "(afnio.cognitive.Parameter or None expected)."
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "Cannot assign module before Module.__init__() call."
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                    self._chats,
                    self._models,
                    self._completion_configs,
                    self._functions,
                )
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"Cannot assign '{type(value).__name__}' as child module "
                        f"'{name}' (afnio.cognitive.Module or None expected)."
                    )
                modules[name] = value
            else:
                chats = self.__dict__.get("_chats")
                if is_multi_turn_messages(value):
                    if chats is None:
                        raise AttributeError(
                            "Cannot assign chat before Module.__init__() call."
                        )
                    remove_from(
                        self.__dict__,
                        self._parameters,
                        self._modules,
                        self._buffers,
                        self._non_persistent_buffers_set,
                        self._models,
                        self._completion_configs,
                        self._functions,
                    )
                    chats[name] = value
                elif chats is not None and name in chats:
                    if value is not None:
                        raise TypeError(
                            f"Cannot assign '{type(value).__name__}' as chat '{name}' "
                            "(afnio.MultiTurnMessages or None expected)."
                        )
                    chats[name] = value
                else:
                    models = self.__dict__.get("_models")
                    if isinstance(value, BaseModel):
                        if models is None:
                            raise AttributeError(
                                "Cannot assign model before Module.__init__() call."
                            )
                        remove_from(
                            self.__dict__,
                            self._parameters,
                            self._modules,
                            self._buffers,
                            self._non_persistent_buffers_set,
                            self._chats,
                            self._completion_configs,
                            self._functions,
                        )
                        models[name] = value
                    elif models is not None and name in models:
                        if value is not None:
                            raise TypeError(
                                f"Cannot assign '{type(value).__name__}' "
                                f"as model '{name}' "
                                "(afnio.models.BaseModel or None expected)."
                            )
                        models[name] = value
                    else:
                        completion_configs = self.__dict__.get("_completion_configs")
                        if isinstance(value, dict):
                            if completion_configs is None:
                                raise AttributeError(
                                    "Cannot assign completion config "
                                    "before Module.__init__() call."
                                )
                            remove_from(
                                self.__dict__,
                                self._parameters,
                                self._modules,
                                self._buffers,
                                self._non_persistent_buffers_set,
                                self._chats,
                                self._models,
                                self._functions,
                            )
                            completion_configs[name] = value
                        elif (
                            completion_configs is not None
                            and name in completion_configs
                        ):
                            if value is not None:
                                raise TypeError(
                                    f"Cannot assign '{type(value).__name__}' "
                                    f"as completion config '{name}' "
                                    "(dict or None expected)."
                                )
                            completion_configs[name] = value
                        else:
                            functions = self.__dict__.get("_functions")
                            if _is_valid_function(value):
                                if functions is None:
                                    raise AttributeError(
                                        "Cannot assign function "
                                        "before Module.__init__() call."
                                    )
                                remove_from(
                                    self.__dict__,
                                    self._parameters,
                                    self._modules,
                                    self._buffers,
                                    self._non_persistent_buffers_set,
                                    self._chats,
                                    self._models,
                                    self._completion_configs,
                                )
                                functions[name] = value
                            elif functions is not None and name in functions:
                                if value is not None:
                                    raise TypeError(
                                        f"Cannot assign '{type(value).__name__}' "
                                        f"as function '{name}' "
                                        "(standalone function or None expected)."
                                    )
                                functions[name] = value
                            else:
                                buffers = self.__dict__.get("_buffers")
                                if buffers is not None and name in buffers:
                                    if value is not None and not isinstance(
                                        value, Variable
                                    ):
                                        raise TypeError(
                                            f"Cannot assign '{type(value).__name__}' "
                                            f"as buffer '{name}' "
                                            f"(afnio.Variable or None expected)."
                                        )
                                    buffers[name] = value
                                else:
                                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        elif name in self._chats:
            del self._chats[name]
        elif name in self._models:
            del self._models[name]
        elif name in self._completion_configs:
            del self._completion_configs[name]
        elif name in self._functions:
            del self._functions[name]
        else:
            super().__delattr__(name)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~afnio.cognitive.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): A dict where state will be stored.
            prefix (str): The prefix for parameters, buffers, chats, models,
                completion configs and functions used in this module.
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        for name, chat in self._chats.items():
            if chat is not None:
                detached_chat = [
                    {
                        "role": message["role"],
                        "content": [
                            var if keep_vars else var.detach()
                            for var in message["content"]
                        ],
                    }
                    for message in chat
                ]
                destination[prefix + name] = detached_chat
        for name, model in self._models.items():
            if model is not None:
                # Always trigger custom __deepcopy__
                destination[prefix + name] = deepcopy(model)
        for name, config in self._completion_configs.items():
            if config is not None:
                destination[prefix + name] = config
        for name, func in self._functions.items():
            if func is not None:
                destination[prefix + name] = func

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "get_extra_state", Module.get_extra_state)
            is not Module.get_extra_state
        ):
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which
    # case `state_dict` returns back that same object. But if they pass nothing, an
    # `OrderedDict` is created and returned.
    T_destination = TypeVar("T_destination", bound=Dict[str, Any])

    def state_dict(
        self,
        *,
        destination: T_destination = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> T_destination:
        r"""Return a dictionary containing references to the whole state of the module.

        Parameters, persistent buffers (e.g. running averages), multi-turn chats,
        models, completion configs and functions are included. Keys are corresponding
        parameter, buffer, chat, model, completion config and function names.
        Parameters, buffers, chats, models, completion configs and functions
        set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters, buffers, chats, models, completion configs
            and functions.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): A prefix added to parameter, buffer, chat, model,
                completion config and function names to compose the keys in state_dict.
                Default: ``''``.
            keep_vars (bool, optional): By default the :class:`~afnio.Variable` s
                returned in the state dict are detached from autodiff. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                A dictionary containing a whole state of the module.

        Example::

            >>> module.state_dict().keys()
            ['system_prompt', 'classification_labels', 'format_type', 'user_prompt']

        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        return destination

    def _load_chat_from_state_dict(
        self, name, key, param, input_param, error_msgs, assign
    ):
        """
        Load chat from `state_dict` while handling structure mismatches.

        This function ensures chat messages are correctly loaded into the module,
        validating structure, roles, and content sizes when `param` is already
        initialized. If `param` is `None`, the chat is registered dynamically, allowing
        `self.register_chat("messages", None)` to be used in `__init__` without
        predefining the number of messages and Variables.

        Args:
            name (str): Attribute name of the chat in the module.
            key (str): Attribute name of the chat in the state dictionary.
            param (Optional[MultiTurnMessages]): Existing chat structure.
            input_param (MultiTurnMessages): Chat data from `state_dict`.
            error_msgs (List[str]): Accumulator for mismatch error messages.
            assign (bool): Whether to directly assign `input_param` to the module.
        """
        if param is not None:
            if not is_multi_turn_messages(input_param):
                error_msgs.append(
                    f"While copying the chat '{key}', expected a "
                    f"afnio.MultiTurnMessages from checkpoint, "
                    f"but received {type(input_param).__name__}."
                )
                return

            if len(input_param) != len(param):
                error_msgs.append(
                    f"Size mismatch for chat '{key}': copying a chat with "
                    f"{len(input_param)} messages from checkpoint, but the "
                    f"chat in the current model has {len(param)} messages."
                )
                return

            for i, (msg_input, msg_param) in enumerate(zip(input_param, param)):
                if msg_input["role"] != msg_param["role"]:
                    error_msgs.append(
                        f"Role mismatch for chat '{key}' at message {i}: "
                        f"copying a role '{msg_input['role']}' from checkpoint, "
                        f"but the role in the current model is '{msg_param['role']}'."
                    )
                    return

                if len(msg_input["content"]) != len(msg_param["content"]):
                    error_msgs.append(
                        f"Content size mismatch for chat '{key}' at message {i}: "
                        f"copying {len(msg_input['content'])} variables from "
                        f"checkpoint, but the message in the current model has "
                        f"{len(msg_param['content'])} variables."
                    )
                    return

                for j, (var_input, var_param) in enumerate(
                    zip(msg_input["content"], msg_param["content"])
                ):
                    is_input_scalar = not isinstance(var_input.data, list)
                    is_param_scalar = not isinstance(var_param.data, list)

                    if is_input_scalar != is_param_scalar:
                        error_msgs.append(
                            f"Type mismatch for chat '{key}' at message {i}, "
                            f"variable {j}: copying a "
                            f"{'scalar' if is_input_scalar else 'non-scalar'} "
                            f"param from checkpoint, but the param in the "
                            f"current model is "
                            f"{'scalar' if is_param_scalar else 'non-scalar'}."
                        )
                        return

                    if not is_input_scalar and len(var_input.data) != len(
                        var_param.data
                    ):
                        error_msgs.append(
                            f"Size mismatch for chat '{key}' at message {i}, "
                            f"variable {j}: copying a param with `.data` list "
                            f"of length {len(input_param.data)}from "
                            f"checkpoint, but the param in the current model "
                            f"has length {len(var_param.data)}."
                        )
                        return
        try:
            with hf.no_grad():
                if assign or param is None:
                    setattr(self, name, input_param)
                else:
                    # Shape checks are already done above
                    for i, (msg_input, msg_param) in enumerate(zip(input_param, param)):
                        for j, (var_input, var_param) in enumerate(
                            zip(msg_input["content"], msg_param["content"])
                        ):
                            var_param.copy_(var_input)
        except Exception as ex:
            error_msgs.append(
                f"While copying the chat named '{key}', "
                f"an exception occurred : {ex.args}."
            )

    def _load_param_buf_from_state_dict(
        self, name, key, param, input_param, error_msgs, assign
    ):
        """
        Load parameters and buffers from `state_dict`, ensuring consistency with
        the model.

        This function validates and assigns parameters or buffers from `state_dict`,
        ensuring they match the expected type, shape, and scalar properties. If `param`
        is `None`, the parameter is registered dynamically, allowing
        `self.register_parameter(name, None)` or `self.register_buffer(name, None)`
        in `__init__` without requiring a predefined structure.

        Args:
            name (str): Attribute name of the parameter or buffer in the module.
            key (str): Attribute name of the parameter or buffer in
                the state dictionary.
            param (Optional[Variable]): Existing parameter or buffer.
            input_param (Variable): Parameter or buffer data from `state_dict`.
            error_msgs (List[str]): Accumulator for mismatch error messages.
            assign (bool): Whether to directly assign `input_param` to the module.
        """
        if param is not None:
            if not isinstance(input_param, Variable):
                error_msgs.append(
                    f'While copying the parameter named "{key}", '
                    f"expected afnio.Variable from checkpoint, "
                    f"but received {type(input_param)}"
                )
                return

            is_scalar_input_param = is_scalar_variable(input_param)
            is_scalar_param = is_scalar_variable(param)

            if (
                not is_scalar_input_param
                and not is_scalar_param
                and len(input_param.data) != len(param.data)
            ):
                # local shape should match the one in checkpoint
                error_msgs.append(
                    f"Size mismatch for '{key}': copying a param with `.data` list "
                    f"of length {len(input_param.data)} from checkpoint, "
                    f"but the param in the current model has length {len(param.data)}."
                )
                return

            if is_scalar_input_param != is_scalar_param:
                # local and checkpoint params should be both either scalar or not
                error_msgs.append(
                    f"Type mismatch for {key}: copying a "
                    f"{'scalar' if is_scalar_variable(input_param) else 'non-scalar'} "
                    f"param from checkpoint, but the param in the current model is "
                    f"{'scalar' if is_scalar_variable(param) else 'non-scalar'}."
                )
                return
        try:
            with hf.no_grad():
                if assign or param is None:
                    # Shape checks are already done above
                    if isinstance(param, Parameter):
                        if not isinstance(input_param, Parameter):
                            input_param = Parameter(
                                input_param.data,
                                input_param.role,
                                requires_grad=param.requires_grad,
                            )
                        else:
                            input_param.requires_grad_(param.requires_grad)
                    setattr(self, name, input_param)
                else:
                    param.copy_(input_param)
        except Exception as ex:
            model_data_info = (
                "scalar value"
                if is_scalar_param
                else f"list of length {len(param.data)}"
            )
            checkpoint_data_info = (
                "scalar value"
                if is_scalar_input_param
                else f"list of length {len(input_param.data)}"
            )
            error_msgs.append(
                f"While copying the parameter named '{key}', "
                f"which is a {model_data_info} in the current model and "
                f"which is a {checkpoint_data_info} in the checkpoint, "
                f"an exception occurred : {ex.args}."
            )

    def _load_model_from_state_dict(
        self, name, key, param, input_param, error_msgs, model_clients
    ):
        """
        Load model clients from `state_dict`, ensuring they are provided
        via `model_clients`.

        This function enforces that model clients must be pre-initialized and passed
        through `model_clients`. It validates the class type and ensures consistency
        between the expected and provided model client types.

        Args:
            name (str): Attribute name of the model client in the module.
            key (str): Attribute name of the model client in the state dictionary.
            param (Optional[BaseModel]): Existing model instance.
            input_param (dict): Serialized model client data from `state_dict`.
            error_msgs (List[str]): Accumulator for mismatch error messages.
            model_clients (Dict[str, BaseModel]): Pre-initialized model clients.

        Raises:
            ValueError: If the required model client is missing from `model_clients`.
        """
        if not isinstance(input_param, dict) or "class_type" not in input_param:
            error_msgs.append(
                f"While copying the model client '{key}', expected a serialized "
                f"dictionary with a 'class_type' entry from checkpoint, "
                f"but received {type(input_param).__name__}."
            )
            return

        model_cls_name = input_param["class_type"]
        model_cls = MODEL_REGISTRY.get(model_cls_name)

        if model_cls is None or not issubclass(model_cls, BaseModel):
            error_msgs.append(
                f"Model client '{key}' referenced an unknown or invalid class "
                f"type '{model_cls_name}' from checkpoint. "
                f"Ensure that '{model_cls_name}' is registered in MODEL_REGISTRY "
                f"and inherits from BaseModel."
            )
            return

        if key not in model_clients or not isinstance(model_clients[key], model_cls):
            error_msgs.append(
                f"Missing model client for '{key}' of expected "
                f"type '{model_cls}'. Please provide an instance "
                f"of '{model_cls}' using the `model_clients` "
                f"dictionary when calling `load_state_dict()`."
            )
            return

        if param is not None and param.get("class_type") != model_cls_name:
            error_msgs.append(
                f"Type mismatch for model client '{key}': expected an instance of "
                f"'{param.get('class_type', 'Unknown')}' from checkpoint, "
                f"but received '{model_cls_name}'."
            )
            return

        try:
            # Create new model client istance
            setattr(self, name, model_clients[key])

            # Add usage metadata to new model client instance
            usage = input_param.get("usage", {})
            new_model = getattr(self, name)
            new_model.update_usage(usage)
        except Exception as ex:
            error_msgs.append(
                f"Failed to initialize model client '{key}' of type '{model_cls_name}' "
                f"from state_dict: {ex.args}."
            )

    def _load_completion_config_from_state_dict(
        self, name, key, param, input_param, error_msgs
    ):
        """
        Load completion configuration from `state_dict`, ensuring consistency
        with the model.

        This function assigns completion configurations from `state_dict`. If `param` is
        `None`, the completion config is registered dynamically, allowing
        `self.register_completion_config("completion_config", None)`
        to be used in `__init__` without requiring predefined values.

        Args:
            name (str): Attribute name of the completion config in the module.
            key (str): Attribute name of the completion config in the state dictionary.
            param (Optional[Dict[str, Any]]): Existing completion config dictionary.
            input_param (Dict[str, Any]): Completion config data from `state_dict`.
            error_msgs (List[str]): Accumulator for mismatch error messages.
        """
        if param is not None:
            if not isinstance(input_param, dict):
                error_msgs.append(
                    f"While copying the completion config '{key}', "
                    f"expected a dictionary from checkpoint, "
                    f"but received {type(input_param).__name__}."
                )
                return

        try:
            setattr(self, name, input_param)
        except Exception as ex:
            error_msgs.append(
                f"While copying the completion config named '{key}', "
                f"an exception occurred: {ex.args}."
            )

    def _load_function_from_state_dict(self, name, key, param, input_param, error_msgs):
        """
        Load function from `state_dict`, ensuring consistency with the model.

        This function assigns functions from `state_dict`. If `param` is
        `None`, the function is registered dynamically, allowing
        `self.register_function("function", None)` to be used in `__init__`
        without requiring predefined values.

        Args:
            name (str): Attribute name of the function in the module.
            key (str): Attribute name of the function in the state dictionary.
            param (Optional[Callable[..., Any]]): Existing function reference.
            input_param (Callable[..., Any]): Function data from `state_dict`.
            error_msgs (List[str]): Accumulator for mismatch error messages.
        """
        if param is not None:
            if not _is_valid_function(input_param):
                error_msgs.append(
                    f"While copying the function '{key}', "
                    f"expected a standalone function from checkpoint, "
                    f"but received {type(input_param).__name__}."
                )
                return

        try:
            setattr(self, name, input_param)
        except Exception as ex:
            error_msgs.append(
                f"While copying the function named '{key}', "
                f"an exception occurred: {ex.args}."
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        model_clients: Dict[str, BaseModel] = None,
    ):
        r"""Copy parameters, buffers, chats, models, completion configs and functions
        from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~afnio.cognitive.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers_chats` that indicates whether keys should be
        assigned their corresponding Variable or MultiTurnMessages in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~afnio.cognitive.Module.load_state_dict`.
            So it can be modified.

        Args:
            state_dict (dict): A dict containing parameters, persistent buffers,
                chats, models, completion configs and functions.
            prefix (str): The prefix for parameters, buffers, chats, models,
                completion configs and functions used in this module.
            local_metadata (dict): A dict containing the metadata for this module.
            strict (bool): Whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters, buffers, chats, models, completion configs and functions
                in this module.
            missing_keys (list of str): If ``strict=True``, add missing keys to
                this list.
            unexpected_keys (list of str): If ``strict=True``, add unexpected
                keys to this list.
            error_msgs (list of str): Error messages should be added to this
                list, and will be reported together in
                :meth:`~afnio.cognitive.Module.load_state_dict`.
            model_clients (dict, optional): A dictionary mapping model client keys
                (e.g., 'fw_model_client') to their respective instances of
                :class:`BaseModel`. These instances will be used to reconstruct
                any model clients referenced within the optimizer state.
                If a required model client is missing, an error will be raised
                with instructions on how to provide the missing client.
        """
        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(),
            persistent_buffers.items(),
            self._chats.items(),
            self._models.items(),
            self._completion_configs.items(),
            self._functions.items(),
        )
        local_state = {k: v for k, v in local_name_params}
        assign_to_params_buffers_chats = local_metadata.get(
            "assign_to_params_buffers_chats", False
        )
        model_clients = model_clients or {}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # Handle chats
                if name in self._chats:
                    self._load_chat_from_state_dict(
                        name,
                        key,
                        param,
                        input_param,
                        error_msgs,
                        assign_to_params_buffers_chats,
                    )
                # Handle models
                elif name in self._models:
                    self._load_model_from_state_dict(
                        name,
                        key,
                        param,
                        input_param,
                        error_msgs,
                        model_clients,
                    )
                # Handle completion configs
                elif name in self._completion_configs:
                    self._load_completion_config_from_state_dict(
                        name,
                        key,
                        param,
                        input_param,
                        error_msgs,
                    )
                # Handle functions
                elif name in self._functions:
                    self._load_function_from_state_dict(
                        name,
                        key,
                        param,
                        input_param,
                        error_msgs,
                    )
                else:
                    # Handle parameters and buffers
                    self._load_param_buf_from_state_dict(
                        name,
                        key,
                        param,
                        input_param,
                        error_msgs,
                        assign_to_params_buffers_chats,
                    )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(self.__class__, "set_extra_state", Module.set_extra_state)
            is not Module.set_extra_state
        ):
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix) :].split(".", 1)  # noqa: E203
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        model_clients: Dict[str, BaseModel] = None,
    ):
        r"""Copy parameters, buffers, chats, models, completion configs and functions
        from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must exactly
        match the keys returned by this module's
        :meth:`~afnio.cognitive.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict`.

        Args:
            state_dict (dict): A dict containing parameters, persistent buffers,
                chats, models, completion configs and functions.
            strict (bool, optional): Whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~afnio.cognitive.Module.state_dict` function.
                Default: ``True``
            assign (bool, optional): When ``False``, the properties of the Variables
                in the current module are preserved while when ``True``, the
                properties of the Variables in the state dict are preserved. The only
                exception is the ``requires_grad`` field of
                :class:`~afnio.cognitive.Parameter`s for which the value from the
                module is preserved.
                Default: ``False``
            model_clients (dict, optional): A dictionary mapping model client keys
                (e.g., 'fw_model_client') to their respective instances of
                :class:`BaseModel`. These instances will be used to reconstruct
                any model clients referenced within the optimizer state.
                If a required model client is missing, an error will be raised
                with instructions on how to provide the missing client.

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are
                    expected by this module but missing from
                    the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.

        Note:
            If a parameter, or buffer, or chat, or model, or completion config,
            or function is registered as ``None`` and its corresponding key exists in
            :attr:`state_dict`, :meth:`load_state_dict` will raise a ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata["assign_to_params_buffers_chats"] = assign
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
                model_clients,
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in local_state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)  # noqa: F821

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        r"""Add a child module to the current module.

        This method explicitly adds a child module to the current module's hierarchy.
        The child module can then be accessed as an attribute using the given name
        and will be registered in the `_modules` dictionary.

        **When to use**:
        - Use `register_module()` when dynamically adding submodules at runtime,
        especially when the submodule name is determined programmatically.
        - This can be useful for creating flexible and modular architectures.

        **When it's unnecessary**:
        - Directly assigning the module to an attribute (e.g.,
        `self.module_name = SubModule()`) automatically registers it, so using
        `register_module()` is unnecessary in such cases.

        Args:
            name (str): Name of the child module. The child module can be accessed from
                this module using the given name.
            module (Module): Child module to be added to the module.

        Raises:
            TypeError: If `module` is not a subclass of `Module` or
                if `name` is not a string.
            KeyError: If `name` is already an attribute of the module but not
                in `_modules`, or if `name` contains invalid characters
                such as '.' or is empty.

        Example::
            >>> class DynamicPipeline(cog.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         # Dynamically add submodules
            >>>         for i in range(3):
            >>>             self.register_module(f"layer_{i}", cog.Module())

            >>> pipeline = DynamicPipeline()
            >>> print(pipeline._modules.keys())
            odict_keys(['layer_0', 'layer_1', 'layer_2'])

        .. note::
            If assigning submodules using standard attribute assignment
            (e.g., `self.submodule = SubModule()`), calling `register_module()`
            explicitly is not required. Direct assignment automatically registers
            the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(
                f"'{type(module).__name__}' is not a valid Module subclass."
            )
        elif not isinstance(name, str):
            raise TypeError(
                f"Module name must be a string, but got '{type(name).__name__}'."
            )
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(
                f"Attribute '{name}' already exists and "
                f"cannot be used as a module name."
            )
        elif "." in name:
            raise KeyError(f"Module name cannot contain '.', but got: '{name}'.")
        elif name == "":
            raise KeyError('Module name cannot be an empty string ""')
        self._modules[name] = module

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict.

        Returns:
            object: Any extra state to store in the module's state_dict.
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that "
            "should never be called."
        )

    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within
        its `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`.
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that "
            "should never be called. "
        )

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                value = v

                # Convert chat messages into a hashable structure
                if is_multi_turn_messages(v):
                    v = tuple((entry["role"], tuple(entry["content"])) for entry in v)

                # Convert dictionaries (e.g., completion_args) into hashable tuples
                elif isinstance(v, dict):
                    v = tuple(sorted(v.items()))

                if v is None or v in memo:
                    continue

                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, value

    def buffers(self, recurse: bool = True) -> Iterator[Variable]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            hf.Variable: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.data)
            <class 'afnio.Variable'> ("Structure your answer as JSON.")
            <class 'afnio.Variable'> ("Use the format\n\n{\n  \"response\": \"Your concise answer here.\"\n}")
        """  # noqa: E501
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Variable]]:
        r"""Return an iterator over module buffers, yielding both the name of
        the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers
                in the result. Defaults to True.

        Yields:
            (str, hf.Variable): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>     if "format_type" in name:
            >>>         print(param.data)
        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in pipeline.parameters():
            >>>     print(type(param), param.data)
            <class 'cog.Parameter'> ("You are a doctor.")
            <class 'cog.Parameter'> ("Only answer with YES or NO.")
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the
        parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>     if "prompt" in name:
            >>>         print(param.data)
        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def chats(self, recurse: bool = True) -> Iterator[MultiTurnMessages]:
        r"""Return an iterator over module multi-turn chats.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields chats of this module
                and all submodules. Otherwise, yields only chats that
                are direct members of this module.

        Yields:
            MultiTurnMessages: module chats

        Example::

            >>> for chat in pipeline.chats():
            >>>     print(type(chat), chat)
            <class 'cog.MultiTurnMessages'> [{'role': 'system', 'content': [Variable(data=You are a doctor., role=system instruction, requires_grad=False)]}, {'role': 'user', 'content': [Variable(data=Is {item} a disease?, role=user query, requires_grad=False)]}]
            <class 'cog.MultiTurnMessages'> [{'role': 'system', 'content': [Variable(data=You are a helpful assistant., role=system instruction, requires_grad=False), Variable(data=Only answer with YES or NO., role=user query, requires_grad=False)]}]
        """  # noqa: E501
        for _, chat in self.named_chats(recurse=recurse):
            yield chat

    def named_chats(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[MultiTurnMessages]:
        r"""Return an iterator over module multi-turn chats, yielding both
        the name of chat as well as the chat itself.

        Args:
            prefix (str): prefix to prepend to all chat names.
            recurse (bool): if True, then yields chats of this module
                and all submodules. Otherwise, yields only chats that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                chats in the result. Defaults to True.

        Yields:
            (str, MultiTurnMessages): Tuple containing the name and chat

        Example::

            >>> for name, chat in self.named_chats():
            >>>     if "messages" in name:
            >>>         print(messages[0]["role"])
        """
        gen = self._named_members(
            lambda module: module._chats.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def models(self, recurse: bool = True) -> Iterator[BaseModel]:
        r"""Return an iterator over module language model clients.

        Args:
            recurse (bool): if True, then yields models of this module
                and all submodules. Otherwise, yields only models that
                are direct members of this module.

        Yields:
            BaseModel: module model

        Example::

            >>> for model in pipeline.models():
            >>>     print(type(model))
            <class 'afnio.models.openai.AsyncOpenAI'>
        """
        for _, model in self.named_models(recurse=recurse):
            yield model

    def named_models(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, BaseModel]]:
        r"""Return an iterator over module model clients, yielding both the name of the
        model as well as the model itself.

        Args:
            prefix (str): prefix to prepend to all model names.
            recurse (bool): if True, then yields models of this module
                and all submodules. Otherwise, yields only models that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                models in the result. Defaults to True.

        Yields:
            (str, BaseModel): Tuple containing the name and model

        Example::

            >>> for name, model in self.named_models():
            >>>     print(name, type(model))
            model_client <class 'afnio.models.openai.AsyncOpenAI'>
        """
        gen = self._named_members(
            lambda module: module._models.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def completion_configs(self, recurse: bool = True) -> Iterator[Dict[str, Any]]:
        r"""Return an iterator over registered completion configs.

        Args:
            recurse (bool): if True, then yields completion configs of this module
                and all submodules. Otherwise, yields only completion configs that
                are direct members of this module.

        Yields:
            dict: completion arguments

        Example::
            >>> for config in model.completion_configs():
            >>>     print(config)
            {"model": "gpt-4o", "seed": 42, "temperature": 0}
        """
        for _, config in self.named_completion_configs(recurse=recurse):
            yield config

    def named_completion_configs(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        r"""Return an iterator over module completion configs, yielding both the name of
        the completion config as well as the completion config itself.

        Args:
            prefix (str): prefix to prepend to all completion config names.
            recurse (bool): if True, then yields completion configs of this module
                and all submodules. Otherwise, yields only completion configs that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                completion configs in the result. Defaults to True.

        Yields:
            (str, dict): Tuple containing the name and completion configs

        Example::

            >>> for name, config in self.named_completion_configs():
            >>>     print(name, type(config))
            chat.completion_args {'model': 'gpt-4o', 'seed': 42, 'temperature': 0}
        """
        gen = self._named_members(
            lambda module: module._completion_configs.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def functions(self, recurse: bool = True) -> Iterator[Dict[str, Any]]:
        r"""Return an iterator over registered functions.

        Args:
            recurse (bool): if True, then yields functions of this module
                and all submodules. Otherwise, yields only functions that
                are direct members of this module.

        Yields:
            Callable: functions

        Example::
            >>> for func in model.functions():
            >>>     print(func)
            <built-in function sum>
            <function my_func at 0x7e7a0665b9c0>
        """
        for _, config in self.named_functions(recurse=recurse):
            yield config

    def named_functions(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        r"""Return an iterator over module functions, yielding both the name of
        the function as well as the function itself.

        Args:
            prefix (str): prefix to prepend to all function names.
            recurse (bool): if True, then yields functions of this module
                and all submodules. Otherwise, yields only functions that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                functions in the result. Defaults to True.

        Yields:
            (str, Callable): Tuple containing the name and functions

        Example::

            >>> for name, func in self.named_functions():
            >>>     print(name, func)
            reduction_fn <built-in function sum>
            eval_fn <function my_func at 0x7e7a0665b9c0>
        """
        gen = self._named_members(
            lambda module: module._functions.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding both the name
        of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]:
        r"""Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``add`` will be returned only once.

        Example::

            >>> class MyPipeline(cog.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         add = cog.Add()
            ...         self.module1 = add
            ...         self.module2 = add
            >>>     def forward(self, x, y):
            ...         out1 = self.module1(x, x)
            ...         out2 = self.module2(x, y)
            ...         return out1 + out2
            >>> pipeline = MyPipeline()
            >>> for idx, m in enumerate(model.modules()):
            ...     print(idx, '->', m)
            0 -> MyModel(
            (module1): Module()
            (module2): Module()
            )
            1 -> Module()
        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all modules in the network, yielding both
        the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances
                in the result or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``add`` will be returned only once.

        Example::

            >>> class MyPipeline(cog.Module):
            ...     def __init__(self):
            ...     super().__init__()
            ...     add = cog.Add()
            ...     self.module1 = add
            ...     self.module2 = add
            >>> def forward(self, x, y):
            ...     out1 = self.module1(x, x)
            ...     out2 = self.module2(x, y)
            ...     return out1 + out2
            >>> pipeline = MyPipeline()
            >>> for idx, m in enumerate(model.named_modules()):
            ...     print(idx, '->', m)
            0 -> ('', MyModel(
            (module1): Module()
            (module2): Module()
            ))
            1 -> ('module1', Module())

        Example::

            >>> class MyPipeline(cog.Module):
            ...     def __init__(self):
            ...     super().__init__()
            ...     add = cog.Add()
            ...     self.module1 = add
            ...     self.module2 = add
            >>> def forward(self, x, y):
            ...     out1 = self.module1(x, x)
            ...     out2 = self.module2(x, y)
            ...     return out1 + out2
            >>> pipeline = MyPipeline()
            >>> for idx, m in enumerate(model.named_modules(remove_duplicate=False)):
            ...     print(idx, '->', m)
            0 -> ('', MyModel(
            (module1): Module()
            (module2): Module()
            ))
            1 -> ('module1', Module())
            2 -> ('module2', Module())
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )

    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("Training mode is expected to be boolean.")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected.

        This is equivalent with
        :meth:`self.train(False) <afnio.cognitive.Module.train>`.

        Returns:
            Module: self
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autodiff should record operations on parameters and chats
        in this module.

        This method sets the :attr:`requires_grad` attributes of all module parameters
        in-place. It also sets the :attr:`requires_grad` attributes of all the
        `Variables` within the content of multi-turn chats.

        **Effect on Parameters:**
            - Sets :attr:`requires_grad` for each registered parameter in the module.

        **Effect on Chats:**
            - Iterates through all multi-turn chats and sets :attr:`requires_grad`
            for each `Variable` in the "content" key of the chat's message.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually.

        Args:
            requires_grad (bool): Whether autodiff should record operations on
                                  parameters and chats in this module.
                                  Default: ``True``.

        Returns:
            Module: self
        """
        # Set requires_grad on all parameters
        for p in self.parameters():
            p.requires_grad_(requires_grad)

        # Set requires_grad on all variables in message content
        for chat in self.chats():
            for message in chat:
                for variable in message["content"]:
                    variable.requires_grad_(requires_grad)

        return self

    def empty_grad(self) -> None:
        r"""Reset gradients of all model parameters and content variables
        in chats' messages.

        This method is useful for clearing out gradients before starting a new
        optimization step. It ensures that both module parameters and Variables within
        multi-turn chat's message contents have their gradients reset, avoiding
        unintended gradient accumulation.
        """
        # Reset gradients of all parameters
        for p in self.parameters():
            if p.grad:
                p.grad = []

        # Reset gradients of all variables in message content
        for chat in self.chats():
            for message in chat:
                for variable in message["content"]:
                    if variable.grad:
                        variable.grad = []
