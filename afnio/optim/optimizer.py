import logging
import time
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from typing_extensions import TypeAlias, overload

from afnio._utils import _deserialize_output, _serialize_arg
from afnio._variable import Variable
from afnio.logging_config import configure_logging
from afnio.models.model import BaseModel
from afnio.models.model_registry import MODEL_REGISTRY
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio._optimizer_registry import register_optimizer
from afnio.tellurio._variable_registry import VARIABLE_REGISTRY
from afnio.tellurio.client import get_default_client

StateDict: TypeAlias = Dict[str, Any]

ParamsT: TypeAlias = Union[Iterable[Variable], Iterable[Dict[str, Any]]]


# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): An iterable of :class:`afnio.Variable` s or
            :class:`dict` s. Specifies what Variables should be optimized.
        defaults: (dict): A dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    defaults: Dict[str, Any] = {}
    state: DefaultDict[Variable, Any] = defaultdict(dict)
    param_groups: List[Dict[str, Any]] = []
    optimizer_id: Optional[str]

    def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:
        # Websocket attributes
        self.optimizer_id = None
        # Internal attributes
        self.defaults = {}
        self.state = defaultdict(dict)
        self.param_groups = []

        # Determine which child class is instantiating this Optimizer
        child_class = self.__class__.__name__

        try:
            # Get the singleton websocket client
            _, ws_client = get_default_client()

            params = list(params)
            payload = {
                "optimizer_name": child_class,
                "params": _serialize_arg(params),
                "defaults": _serialize_arg(defaults),
            }
            response = run_in_background_loop(
                ws_client.call("create_optimizer", payload)
            )
            logger.debug(f"Optimizer created and shared with the server: {self!r}")

            result = response.get("result", {})
            optimizer_id = result.get("optimizer_id")
            defaults = result.get("defaults")
            state = result.get("state")
            param_groups = result.get("param_groups")

            if not optimizer_id:
                logger.error(
                    f"Server did not return an optimizer_id "
                    f"for payload: {payload!r}, response: {response!r}"
                )
                raise RuntimeError(
                    "Failed to create Optimizer: server did not return a variable_id."
                )
            self.optimizer_id = optimizer_id
            self.defaults = _deserialize_output(defaults)
            self.state = _deserialize_output(state)
            self.param_groups = _deserialize_output(param_groups)
            register_optimizer(self)
        except Exception as e:
            logger.error(f"Failed to share Optimizer with the server: {e}")
            raise

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    def _patch_step_function(self) -> None:
        self._clear_grad_profile_name = (
            f"Optimizer.clear_grad#{self.__class__.__name__}.clear_grad"
        )

    def state_dict(self) -> StateDict:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and momentum,
            as well as a List of parameter IDs of the parameters in the group.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``cog.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {
                        'momentum_buffer': [
                            (
                                Parameter(data='You are...', role='system prompt', requires_grad=True),
                                [Variable(data='The system prompt should...', role='gradient for system prompt')]
                            )
                        ]
                    },
                    1: {
                        'momentum_buffer': [
                            (
                                Parameter(data='Answer this...', role='instructin prompt', requires_grad=True),
                                [Variable(data='The instruction prompt must...', role='gradient to instruction prompt')]
                            )
                        ]
                    }
                },
                'param_groups': [
                    {
                        'model_client': {'class_type': 'AsyncOpenAI'},
                        'messages': [
                            {
                                'role': 'system',
                                'content': [Variable(data='You are part of an optimization system...', role='optimizer system prompt', requires_grad=False)]
                            },
                            {
                                'role': 'user',
                                'content': [Variable(data='Here is the variable you need...', role='optimizer user prompt', requires_grad=False)]
                            }
                        ],
                        'inputs': {},
                        'constraints': [],
                        'momentum': 2,
                        'completion_args': {'model': 'gpt-4o'},
                        'params': [0, 1]
                    }
                ]
            }

        """  # noqa: E501

        # Save order indices instead of Variables
        param_mappings: Dict[int, int] = {}
        start_index = 0

        def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}

            # Custom serialization for model clients
            for key, value in packed.items():
                if isinstance(value, BaseModel):
                    packed[key] = deepcopy(value)  # Trigger custom __deepcopy__

            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, Variable) else k): v
            for k, v in self.state.items()
        }

        state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }

        return state_dict

    @staticmethod
    def _process_value_according_to_param_policy(
        param: Variable,
        value: Variable,
        param_id: int,
        param_groups: List[Dict[Any, Any]],
        key: Hashable = None,
    ) -> Variable:
        assert param_groups is not None
        if key == "step":
            return value
        else:
            if param.is_floating_point():
                return value.to(dtype=float)
            else:
                return value

    def load_state_dict(
        self, state_dict: StateDict, model_clients: Dict[str, BaseModel] = None
    ) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): Optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
            model_clients (dict, optional): A dictionary mapping model client keys
                (e.g., 'fw_model_client') to their respective instances of
                :class:`BaseModel`. These instances will be used to reconstruct
                any model clients referenced within the optimizer state.
                If a required model client is missing, an error will be raised
                with instructions on how to provide the missing client.

        Raises:
            ValueError: If the provided state_dict is invalid, such as when the
                parameter groups or their sizes do not match the current optimizer
                configuration.

            ValueError: If a required model client is missing from the
                `model_clients` dictionary, with details about the expected
                model client type and key.

        Example:
            >>> openai_client = AsyncOpenAI()
            >>> optimizer.load_state_dict(saved_state_dict, model_clients={
            ...     'model_client': openai_client)
            ... })
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "Loaded state dict has a different number of parameter groups."
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "Loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group."
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all variables to device of param."""
            if isinstance(value, Variable):
                return Optimizer._process_value_according_to_param_policy(
                    param, value, param_id, param_groups, key
                )
            elif isinstance(value, dict):
                return {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
            elif isinstance(value, Iterable):
                return type(value)(
                    _cast(param, v, param_id=param_id, param_groups=param_groups)
                    for v in value
                )
            else:
                return value

        # Copy state assigned to params (and cast variables to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: DefaultDict[Variable, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param, v, param_id=k, param_groups=state_dict["param_groups"]
                )
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(
            group: Dict[str, Any], new_group: Dict[str, Any]
        ) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group

        # Reconstruct model clients if needed
        model_clients = model_clients or {}
        for group in saved_groups:
            for key, value in group.items():
                if isinstance(value, dict) and "class_type" in value:
                    cls_name = value["class_type"]
                    cls = MODEL_REGISTRY.get(cls_name)
                    if cls and issubclass(cls, BaseModel):
                        if key in model_clients:
                            # Create new model client istance
                            group[key] = model_clients[key]

                            # Add usage metadata to new model client instance
                            usage = value.get("usage", {})
                            group[key].update_usage(usage)
                        else:
                            raise ValueError(
                                f"Missing model client for '{key}' of expected type "
                                f"'{cls_name}'. Please provide an instance of "
                                f"'{cls_name}' using the `model_clients` input "
                                f"dictionary and retry "
                                f"the `load_state_dict()` operation."
                            )

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def _on_clear_grad(self):
        """
        Notify the server that all gradients for this optimizer are being cleared.

        This method sends a 'clear_grad' RPC request to the server with the optimizer's
        ID. It waits for the server to acknowledge the request and checks that the
        response matches the optimizer's ID. If the server confirms, the method returns
        normally; otherwise, it raises an error. This ensures that the server and client
        remain synchronized regarding the clearing of gradients.

        Raises:
            RuntimeError: If the server response does not match the optimizer ID or if
                the notification fails for any reason.
        """
        payload = {
            "optimizer_id": self.optimizer_id,
        }

        try:
            _, ws_client = get_default_client()
            response = run_in_background_loop(ws_client.call("clear_grad", payload))

            # Check server response
            result_message = response.get("result", {}).get("message")
            if result_message != "Gradients cleared successfully.":
                logger.error(
                    f"Server response mismatch: {response['result']!r} "
                    f"(expected optimizer_id={self.optimizer_id!r}"
                )
                raise RuntimeError(
                    "Failed to notify server of gradient clearing: "
                    "server response does not match the update sent."
                )
            logger.debug(
                f"Gradient clearing notified to server and confirmed: "
                f"optimizer_id={self.optimizer_id!r}"
            )

        except Exception:
            logger.exception(
                f"Failed to notify server of gradient clearing: "
                f"optimizer_id={self.optimizer_id!r}"
            )
            raise

    def clear_grad(self) -> None:
        """
        Resets the gradients of all optimized :class:`Variable` s by setting
        the `.grad` attribute of each parameter to an empty list.
        """
        self._on_clear_grad()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad:
                    p.grad.clear()

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(
        self, closure: Callable[[], Tuple[Variable, Variable]]
    ) -> Tuple[Variable, Variable]: ...

    def step(
        self, closure: Optional[Callable[[], Tuple[Variable, Variable]]] = None
    ) -> Optional[Tuple[Variable, Variable]]:
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable, optional): A closure that reevaluates the model and
                returns the loss as a tuple containing a numerical score and a textual
                explanation. This closure is optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.

        .. note::
            Some optimization algorithms need to reevaluate the function multiple times,
            so you have to pass in a `closure` that allows them to recompute your model.
            The closure should clear the gradients, compute the loss, and return it.

            Example:

            .. code-block:: python

                for input, target in dataset:
                    def closure():
                        optimizer.clear_grad()
                        output = model(input)
                        loss = loss_fn(output, target)
                        loss.backward()
                        return loss

                    optimizer.step(closure)
        """
        # Set `_pending_data` for all parameters that will be optimized
        for group in self.param_groups:
            for p in group["params"]:
                p._pending_data = True
                logger.debug(
                    f"Marked variable {p.variable_id!r} as pending for data update."
                )

        try:
            # Get the singleton websocket client
            _, ws_client = get_default_client()

            payload = {
                "optimizer_id": self.optimizer_id,
            }
            response = run_in_background_loop(ws_client.call("run_step", payload))
            logger.debug(
                f"Optimization instantiated and shared with the server: "
                f"optimizer_id={self.optimizer_id!r}"
            )

            result = response.get("result", {})
            result_message = result.get("message")
            result_state = result.get("state", [])

            # Extract all variable_ids from the result_state
            # and wait for them to be registered in VARIABLE_REGISTRY
            all_var_ids = self._extract_variable_ids_from_state(result_state)
            for var_id in all_var_ids:
                _wait_for_variable(var_id)

            des_result_state = _deserialize_state(result_state)

            # Convert [param, grads] lists to (param, grads) tuples
            for state in des_result_state.values():
                if "momentum_buffer" in state:
                    state["momentum_buffer"] = [
                        (
                            tuple(pair)
                            if isinstance(pair, list) and len(pair) == 2
                            else pair
                        )
                        for pair in state["momentum_buffer"]
                    ]

            if result_message != "Optimizer step executed successfully.":
                logger.error(
                    f"Server did not return any data for optimization operation: "
                    f"payload={payload!r}, response={response!r}"
                )
                raise RuntimeError(
                    "Failed to run optimization: server did not return data."
                )

            self.state = des_result_state

            logger.debug(
                f"Optimization executed successfully: "
                f"optimizer_id={self.optimizer_id!r}"
            )

        except Exception as e:
            logger.error(f"Failed to run optimization on the server: {e}")
            raise

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can
        be made trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Variables should be optimized along with
                group specific optimization options.
        """
        try:
            # Get the singleton websocket client
            _, ws_client = get_default_client()

            messages = self.defaults.get("messages", [])
            payload = {
                "optimizer_id": self.optimizer_id,
                "messages": _serialize_arg(messages),
                "param_group": _serialize_arg(param_group),
            }
            response = run_in_background_loop(
                ws_client.call("add_param_group", payload)
            )
            logger.debug(
                f"Param group added and shared with the server: {param_group!r}"
            )

            result = response.get("result", {})
            param_group = result.get("param_group")

            if not param_group:
                logger.error(
                    f"Server did not return a param_group "
                    f"for payload: {payload!r}, response: {response!r}"
                )
                raise RuntimeError(
                    "Failed to add param_group: server did not return a param_groups."
                )
            self.param_groups.append(_deserialize_output(param_group))
        except Exception as e:
            logger.error(f"Failed to add param group to the optimizer: {e}")
            raise

    def _extract_variable_ids_from_state(self):
        raise NotImplementedError

    # TODO: Implement `_on_optimizer_change` like done for `_on_variable_change`.
    #       This is useful for example to modify dynamically the `momentum` value:
    #       >>> for param_group in optimizer.param_groups:
    #       ...    param_group['momentum'] = new_lr


def _deserialize_state(
    state: list,
) -> DefaultDict[Variable, Any]:
    """
    Deserialize a state list (as produced by the server's _serialize_optimizer)
    into a DefaultDict mapping Variables to their state.

    Args:
        state (list): A list of dicts, each with "key" and "value" fields.

    Returns:
        DefaultDict[Variable, Any]: A DefaultDict mapping Variables to their state.
    """
    deserialized_state = defaultdict(dict)
    for item in state:
        key = _deserialize_output(item.get("key"))
        value = _deserialize_output(item.get("value"))
        deserialized_state[key] = value
    return deserialized_state


def _extract_variable_ids(obj):
    """
    Recursively extract variable_ids from a given object, which can be a dict or a list.
    If the object is a dict, it checks for keys "__variable__" or "__parameter__"
    and returns the value of "variable_id" if present. If the object is a list,
    it recursively extracts variable_ids from each element.
    If no variable_ids are found, it returns an empty list.
    """
    if isinstance(obj, dict):
        if (
            obj.get("__variable__") or obj.get("__parameter__")
        ) and "variable_id" in obj:
            return [obj["variable_id"]]
        return sum([_extract_variable_ids(v) for v in obj.values()], [])
    elif isinstance(obj, list):
        return sum([_extract_variable_ids(v) for v in obj], [])
    return []


def _wait_for_variable(variable_id: str, timeout: float = 3, interval: float = 0.01):
    """
    Wait until a variable with the given variable_id is registered in VARIABLE_REGISTRY,
    or raise TimeoutError after the specified timeout (in seconds).
    """
    end_time = time.monotonic() + timeout
    while time.monotonic() < end_time:
        if variable_id in VARIABLE_REGISTRY:
            return VARIABLE_REGISTRY[variable_id]
        time.sleep(0.01)
    raise TimeoutError(
        f"Variable with id {variable_id} not registered after {timeout} seconds"
    )
