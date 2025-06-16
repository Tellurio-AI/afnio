from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from afnio.optim.optimizer import Optimizer

# Registry mapping optimizer_id to Optimizer instance
OPTIMIZER_REGISTRY: Dict[str, "Optimizer"] = {}


def register_optimizer(optimizer: "Optimizer"):
    """
    Register an Optimizer instance in the registry.

    Args:
        optimizer (Optimizer): The optimizer instance to register.
    """
    if optimizer.optimizer_id:
        OPTIMIZER_REGISTRY[optimizer.optimizer_id] = optimizer


def get_optimizer(optimizer_id: str) -> Optional["Optimizer"]:
    """
    Retrieve an Optimizer instance from the registry by its optimizer_id.

    Args:
        optimizer_id (str): The unique identifier of the Optimizer.

    Returns:
        Optimizer or None: The optimizer instance if found, else None.
    """
    return OPTIMIZER_REGISTRY.get(optimizer_id)
