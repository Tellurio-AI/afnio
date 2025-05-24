from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from afnio.models.model import BaseModel

# MODEL_REGISTRY is a mapping from model_id (str) to `afnio.models.model.BaseModel`
# instances.
# It is used to look up and update registered Model objects by their ID,
# typically when processing server-initiated updates.
MODEL_REGISTRY: Dict[str, "BaseModel"] = {}


def register_model(model: "BaseModel"):
    """
    Register an LM model in the local registry.

    Args:
        model (BaseModel): The model instance to register.
    """
    if model.model_id:
        MODEL_REGISTRY[model.model_id] = model


def get_model(model_id: str) -> Optional["BaseModel"]:
    """
    Retrieve an LM model instance from the registry by its model_id.

    Args:
        model_id (str): The unique identifier of the LM model.

    Returns:
        BaseModel: The model instance if found, else None.
    """
    return MODEL_REGISTRY.get(model_id)


def update_local_model_field(model_id: str, field: str, value):
    """
    Update a specific field of a registered Model instance as a consequence of a
    notification from the server.
    """
    model = get_model(model_id)
    if model is None:
        raise RuntimeError(f"Model with id '{model_id}' not found in registry.")
    setattr(model, field, value)
