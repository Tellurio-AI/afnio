from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from afnio.autodiff.graph import GradientEdge, Node

# Registry mapping node_id to Node instance
NODE_REGISTRY: Dict[str, "Node"] = {}


def register_node(node: "Node"):
    """
    Register a Node instance in the local registry.

    Args:
        node (Node): The Node instance to register.
    """
    if node.node_id:
        NODE_REGISTRY[node.node_id] = node


def get_node(node_id: str) -> Optional["Node"]:
    """
    Retrieve a Node instance from the registry by its node_id.

    Args:
        node_id (str): The unique identifier of the Node.

    Returns:
        Node or None: The Node instance if found, else None.
    """
    return NODE_REGISTRY.get(node_id)


def create_node(data: dict) -> "Node":
    """
    Create and register a Node from serialized data received from the server.

    Args:
        data (dict): Serialized node data with keys 'name' and 'node_id'.

    Returns:
        Node: The created and registered Node instance.
    """
    from afnio._variable import _allow_grad_fn_assignment
    from afnio.autodiff.graph import Node
    from afnio.tellurio._variable_registry import (
        PENDING_GRAD_FN_ASSIGNMENTS,
        suppress_variable_notifications,
    )

    node = Node()
    node._name = data["name"]
    node.node_id = data["node_id"]
    register_node(node)

    # After registering, resolve any pending grad_fn assignments
    pending_vars = PENDING_GRAD_FN_ASSIGNMENTS.pop(node.node_id, [])
    for var in pending_vars:
        with suppress_variable_notifications(), _allow_grad_fn_assignment():
            var._pending_grad_fn_id = None
            var.grad_fn = node

    return node


def create_and_append_edge(data: dict) -> "GradientEdge":
    """
    Create a GradientEdge from serialized data
    and append it to from_node.next_functions.

    Note:
        The edge is appended to from_node.next_functions and points to to_node.
        This follows the backward pass convention.

    Args:
        data (dict): Serialized edge data with keys 'from_node_id', 'to_node_id',
          and 'output_nr'.

    Returns:
        GradientEdge: The created GradientEdge instance.
    """
    from afnio.autodiff.graph import GradientEdge

    from_node_id = data["from_node_id"]
    to_node_id = data["to_node_id"]
    from_node = get_node(from_node_id)
    to_node = get_node(to_node_id)
    output_nr = data["output_nr"]

    if not from_node:
        raise ValueError(f"from_node with id '{from_node_id}' not found in registry.")
    if to_node_id is not None and to_node is None:
        raise ValueError(f"to_node with id '{to_node_id}' not found in registry.")
    # If to_node_id=None, to_node=None (leaf Variable that doesn't require gradient)

    edge = GradientEdge(node=to_node, output_nr=output_nr)
    from_node.next_functions = from_node.next_functions + (edge,)
    return edge
