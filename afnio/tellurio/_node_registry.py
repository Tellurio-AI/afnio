from typing import Dict, Optional

from afnio.autodiff.graph import GradientEdge, Node

# Registry mapping node_id to Node instance
NODE_REGISTRY: Dict[str, Node] = {}


def register_node(node: Node):
    """
    Register a Node instance in the local registry.

    Args:
        node (Node): The Node instance to register.
    """
    if node.node_id:
        NODE_REGISTRY[node.node_id] = node


def get_node(node_id: str) -> Optional[Node]:
    """Retrieve a Node instance by its node_id."""

    """
    Retrieve a Node instance from the registry by its node_id.

    Args:
        node_id (str): The unique identifier of the Node.

    Returns:
        Node or None: The Node instance if found, else None.
    """
    return NODE_REGISTRY.get(node_id)


def create_node(data: dict) -> Node:
    """
    Create and register a Node from serialized data received from the server.

    Args:
        data (dict): Serialized node data with keys 'name' and 'node_id'.

    Returns:
        Node: The created and registered Node instance.
    """
    node = Node()
    node._name = data["name"]
    node.node_id = data["node_id"]
    register_node(node)
    return node


def create_and_append_edge(data: dict) -> GradientEdge:
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
    from_node_id = data["from_node_id"]
    to_node_id = data["to_node_id"]
    from_node = get_node(from_node_id)
    to_node = get_node(to_node_id)
    output_nr = data["output_nr"]

    if not from_node:
        raise ValueError(f"from_node with id '{from_node_id}' not found in registry.")
    if not to_node:
        raise ValueError(f"to_node with id '{to_node_id}' not found in registry.")

    edge = GradientEdge(node=to_node, output_nr=output_nr)
    from_node.next_functions = from_node.next_functions + (edge,)
    return edge
