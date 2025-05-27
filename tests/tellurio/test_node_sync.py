import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from afnio.tellurio._node_registry import NODE_REGISTRY, get_node
from afnio.tellurio.websocket_client import TellurioWebSocketClient


class TestServerToClientNodeSync:
    @pytest.fixture
    def mock_server_rpc(self):
        """
        Fixture to simulate receiving a server-initiated RPC call
        (create_node/create_edge) and to monitor the client's response.
        """

        def _mock(method, params):
            # Prepare the JSON-RPC message
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": "test-id-456",
                }
            )

            class FakeConnection:
                def __init__(self):
                    self.sent_messages = []
                    self._closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if hasattr(self, "_sent"):
                        self._closed = True
                        raise StopAsyncIteration
                    self._sent = True
                    return message

                async def send(self, msg):
                    self.sent_messages.append(msg)

                @property
                def closed(self):
                    return self._closed

            # Create a fresh client instance (no singleton)
            ws_client = TellurioWebSocketClient()
            ws_client.connection = FakeConnection()

            # Patch send to track calls
            with patch.object(
                ws_client.connection, "send", new_callable=AsyncMock
            ) as mock_send:
                loop = asyncio.get_event_loop()
                listener_task = loop.create_task(ws_client._listener())
                loop.run_until_complete(asyncio.sleep(0.1))
                listener_task.cancel()
                try:
                    loop.run_until_complete(listener_task)
                except asyncio.CancelledError:
                    pass

                # Return the mock so the test can assert on it
                return mock_send

        return _mock

    @staticmethod
    def assert_valid_rpc_response(send_mock):
        """
        Assert that the client sent a valid JSON-RPC response
        to an RPC call (create_node/create_edge) request.
        """
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id-456"
        assert response["result"]["message"] == "Ok"

    def test_create_node_registers_node(self, mock_server_rpc):
        """
        Test that creating a node via RPC registers it in the NODE_REGISTRY
        and sends a valid response back to the server.
        """
        # Clear registry for test isolation
        NODE_REGISTRY.clear()
        node_id = str(uuid.uuid4())
        params = {"node_id": node_id, "name": "AddBackward"}
        send_mock = mock_server_rpc("create_node", params)

        # Assert node is registered
        node = get_node(node_id)
        assert node is not None
        assert node.node_id == node_id
        assert node.name() == "AddBackward"

        # Assert correct response sent
        self.assert_valid_rpc_response(send_mock)

    def test_create_edge_registers_edge(self, mock_server_rpc):
        """
        Test that creating an edge via RPC registers it in the from_node's
        next_functions and sends a valid response back to the server.
        """
        # Clear registry for test isolation
        NODE_REGISTRY.clear()
        # Create two nodes first
        node_id_1 = str(uuid.uuid4())
        node_id_2 = str(uuid.uuid4())
        mock_server_rpc("create_node", {"node_id": node_id_1, "name": "AddBackward"})
        mock_server_rpc("create_node", {"node_id": node_id_2, "name": "AccumulateGrad"})

        # Create edge from node1 to node2 (node1.next_functions += (edge to node2,))
        params = {
            "from_node_id": node_id_1,
            "to_node_id": node_id_2,
            "output_nr": 0,
        }
        send_mock = mock_server_rpc("create_edge", params)

        # Assert edge is registered in node1.next_functions
        node1 = get_node(node_id_1)
        node2 = get_node(node_id_2)
        assert node1 is not None and node2 is not None
        assert len(node1.next_functions) == 1
        edge = node1.next_functions[0]
        assert edge.node is node2
        assert edge.output_nr == 0

        # Assert correct response sent
        self.assert_valid_rpc_response(send_mock)

    def test_create_edge_allows_to_node_id_none(self, mock_server_rpc):
        """
        Test that creating an edge with to_node_id=None is allowed
        (e.g., for leaf nodes).
        """
        NODE_REGISTRY.clear()
        node_id_1 = str(uuid.uuid4())
        # Register only from_node
        mock_server_rpc("create_node", {"node_id": node_id_1, "name": "AddBackward"})

        params = {
            "from_node_id": node_id_1,
            "to_node_id": None,
            "output_nr": 0,
        }
        send_mock = mock_server_rpc("create_edge", params)

        # Assert edge is registered in node1.next_functions
        node1 = get_node(node_id_1)
        assert node1 is not None
        assert len(node1.next_functions) == 1
        edge = node1.next_functions[0]
        assert edge.node is None
        assert edge.output_nr == 0

        # Assert correct response sent
        self.assert_valid_rpc_response(send_mock)

    def test_create_edge_raises_if_from_node_missing(self, mock_server_rpc):
        """
        Test that creating an edge fails with an error
        if from_node is not in the registry.
        """
        NODE_REGISTRY.clear()
        node_id_1 = "not-in-registry"
        node_id_2 = str(uuid.uuid4())
        # Only register to_node
        mock_server_rpc("create_node", {"node_id": node_id_2, "name": "AccumulateGrad"})

        params = {
            "from_node_id": node_id_1,
            "to_node_id": node_id_2,
            "output_nr": 0,
        }
        send_mock = mock_server_rpc("create_edge", params)
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert "error" in response
        assert response["error"]["message"] == "Internal error"
        assert response["error"]["data"]["exception"] == (
            "from_node with id 'not-in-registry' not found in registry."
        )

    def test_create_edge_raises_if_to_node_missing(self, mock_server_rpc):
        """
        Test that creating an edge fails with an error
        if to_node is not in the registry.
        """
        NODE_REGISTRY.clear()
        node_id_1 = str(uuid.uuid4())
        node_id_2 = "not-in-registry"
        # Only register from_node
        mock_server_rpc("create_node", {"node_id": node_id_1, "name": "AddBackward"})

        params = {
            "from_node_id": node_id_1,
            "to_node_id": node_id_2,
            "output_nr": 0,
        }
        send_mock = mock_server_rpc("create_edge", params)
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert "error" in response
        assert response["error"]["message"] == "Internal error"
        assert response["error"]["data"]["exception"] == (
            "to_node with id 'not-in-registry' not found in registry."
        )
