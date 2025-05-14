import os
import tempfile
from typing import List

import pytest
import pytest_asyncio

from afnio.tellurio import client as tellurio_client_module
from afnio.tellurio.client import TellurioClient, get_default_client
from afnio.tellurio.project import Project, create_project, delete_project
from afnio.tellurio.websocket_client import TellurioWebSocketClient

TEST_ORG_DISPLAY_NAME = os.getenv("TEST_ORG_DISPLAY_NAME", "Tellurio Test")
TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")


@pytest.fixture(autouse=True)
def patch_config_path(monkeypatch):
    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        dummy_config_path = tmp.name
    # Patch the CONFIG_PATH in the tellurio_client_module
    monkeypatch.setattr(tellurio_client_module, "CONFIG_PATH", dummy_config_path)
    yield
    # Clean up the file after the test
    if os.path.exists(dummy_config_path):
        os.remove(dummy_config_path)


@pytest.fixture(scope="module")
def client():
    """
    Fixture to provide a real TellurioClient instance.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    client = TellurioClient()
    client.login(api_key=api_key)  # Replace with a valid API key
    return client


@pytest.fixture
def create_and_delete_project(client):
    """
    Fixture to create a project before a test and delete it after the test.
    """
    # Create the project
    project = create_project(
        namespace_slug=TEST_ORG_SLUG,
        display_name=TEST_PROJECT,
        visibility="TEAM",
        client=client,
    )

    # Track whether the project has already been deleted
    project_deleted = False

    def mark_deleted():
        nonlocal project_deleted
        project_deleted = True

    yield project, mark_deleted

    # Delete the project only if it hasn't already been deleted
    if not project_deleted:
        delete_project(
            namespace_slug=TEST_ORG_SLUG,
            project_slug=project.slug,
            client=client,
        )


@pytest.fixture
def delete_project_fixture(client):
    """
    Fixture to delete a project after a test.
    """
    projects_to_delete: List[Project] = []

    yield projects_to_delete  # Provide a list to the test to track projects to delete

    # Delete all projects in the list after the test
    for project in projects_to_delete:
        delete_project(
            namespace_slug=TEST_ORG_SLUG,
            project_slug=project.slug,
            client=client,
        )


@pytest_asyncio.fixture
async def connected_ws_client():
    """
    Fixture to create and connect a TellurioWebSocketClient instance for testing.
    Ensures the connection is properly closed after the test.
    """
    client = TellurioWebSocketClient(
        base_url=os.getenv(
            "TELLURIO_BACKEND_WS_BASE_URL", "wss://platform.tellurio.ai"
        ),
        port=int(os.getenv("TELLURIO_BACKEND_WS_PORT", 443)),
        default_timeout=10,
    )
    await client.connect(api_key=os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key"))
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def close_ws_client():
    """
    Fixture to ensure the WebSocket client is closed after each test that uses it.
    """
    _, ws_client = get_default_client()
    assert ws_client is not None

    # Provide the WebSocket client to the test
    yield ws_client

    # Cleanup: Close the WebSocket client after the test
    await ws_client.close()
