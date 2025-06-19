import os
import tempfile
from typing import List

import pytest
from slugify import slugify

from afnio.tellurio import utils as tellurio_utils
from afnio.tellurio._eventloop import _event_loop_thread
from afnio.tellurio.client import TellurioClient, get_default_client
from afnio.tellurio.project import Project, create_project, delete_project, get_project

TEST_ORG_DISPLAY_NAME = os.getenv("TEST_ORG_DISPLAY_NAME", "Tellurio Test")
TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")


@pytest.fixture(autouse=True)
def patch_config_path(monkeypatch):
    """
    Fixture to patch the CONFIG_PATH in the tellurio_client_module
    to use a temporary file for testing.
    This prevents the need for a real config file during tests.
    """
    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        dummy_config_path = tmp.name

    # Patch get_config_path to always return the dummy path
    monkeypatch.setattr(tellurio_utils, "get_config_path", lambda: dummy_config_path)
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


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_project(client):
    """
    Fixture to ensure TEST_PROJECT does not exist before each module.
    If it exists, delete it.
    """
    project_slug = slugify(TEST_PROJECT)
    try:
        project = get_project(
            namespace_slug=TEST_ORG_SLUG,
            project_slug=project_slug,
            client=client,
        )
        delete_project(
            namespace_slug=TEST_ORG_SLUG,
            project_slug=project.slug,
            client=client,
        )
    except Exception:
        # Project does not exist or could not be fetched; ignore
        pass
    yield


@pytest.fixture(scope="session", autouse=True)
def shutdown_event_loop_thread():
    """
    Fixture to shut down the event loop thread after all tests in the module.
    This ensures that the WebSocket client is properly closed and the event loop
    thread is cleaned up.
    """
    yield

    # Close the WebSocket client if it exists
    try:
        _, ws_client = get_default_client()
        if ws_client and ws_client.connection:
            # Use the event loop thread to close the client
            _event_loop_thread.run(ws_client.close())
    except Exception:
        pass

    # Now shut down the event loop thread
    _event_loop_thread.shutdown()
