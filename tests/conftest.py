import os
from typing import List

import pytest

from afnio.tellurio.client import TellurioClient
from afnio.tellurio.project import Project, create_project, delete_project

TEST_ORG_DISPLAY_NAME = os.getenv("TEST_ORG_DISPLAY_NAME", "Tellurio Test")
TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")


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
