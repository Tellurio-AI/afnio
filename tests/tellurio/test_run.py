import os
from typing import List

import httpx
import pytest
from slugify import slugify

from afnio.tellurio.project import Project, get_project
from afnio.tellurio.run import Run, RunStatus, init

TEST_USER_USERNAME = os.getenv("TEST_USER_USERNAME", "testuser")
TEST_USER_SLUG = os.getenv("TEST_USER_SLUG", "testuser")
TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")

NON_EXISTING_PROJECT = "Non Existing Project"


def test_init_run_in_existing_project(client, create_and_delete_project):
    """
    Test the tellurio.init() function with valid inputs.
    """
    project, _ = create_and_delete_project

    # Create a run within the project
    run = init(
        namespace_slug=TEST_ORG_SLUG,
        project_display_name=TEST_PROJECT,
        name="MyRun",
        description="This is a test run",
        status=RunStatus.RUNNING,
        client=client,
    )

    assert isinstance(run, Run)
    assert run.uuid is not None
    assert run.name == "MyRun"
    assert run.description == "This is a test run"
    assert run.status == RunStatus.RUNNING
    assert run.date_created is not None
    assert run.date_updated is not None
    assert run.project.uuid == project.uuid
    assert run.project.display_name == project.display_name
    assert run.project.slug == project.slug
    assert run.user.uuid is not None
    assert run.user.username == TEST_USER_USERNAME
    assert run.user.slug == TEST_USER_SLUG


def test_init_run_in_non_existing_project(
    client, delete_project_fixture: List[Project]
):
    """
    Test creating a new run in a non-existing project.
    """
    # Create a run in a non-existing project
    run = init(
        namespace_slug=TEST_ORG_SLUG,
        project_display_name=NON_EXISTING_PROJECT,
        name="MyRun",
        description="This is a test run in a non-existing project",
        status=RunStatus.RUNNING,
        client=client,
    )

    # Check if the project was created
    project = get_project(
        namespace_slug=TEST_ORG_SLUG,
        project_slug=slugify(NON_EXISTING_PROJECT),
        client=client,
    )
    assert isinstance(project, Project)
    assert project.display_name == NON_EXISTING_PROJECT
    assert project.slug == slugify(NON_EXISTING_PROJECT)
    assert project.visibility == "RESTRICTED"

    # Add the newly created project to the delete list to ensure cleanup
    delete_project_fixture.append(project)

    # Check if the run was created
    assert isinstance(run, Run)
    assert run.uuid is not None
    assert run.name == "MyRun"
    assert run.description == "This is a test run in a non-existing project"
    assert run.status == RunStatus.RUNNING
    assert run.date_created is not None
    assert run.date_updated is not None
    assert run.project.uuid == project.uuid
    assert run.project.display_name == project.display_name
    assert run.project.slug == project.slug
    assert run.user.uuid is not None
    assert run.user.username == TEST_USER_USERNAME
    assert run.user.slug == TEST_USER_SLUG


@pytest.mark.parametrize(
    "namespace_slug, name, description, status, should_succeed",
    [
        (TEST_ORG_SLUG, None, "This is a test run", RunStatus.RUNNING, True),
        ("invalid-org", None, "This is a test run", RunStatus.RUNNING, False),
        (TEST_ORG_SLUG, "MyRun", None, RunStatus.RUNNING, True),
        (TEST_ORG_SLUG, "MyRun", "This is a test run", None, True),
        (TEST_ORG_SLUG, None, None, None, True),
    ],
)
def test_init_run_with_various_inputs(
    client,
    create_and_delete_project,
    namespace_slug,
    name,
    description,
    status,
    should_succeed,
):
    """
    Test the init() function with both valid and invalid inputs.
    """
    project, _ = create_and_delete_project

    if should_succeed:
        run = init(
            namespace_slug=namespace_slug,
            project_display_name=project.display_name,
            name=name,
            description=description,
            status=status,
            client=client,
        )

        # Assertions for success
        assert isinstance(run, Run)
        assert run.uuid is not None
        if name is not None:
            assert run.name == name
        if description is not None:
            assert run.description == description
        if status is not None:
            assert run.status == status
        assert run.date_created is not None
        assert run.date_updated is not None
        assert run.project.uuid == project.uuid
        assert run.project.display_name == project.display_name
        assert run.project.slug == project.slug
        assert run.user.uuid is not None
        assert run.user.username == TEST_USER_USERNAME
        assert run.user.slug == TEST_USER_SLUG
    else:
        with pytest.raises(httpx.HTTPStatusError):
            init(
                namespace_slug=namespace_slug,
                project_display_name=project.display_name,
                name=name,
                description=description,
                status=status,
                client=client,
            )
