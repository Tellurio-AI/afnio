import os
from typing import List

import httpx
import pytest
from slugify import slugify

from afnio.tellurio.project import Project, create_project, delete_project, get_project

TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")


@pytest.mark.parametrize(
    "namespace_slug, display_name, visibility, should_succeed",
    [
        (TEST_ORG_SLUG, TEST_PROJECT, "TEAM", True),
        ("invalid-org", TEST_PROJECT, "TEAM", False),
        (TEST_ORG_SLUG, TEST_PROJECT, "INVALID_VISIBILITY", False),
    ],
)
def test_create_project_with_various_inputs(
    client,
    delete_project_fixture: List[Project],
    namespace_slug,
    display_name,
    visibility,
    should_succeed,
):
    """
    Test the create_project function with both valid and invalid inputs.
    """
    if should_succeed:
        project = create_project(
            namespace_slug=namespace_slug,
            display_name=display_name,
            visibility=visibility,
            client=client,
        )

        # Add the newly created project to the delete list to ensure cleanup
        delete_project_fixture.append(project)

        # Assertions for success
        assert isinstance(project, Project)
        assert project.display_name == display_name
        assert project.slug == slugify(display_name)
        assert project.visibility == visibility
        assert project.role == "OWNER"
    else:
        with pytest.raises(httpx.HTTPStatusError):
            create_project(
                namespace_slug=namespace_slug,
                display_name=display_name,
                visibility=visibility,
                client=client,
            )


@pytest.mark.parametrize(
    "namespace_slug, project_slug, should_succeed",
    [
        (TEST_ORG_SLUG, TEST_PROJECT, True),
        (TEST_ORG_SLUG, "non-existing-project", False),
        ("invalid-org", TEST_PROJECT, False),
    ],
)
def test_get_project_with_various_inputs(
    client, create_and_delete_project, namespace_slug, project_slug, should_succeed
):
    """
    Test the get_project function with both valid and invalid inputs.
    """
    project, _ = create_and_delete_project

    if should_succeed:
        project = get_project(
            namespace_slug=namespace_slug,
            project_slug=project.slug,
            client=client,
        )

        # Assertions for success
        assert isinstance(project, Project)
        assert project.uuid is not None
        assert project.display_name == TEST_PROJECT
        assert project.slug == slugify(TEST_PROJECT)
        assert project.date_created is not None
        assert project.last_run_date is None
        assert project.visibility == "TEAM"
        assert project.role is None
        assert project.org_uuid is None
        assert project.org_display_name is None
        assert project.org_slug is None
    else:
        project = get_project(
            namespace_slug=namespace_slug,
            project_slug=slugify(project_slug),
            client=client,
        )
        assert project is None


@pytest.mark.parametrize(
    "namespace_slug, project_slug, should_succeed",
    [
        (TEST_ORG_SLUG, TEST_PROJECT, True),
        (TEST_ORG_SLUG, "non-existing-project", False),
        ("invalid-org", TEST_PROJECT, False),
    ],
)
def test_delete_project_with_various_inputs(
    client, create_and_delete_project, namespace_slug, project_slug, should_succeed
):
    """
    Test the delete_project function with both valid and invalid inputs.
    """
    project, mark_deleted = create_and_delete_project

    if should_succeed:
        delete_project(
            namespace_slug=namespace_slug,
            project_slug=project.slug,
            client=client,
        )

        # Mark the project as deleted
        mark_deleted()

        # Verify the project no longer exists
        project = get_project(
            namespace_slug=namespace_slug,
            project_slug=project.slug,
            client=client,
        )
        assert project is None
    else:
        with pytest.raises(httpx.HTTPStatusError):
            delete_project(
                namespace_slug=namespace_slug,
                project_slug=slugify(project_slug),
                client=client,
            )
