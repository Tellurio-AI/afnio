import datetime
import os
from typing import List

import httpx
import pytest
from slugify import slugify

import afnio.tellurio.client as client_mod
from afnio.tellurio import run as run_mod
from afnio.tellurio.project import Project, get_project
from afnio.tellurio.run import Run, RunStatus, init
from afnio.tellurio.run_context import get_active_run

TEST_USER_USERNAME = os.getenv("TEST_USER_USERNAME", "testuser")
TEST_USER_SLUG = os.getenv("TEST_USER_SLUG", "testuser")
TEST_ORG_SLUG = os.getenv("TEST_ORG_SLUG", "tellurio-test")
TEST_PROJECT = os.getenv("TEST_PROJECT", "Test Project")

NON_EXISTING_PROJECT = "Non Existing Project"


def get_run_status_from_server(client, namespace_slug, project_slug, run_uuid):
    """
    Helper function to get the status of a run from the server.
    Raises an AssertionError if the run is not found.
    """
    endpoint = f"/api/v0/{namespace_slug}/projects/{project_slug}/runs/"
    response = client.get(endpoint)
    assert response.status_code == 200
    runs = response.json()
    for run in runs:
        if run["uuid"] == run_uuid:
            return run["status"]
    raise AssertionError(f"Run {run_uuid} not found on server.")


def get_run_metrics_from_server(client, namespace_slug, project_slug, run_uuid):
    """
    Helper function to get the metrics of a run from the server.
    Returns a list of metrics.
    """
    endpoint = (
        f"/api/v0/{namespace_slug}/projects/{project_slug}/runs/{run_uuid}/metrics/"
    )
    response = client.get(endpoint)
    assert response.status_code == 200
    return response.json()


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


def test_run_finish_sets_status_and_clears_active_uuid(
    client, create_and_delete_project
):
    """
    Test that Run.finish() sets the run status to COMPLETED on the server
    and clears the active run UUID.
    """
    project, _ = create_and_delete_project

    # Create a run
    run = init(
        namespace_slug=TEST_ORG_SLUG,
        project_display_name=project.display_name,
        name="FinishTestRun",
        description="Testing finish()",
        status=RunStatus.RUNNING,
        client=client,
    )

    # Ensure the run is active and status is RUNNING
    assert run.status == RunStatus.RUNNING
    assert get_active_run() == run

    # Call finish and capture PATCH response
    # Patch the Run.finish method to return the response for testing
    original_patch = client.patch
    patch_response = {}

    def patch_and_capture(endpoint, json):
        resp = original_patch(endpoint, json)
        patch_response["response"] = resp
        return resp

    client.patch = patch_and_capture
    run.finish(client=client)
    client.patch = original_patch  # Restore

    # The status should be COMPLETED
    assert run.status == RunStatus.COMPLETED

    # The active run UUID should be cleared
    with pytest.raises(ValueError):
        get_active_run()

    # Check PATCH response body
    response = patch_response["response"]
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == RunStatus.COMPLETED.value

    # Check date_updated > date_created
    date_created = datetime.datetime.fromisoformat(
        data["date_created"].replace("Z", "+00:00")
    )
    date_updated = datetime.datetime.fromisoformat(
        data["date_updated"].replace("Z", "+00:00")
    )
    assert date_updated >= date_created


def test_run_finish_idempotent(client, create_and_delete_project):
    """
    Test that calling finish() multiple times does not raise
    and always sets status to COMPLETED.
    """
    project, _ = create_and_delete_project

    run = init(
        namespace_slug=TEST_ORG_SLUG,
        project_display_name=project.display_name,
        name="FinishIdempotentRun",
        description="Testing finish() idempotency",
        status=RunStatus.RUNNING,
        client=client,
    )

    run.finish(client=client)
    # Call finish again; should not raise
    run.finish(client=client)
    assert run.status == RunStatus.COMPLETED


def test_run_auto_finish_on_exit(client):
    """
    Test that an unfinished run is automatically finished as COMPLETED at exit.
    """
    # Override the global default client for all code (including atexit)
    client_mod._default_client = client

    namespace_slug = TEST_ORG_SLUG
    project_display_name = TEST_PROJECT
    project_slug = slugify(project_display_name)

    run = init(
        namespace_slug,
        project_display_name,
        name="AutoFinishTest",
        client=client,
    )
    run_uuid = run.uuid

    # Simulate script exit
    run_mod._finish_active_run_on_exit()

    # Check the run status from the server and the local run object
    status = get_run_status_from_server(client, namespace_slug, project_slug, run_uuid)
    assert status == RunStatus.COMPLETED.value
    assert run.status.value == RunStatus.COMPLETED.value


def test_run_auto_fail_previous_on_new_init(client):
    """
    Test that starting a new run auto-finishes the previous unfinished run as FAILED.
    """
    # Override the global default client for all code (including atexit)
    client_mod._default_client = client

    namespace_slug = TEST_ORG_SLUG
    project_display_name = TEST_PROJECT
    project_slug = slugify(project_display_name)

    run1 = init(
        namespace_slug, project_display_name, name="AutoFailTest1", client=client
    )
    run1_uuid = run1.uuid

    run2 = init(
        namespace_slug, project_display_name, name="AutoFailTest2", client=client
    )
    run2_uuid = run2.uuid

    # Check the run status from the server and the local run object
    status1 = get_run_status_from_server(
        client, namespace_slug, project_slug, run1_uuid
    )
    status2 = get_run_status_from_server(
        client, namespace_slug, project_slug, run2_uuid
    )
    assert status1 == RunStatus.FAILED.value
    # run2 is still active, so its status should be RUNNING
    assert status2 == RunStatus.RUNNING.value
    assert run1.status.value == RunStatus.FAILED.value
    assert run2.status.value == RunStatus.RUNNING.value


def test_run_context_manager_completed_and_failed(client):
    """
    Test that a run used as a context manager is finished as COMPLETED if no error,
    and as FAILED if an exception occurs.
    """
    # Override the global default client for all code (including atexit)
    client_mod._default_client = client

    namespace_slug = "tellurio-test"
    project_display_name = "Test Project"
    project_slug = slugify(project_display_name)

    # COMPLETED case
    with init(
        namespace_slug, project_display_name, name="ContextCompletedTest", client=client
    ) as run:
        run_uuid = run.uuid

    # Check the run status from the server and the local run object
    status = get_run_status_from_server(client, namespace_slug, project_slug, run_uuid)
    assert status == RunStatus.COMPLETED.value
    assert run.status.value == RunStatus.COMPLETED.value

    # FAILED case
    try:
        with init(
            namespace_slug,
            project_display_name,
            name="ContextFailedTest",
            client=client,
        ) as run:
            run_uuid = run.uuid
            raise RuntimeError("Test crash")
    except RuntimeError:
        pass

    # Check the run status from the server and the local run object
    status = get_run_status_from_server(client, namespace_slug, project_slug, run_uuid)
    assert status == RunStatus.CRASHED.value
    assert run.status.value == RunStatus.CRASHED.value


def test_run_log_metric(client):
    """
    Test that Run.log() correctly logs a metric and it appears on the server.
    """
    namespace_slug = "tellurio-test"
    project_display_name = "Test Project"
    project_slug = slugify(project_display_name)

    run = init(
        namespace_slug=namespace_slug,
        project_display_name=project_display_name,
        name="LogMetricRun",
        description="Testing log()",
        status=RunStatus.RUNNING,
        client=client,
    )

    # Log a metric
    run.log(name="accuracy", value=0.95, client=client)

    # Fetch metrics from the server and check
    metrics = get_run_metrics_from_server(client, TEST_ORG_SLUG, project_slug, run.uuid)
    found = any(
        m["name"] == "accuracy" and float(m["value"]) == 0.95 and m["step"] == 0
        for m in metrics
    )
    assert found, "Logged metric 'accuracy' with value 0.95 not found on server."


def test_run_log_metric_with_step(client):
    """
    Test that Run.log() correctly logs a metric with a step
    and it appears on the server.
    """
    namespace_slug = "tellurio-test"
    project_display_name = "Test Project"
    project_slug = slugify(project_display_name)

    run = init(
        namespace_slug=namespace_slug,
        project_display_name=project_display_name,
        name="LogMetricStepRun",
        description="Testing log() with step",
        status=RunStatus.RUNNING,
        client=client,
    )

    # Log a metric with a step
    run.log(name="loss", value=0.123, step=5, client=client)

    # Fetch metrics from the server and check
    metrics = get_run_metrics_from_server(client, TEST_ORG_SLUG, project_slug, run.uuid)
    found = any(
        m["name"] == "loss" and float(m["value"]) == 0.123 and m.get("step") == 5
        for m in metrics
    )
    assert (
        found
    ), "Logged metric 'loss' with value 0.123 and step 5 not found on server."


def test_run_log_multiple_metrics(client):
    """
    Test that logging multiple metrics works and all appear on the server.
    """
    namespace_slug = "tellurio-test"
    project_display_name = "Test Project"
    project_slug = slugify(project_display_name)

    run = init(
        namespace_slug=namespace_slug,
        project_display_name=project_display_name,
        name="LogMultipleMetricsRun",
        description="Testing multiple log() calls",
        status=RunStatus.RUNNING,
        client=client,
    )

    metrics_to_log = [
        ("accuracy", 0.8, None),
        ("loss", 0.2, 1),
        ("precision", 0.75, 2),
    ]
    for name, value, step in metrics_to_log:
        run.log(name=name, value=value, step=step, client=client)

    # Fetch metrics from the server and check all are present
    metrics = get_run_metrics_from_server(client, TEST_ORG_SLUG, project_slug, run.uuid)
    for name, value, step in metrics_to_log:
        found = any(
            m["name"] == name
            and float(m["value"]) == value
            and (step is None or m.get("step") == step)
            for m in metrics
        )
        assert (
            found
        ), f"Logged metric '{name}' with value {value} and step {step} not found on server."  # noqa: E501
