import atexit
import logging
import os
import sys
import threading
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from slugify import slugify

from afnio.tellurio.client import TellurioClient, get_default_client
from afnio.tellurio.project import create_project, get_project
from afnio.tellurio.run_context import set_active_run

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """
    Represents the status of a Tellurio Run.
    """

    RUNNING = "RUNNING"
    CRASHED = "CRASHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RunOrg:
    """Represents a Tellurio Run Organization."""

    def __init__(self, slug: str):
        self.slug = slug

    def __repr__(self):
        return f"<Organization slug={self.slug}>"


class RunProject:
    """
    Represents a Tellurio Run Project.
    """

    def __init__(self, uuid: str, display_name: str, slug: str):
        self.uuid = uuid
        self.display_name = display_name
        self.slug = slug

    def __repr__(self):
        return f"<Project uuid={self.uuid} display_name={self.display_name}>"


class RunUser:
    """
    Represents a Tellurio Run User.
    """

    def __init__(self, uuid: str, username: str, slug: str):
        self.uuid = uuid
        self.username = username
        self.slug = slug

    def __repr__(self):
        return f"<User uuid={self.uuid} username={self.username}>"


class Run:
    """
    Represents a Tellurio Run.
    """

    def __init__(
        self,
        uuid: str,
        name: str,
        description: str,
        status: RunStatus,
        date_created: Optional[datetime] = None,
        date_updated: Optional[datetime] = None,
        organization: Optional[RunOrg] = None,
        project: Optional[RunProject] = None,
        user: Optional[RunUser] = None,
    ):
        self.uuid = uuid
        self.name = name
        self.description = description
        self.status = RunStatus(status)
        self.date_created = date_created
        self.date_updated = date_updated
        self.organization = organization
        self.project = project
        self.user = user

    def __repr__(self):
        return (
            f"<Run uuid={self.uuid} name={self.name} status={self.status} "
            f"project={self.project.display_name if self.project else None}>"
        )

    def finish(
        self,
        client: Optional[TellurioClient] = None,
        status: Optional[RunStatus] = RunStatus.COMPLETED,
    ):
        """
        Marks the run as COMPLETED on the server by sending a PATCH request,
        and clears the active run UUID.

        Args:
            client (TellurioClient, optional): The client to use for the request.
                If not provided, the default client will be used.

        Raises:
            Exception: If the PATCH request fails.
        """
        client = client or get_default_client()[0]

        namespace_slug = self.organization.slug if self.organization else None
        project_slug = self.project.slug if self.project else None
        run_uuid = self.uuid

        if not (namespace_slug and project_slug and run_uuid):
            raise ValueError("Run object is missing required identifiers.")

        endpoint = f"/api/v0/{namespace_slug}/projects/{project_slug}/runs/{run_uuid}/"
        payload = {"status": status.value}

        try:
            response = client.patch(endpoint, json=payload)
            if response.status_code == 200:
                self.status = status
                if not _IN_ATEXIT:
                    logger.info(f"Run {self.name!r} marked as COMPLETED.")
            else:
                if not _IN_ATEXIT:
                    logger.error(
                        f"Failed to update run status: {response.status_code} - {response.text}"  # noqa: E501
                    )
                response.raise_for_status()
        except Exception as e:
            if not _IN_ATEXIT:
                logger.error(f"An error occurred while updating the run status: {e}")
            raise

        # Clear the active run UUID after finishing
        try:
            set_active_run(None)
        except Exception:
            pass

        # Mark safeguard as finished
        _unregister_safeguard(self)

    def log(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        client: Optional[TellurioClient] = None,
    ):
        """
        Log a metric for this run.

        Args:
            name (str): Name of the metric.
            value (Any): Value of the metric. Can be any type that is JSON serializable.
            step (int, optional): Step number. If not provided, the backend will
                auto-compute it.
            client (TellurioClient, optional): The client to use for the request.
        """
        client = client or get_default_client()[0]

        namespace_slug = self.organization.slug if self.organization else None
        project_slug = self.project.slug if self.project else None
        run_uuid = self.uuid

        if not (namespace_slug and project_slug and run_uuid):
            raise ValueError("Run object is missing required identifiers.")

        endpoint = (
            f"/api/v0/{namespace_slug}/projects/{project_slug}/runs/{run_uuid}/metrics/"
        )
        payload = {
            "name": name,
            "value": value,
        }
        if step is not None:
            payload["step"] = step

        try:
            response = client.post(endpoint, json=payload)
            if response.status_code == 201:
                logger.info(f"Logged metric '{name}'={value} for run '{self.name}'.")
            else:
                logger.error(
                    f"Failed to log metric: {response.status_code} - {response.text}"
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"An error occurred while logging the metric: {e}")
            raise

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        Returns self so it can be used as a context manager.
        """
        _register_safeguard(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and finish the run.
        If an exception occurred, you may want to set status to CRASHED in the future.
        """
        status = RunStatus.CRASHED if exc_type else RunStatus.COMPLETED
        self.finish(status=status)


def init(
    namespace_slug: str,
    project_display_name: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[RunStatus] = RunStatus.RUNNING,
    client: Optional[TellurioClient] = None,
) -> Run:
    """
    Initializes a new Tellurio Run.

    Args:
        namespace_slug (str): The namespace slug where the project resides. It can be
          either an organization slug or a user slug.
        project_display_name (str): The display name of the project. This will be used
            to retrive or create the project through its slugified version.
        name (str, optional): The name of the run. If not provided, a random name is
            generated (e.g., "brave_pasta_123"). If the name is provided but already
            exists, an incremental number is appended to the name (e.g., "test_run_1",
            "test_run_2").
        description (str, optional): A description of the run.
        status (str): The status of the run (default: "RUNNING").
        client (TellurioClient, optional): An instance of TellurioClient. If not
          provided, the default client will be used.

    Returns:
        Run: A Run object representing the created run.
    """
    client = client or get_default_client()[0]

    # Generate the project's slug from its name
    project_slug = slugify(project_display_name)

    # Ensure the project exists
    try:
        project_obj = get_project(
            namespace_slug=namespace_slug,
            project_slug=project_slug,
            client=client,
        )
        if project_obj is not None:
            logger.info(
                f"Project with slug {project_slug!r} already exists "
                f"in namespace {namespace_slug!r}."
            )
        else:
            logger.info(
                f"Project with slug {project_slug!r} does not exist "
                f"in namespace {namespace_slug!r}. "
                f"Creating it now with RESTRICTED visibility."
            )
            project_obj = create_project(
                namespace_slug=namespace_slug,
                display_name=project_display_name,
                visibility="RESTRICTED",
                client=client,
            )
    except Exception as e:
        logger.error(f"An error occurred while retrieving or creating the project: {e}")
        raise

    # Dynamically construct the payload to exclude None values
    payload = {}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if status is not None:
        payload["status"] = status.value

    # Create the run
    endpoint = f"/api/v0/{namespace_slug}/projects/{project_slug}/runs/"

    try:
        response = client.post(endpoint, json=payload)

        if response.status_code == 201:
            data = response.json()
            base_url = os.getenv(
                "TELLURIO_BACKEND_HTTP_BASE_URL", "https://platform.tellurio.ai"
            )
            run_slug = slugify(data["name"])
            logger.info(
                f"Run {data['name']!r} created successfully at: "
                f"{base_url}/{namespace_slug}/projects/{project_slug}/runs/{run_slug}/"
            )

            # Parse date fields
            date_created = datetime.fromisoformat(
                data["date_created"].replace("Z", "+00:00")
            )
            date_updated = datetime.fromisoformat(
                data["date_updated"].replace("Z", "+00:00")
            )

            # Parse project and user fields
            org_obj = RunOrg(
                slug=namespace_slug,
            )
            project_obj = RunProject(
                uuid=data["project"]["uuid"],
                display_name=data["project"]["display_name"],
                slug=data["project"]["slug"],
            )
            user_obj = RunUser(
                uuid=data["user"]["uuid"],
                username=data["user"]["username"],
                slug=data["user"]["slug"],
            )

            # Create and return the Run object
            run = Run(
                uuid=data["uuid"],
                name=data["name"],
                description=data["description"],
                status=RunStatus(data["status"]),
                date_created=date_created,
                date_updated=date_updated,
                organization=org_obj,
                project=project_obj,
                user=user_obj,
            )
            set_active_run(run)
            _register_safeguard(run)
            return run
        else:
            logger.error(
                f"Failed to create run: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"An error occurred while creating the run: {e}")
        raise


# Safeguard system for finishing the active run
_safeguard_lock = threading.RLock()  # Using `RLock` for re-entrant locking in pytests
_safeguard_run = None
_safeguard_finished = False
_safeguard_registered = False

# Flag to indicate if we are in the atexit handler
# This is used to prevent loggging during the atexit handler,
# which typically result in `ValueError: I/O operation on closed file.`
_IN_ATEXIT = False


def _finish_active_run_on_exit():
    global _IN_ATEXIT
    _IN_ATEXIT = True
    try:
        global _safeguard_finished
        with _safeguard_lock:
            if _safeguard_run and not _safeguard_finished:
                exc_type, exc_val, exc_tb = sys.exc_info()
                if exc_type is not None:
                    # There was an unhandled exception
                    try:
                        _safeguard_run.finish(status=RunStatus.CRASHED)
                    except Exception:
                        if not _IN_ATEXIT:
                            logger.error(
                                "Failed to mark run as CRASHED on exit:\n"
                                + traceback.format_exc()
                            )
                else:
                    try:
                        _safeguard_run.finish(status=RunStatus.COMPLETED)
                    except Exception:
                        if not _IN_ATEXIT:
                            logger.error(
                                "Failed to mark run as COMPLETED on exit:\n"
                                + traceback.format_exc()
                            )
                _safeguard_finished = True
    finally:
        _IN_ATEXIT = False


def _register_safeguard(run):
    global _safeguard_run, _safeguard_finished, _safeguard_registered
    with _safeguard_lock:
        # If there is an unfinished previous run, mark it as FAILED
        if _safeguard_run and not _safeguard_finished and _safeguard_run is not run:
            try:
                _safeguard_run.finish(status=RunStatus.FAILED)
            except Exception:
                logger.error(
                    "Failed to mark previous run as FAILED when starting a new run:\n"
                    + traceback.format_exc()
                )
            finally:
                set_active_run(run)
        _safeguard_run = run
        _safeguard_finished = False
        if not _safeguard_registered:
            atexit.register(_finish_active_run_on_exit)
            _safeguard_registered = True


def _unregister_safeguard(run):
    global _safeguard_run, _safeguard_finished
    with _safeguard_lock:
        if _safeguard_run is run:
            _safeguard_finished = True
            _safeguard_run = None
