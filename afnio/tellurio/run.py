import logging
import os
from datetime import datetime
from enum import Enum
from typing import Optional

from slugify import slugify

from afnio.tellurio.client import TellurioClient, get_default_client
from afnio.tellurio.project import create_project, get_project
from afnio.tellurio.run_context import set_active_run_uuid

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

    def finish(self, client: Optional[TellurioClient] = None):
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
        payload = {"status": RunStatus.COMPLETED.value}

        try:
            response = client.patch(endpoint, json=payload)
            if response.status_code == 200:
                self.status = RunStatus.COMPLETED
                logger.info(f"Run {self.name!r} marked as COMPLETED.")
            else:
                logger.error(
                    f"Failed to update run status: {response.status_code} - {response.text}"  # noqa: E501
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"An error occurred while updating the run status: {e}")
            raise

        # Clear the active run UUID after finishing
        try:
            set_active_run_uuid(None)
        except Exception:
            pass

    # TODO: If any error happens we should update the run status to CRASHED; Also the server side should implement this


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
        logger.info(
            f"Project with slug {project_slug!r} already exists "
            f"in namespace {namespace_slug!r}."
        )
    except Exception:
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
            set_active_run_uuid(run.uuid)
            return run
        else:
            logger.error(
                f"Failed to create run: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"An error occurred while creating the run: {e}")
        raise
