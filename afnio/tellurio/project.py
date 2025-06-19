import logging
from datetime import datetime
from typing import Optional

from afnio.tellurio.client import TellurioClient, get_default_client

logger = logging.getLogger(__name__)


class Project:
    """
    Represents a Tellurio Project with detailed information.
    """

    def __init__(
        self,
        uuid: str,
        display_name: str,
        slug: str,
        date_created: datetime,
        last_run_date: Optional[datetime],
        visibility: str,
        role: Optional[str],
        org_uuid: Optional[str],
        org_display_name: Optional[str],
        org_slug: Optional[str],
    ):
        self.uuid = uuid
        self.display_name = display_name
        self.slug = slug
        self.date_created = date_created
        self.last_run_date = last_run_date
        self.visibility = visibility
        self.role = role
        self.org_uuid = org_uuid
        self.org_display_name = org_display_name
        self.org_slug = org_slug

    def __repr__(self):
        return (
            f"<Project uuid={self.uuid} display_name={self.display_name} "
            f"visibility={self.visibility} role={self.role}>"
        )


def get_project(
    namespace_slug: str,
    project_slug: str,
    client: Optional[TellurioClient] = None,
) -> Optional[Project]:
    """
    Retrieves a project by its slug in the specified namespace.

    Args:
        namespace_slug (str): The namespace slug where the project resides. It can be
          either an organization slug or a user slug.
        project_slug (str): The slug of the project to retrieve.
        client (TellurioClient, optional): An instance of TellurioClient. If not
          provided, the default client will be used.

    Returns:
        Project: A Project object representing the retrieved project
            or None if not found.

    Raises:
        Exception: If an unexpected error occurs during the request.
    """
    client = client or get_default_client()[0]

    # Define the endpoint
    endpoint = f"/api/v0/{namespace_slug}/projects/{project_slug}/"

    try:
        response = client.get(endpoint)

        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Project retrieved successfully: {data}")

            # Parse date fields
            date_created = datetime.fromisoformat(
                data["date_created"].replace("Z", "+00:00")
            )
            last_run_date = (
                datetime.fromisoformat(data["last_run_date"].replace("Z", "+00:00"))
                if data.get("last_run_date")
                else None
            )

            # Create and return the Project object
            return Project(
                uuid=data["uuid"],
                display_name=data["display_name"],
                slug=data["slug"],
                date_created=date_created,
                last_run_date=last_run_date,
                visibility=data["visibility"],
                role=data.get("role"),
                org_uuid=data.get("org_uuid"),
                org_display_name=data.get("org_display_name"),
                org_slug=data.get("org_slug"),
            )
        elif response.status_code == 404:
            logger.debug(
                f"Project with slug '{project_slug}' "
                f"not found in namespace '{namespace_slug}'."
            )
            return None
        else:
            logger.error(
                f"Failed to retrieve project: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"An error occurred while retrieving the project: {e}")
        raise


def create_project(
    namespace_slug: str,
    display_name: str,
    visibility: str = "TEAM",
    client: Optional[TellurioClient] = None,
) -> Project:
    """
    Creates a new project in the specified namespace.

    Args:
        namespace_slug (str): The namespace slug where the project resides. It can be
          either an organization slug or a user slug.
        display_name (str): The display name of the project.
        visibility (str): The visibility of the project (default: "TEAM").
        client (TellurioClient, optional): An instance of TellurioClient. If not
          provided, the default client will be used.

    Returns:
        Project: A Project object representing the created project.
    """
    client = client or get_default_client()[0]

    # Define the endpoint and payload
    endpoint = f"/api/v0/{namespace_slug}/projects/"
    payload = {
        "display_name": display_name,
        "visibility": visibility,
    }

    try:
        response = client.post(endpoint, json=payload)

        if response.status_code == 201:
            data = response.json()
            logger.debug(f"Project created successfully: {data}")

            # Parse date fields
            date_created = datetime.fromisoformat(
                data["date_created"].replace("Z", "+00:00")
            )
            last_run_date = (
                datetime.fromisoformat(data["last_run_date"].replace("Z", "+00:00"))
                if data.get("last_run_date")
                else None
            )

            # Create and return the Project object
            return Project(
                uuid=data["uuid"],
                display_name=data["display_name"],
                slug=data["slug"],
                date_created=date_created,
                last_run_date=last_run_date,
                visibility=data["visibility"],
                role=data["role"],
                org_uuid=data["org_uuid"],
                org_display_name=data["org_display_name"],
                org_slug=data["org_slug"],
            )
        else:
            logger.error(
                f"Failed to create project: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"An error occurred while creating the project: {e}")
        raise


def delete_project(
    namespace_slug: str,
    project_slug: str,
    client: Optional[TellurioClient] = None,
) -> None:
    """
    Deletes a project by in the specified namespace.

    Args:
        namespace_slug (str): The namespace slug where the project resides. It can be
          either an organization slug or a user slug.
        project_slug (str): The slug of the project to delete.
        client (TellurioClient, optional): An instance of TellurioClient. If not
          provided, the default client will be used.

    Returns:
        None
    """
    client = client or get_default_client()[0]

    # Define the endpoint
    endpoint = f"/api/v0/{namespace_slug}/projects/{project_slug}/"

    try:
        response = client.delete(endpoint)

        if response.status_code == 204:
            logger.debug(f"Project '{project_slug}' deleted successfully.")
        else:
            logger.error(
                f"Failed to delete project: {response.status_code} - {response.text}"
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"An error occurred while deleting the project: {e}")
        raise
