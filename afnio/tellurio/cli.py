import os

import click
import keyring

from afnio.logging_config import configure_logging
from afnio.tellurio import login as module_login
from afnio.tellurio.client import InvalidAPIKeyError, load_username


def afnio_echo(message, *args, fg=None, **kwargs):
    """
    Print a message to the console with a specific prefix.

    Args:
        message (str): The message to print.
        *args: Additional arguments to pass to click.secho.
        fg (str): The foreground color for the message.
        **kwargs: Additional keyword arguments to pass to click.secho.
    """
    prefix = click.style("[afnio] ", fg="blue")
    click.secho(prefix + message, fg=fg, *args, **kwargs)


@click.group()
@click.option(
    "--verbosity",
    "-v",
    default="warning",
    show_default=True,
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"], case_sensitive=False
    ),
    help="Set the logging verbosity level.",
)
def cli(verbosity):
    """Tellurio CLI Tool"""
    configure_logging(verbosity)


@cli.command()
@click.option("--api-key", help="Your API key.", required=False, hide_input=True)
@click.option(
    "--relogin", is_flag=True, help="Force a re-login and request a new API key."
)
def login(api_key, relogin):
    """
    Log in to Tellurio using an API key.

    This command allows you to authenticate with the Tellurio platform using
    your API key. If the API key is already stored in the system's keyring,
    it will be used automatically without prompting the user.

    Args:
        api_key (str): The user's API key for authentication.

    Returns:
        None
    """
    username = load_username()
    service_name = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    try:
        # Check if the API key is already stored in the keyring
        stored_api_key = (
            keyring.get_password(service_name, username) if username else None
        )

        # Decide which API key to use
        if stored_api_key and not relogin:
            # If stored key is available and not forcing relogin,
            # we let module_login() to retrieve it internally
            api_key_to_use = None
            afnio_echo("Using stored API key from local keyring.")
        else:
            # Use provided api_key or prompt the user
            if not api_key:
                api_key = click.prompt(
                    "Please enter your Tellurio API key and hit enter, "
                    "or press ctrl+c to quit",
                    hide_input=True,
                )
            api_key_to_use = api_key

        # Log in to the Tellurio platform
        response = module_login(api_key=api_key_to_use, relogin=relogin)

        if api_key_to_use:
            afnio_echo("API key provided and stored securely in local keyring.")

        base_url = os.getenv(
            "TELLURIO_BACKEND_HTTP_BASE_URL", "https://platform.tellurio.ai"
        )

        username_str = click.style(repr(response["username"]), fg="yellow")
        base_url_str = click.style(repr(base_url), fg="green")
        afnio_echo(
            f"Currently logged in as {username_str} to {base_url_str}. "
            f"Use `afnio login --relogin` to force relogin."
        )
    except ValueError:
        afnio_echo("Login failed: Missing API key. Please provide a valid API key.")
    except InvalidAPIKeyError:
        afnio_echo("Login failed: Invalid API key. Please try again.")
    except RuntimeError:
        afnio_echo("Login failed: Failed to connect to the backend.")
    except Exception:
        afnio_echo("An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    cli()
