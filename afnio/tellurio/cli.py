import logging

import click
import keyring

from afnio.logging_config import configure_logging
from afnio.tellurio import login as module_login

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Tellurio CLI Tool"""
    pass


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
    prompted_for_key = False  # Track if the user was prompted for the API key

    try:
        # Check if the API key is already stored in the keyring
        stored_api_key = keyring.get_password("tellurio", "api_key")
        if stored_api_key and not relogin:
            click.echo("Using the API key stored in your system's keyring.")
            api_key = stored_api_key

        # If no stored key and no key provided, prompt the user
        if not api_key:
            api_key = click.prompt("API Key", hide_input=True)
            prompted_for_key = True

        # Attempt to log in
        response = module_login(api_key=api_key, relogin=relogin)
        if response:
            if prompted_for_key:
                click.echo(
                    "Your API key has been securely saved "
                    "in your system's keyring for future use."
                )
            click.echo("Login successful!")
            logger.info("User logged in successfully.")
        else:
            click.echo("Login failed. Invalid API key.")
            logger.warning("Login attempt failed.")
    except ValueError as e:
        click.echo(f"Login failed: {e}")
        logger.error(f"Login failed: {e}")
    except Exception as e:
        click.echo("An unexpected error occurred. Please try again.")
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    cli()
