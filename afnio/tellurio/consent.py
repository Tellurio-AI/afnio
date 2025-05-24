import json
import os
import sys

from afnio.tellurio.utils import get_config_path

YELLOW = "\033[93m"
RESET = "\033[0m"


def check_consent():
    """
    Check if the user has consented to share their API key with the Tellurio server.
    This function checks the environment variable ALLOW_API_KEY_SHARING, the local
    configuration file, and prompts the user for consent if necessary.

    Raises:
        ValueError: If the environment variable is set to an invalid value.
        RuntimeError: If consent is required but not given in a non-interactive
          environment.
    """
    config_path = get_config_path()
    consent_key = "allow_api_key_sharing"

    # 1. Check environment variable
    env_value = os.getenv("ALLOW_API_KEY_SHARING", "").strip().lower()
    if env_value:
        if env_value not in ("true", "false"):
            raise ValueError(
                "ALLOW_API_KEY_SHARING must be 'true' or 'false' (case-insensitive)."
            )
        if env_value == "true":
            return

    # 2. Check config file
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        if config.get(consent_key) is True:
            return

    # 3. Prompt only if interactive
    if _is_interactive():
        print(
            f"\n{YELLOW}========== [Consent Required] =========={RESET}\n"
            "Your LM model API key will be sent to the Tellurio server for remote model execution and backpropagation.\n"  # noqa: E501
            "\n"
            "Please review the following:\n"
            "  • Tellurio will never use your key except to execute your requests.\n"
            "  • The key is only used during your session.\n"
            "  • The key is never stored and is removed from memory when your session ends.\n"  # noqa: E501
            "\n"
            "Do you consent to share your API key with the server?\n"
            "Type 'yes' to allow just this time, or 'always' to allow and remember your choice for all future sessions.",  # noqa: E501
            flush=True,
        )
        consent = input("Consent [yes/always/no]: ").strip().lower()
        if consent == "always":
            # Load existing config if present, otherwise start with empty dict
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    try:
                        config = json.load(f)
                    except json.JSONDecodeError:
                        config = {}
            else:
                config = {}
            config[consent_key] = True
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return
        elif consent == "yes":
            return
        else:
            raise RuntimeError(
                "Consent to share your LM API key was not given. Aborting operation."
            )
    else:
        raise RuntimeError(
            "Consent to share your API key with the server is required, "
            "but no consent was found and the environment is non-interactive. "
            "Set ALLOW_API_KEY_SHARING=True or update ~/.tellurio.config to allow."
        )


def _is_interactive():
    """Return True if running in a TTY or Jupyter notebook."""
    try:
        # Jupyter notebook or IPython
        from IPython import get_ipython

        if get_ipython():
            return True
    except ImportError:
        pass
    # Fallback to TTY check
    return sys.stdin.isatty()
