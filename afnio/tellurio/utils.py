import os


def get_config_path():
    """
    Returns the absolute path to the Tellurio configuration file.

    This function always returns the path to the default config file,
    ~/.tellurio_config.json, with '~' expanded to the user's home directory.

    Returns:
        str: The absolute path to the Tellurio config file.
    """
    return os.path.expanduser("~/.tellurio_config.json")
