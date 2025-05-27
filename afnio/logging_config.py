import logging


class NotebookInfoHandler(logging.Handler):
    """
    Custom logging handler for Jupyter notebooks that prints INFO-level log messages
    with special formatting and color. Intended to make user-facing messages
    more visible and user-friendly in notebook environments.
    """

    COLORS = {
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "reset": "\033[0m",
    }

    def emit(self, record):
        """
        Emit a log record, formatting and coloring specified arguments if provided.
        Only colors arguments specified in the 'colors' attribute of the record.
        Prints the message with a blue '[afnio]' prefix.
        """
        msg = self.format(record)
        colors = getattr(record, "colors", {})
        # Replace each value in the message with its color if specified
        if colors:
            # We assume the log message uses %r or %s for the colored fields
            # and that the order of args matches the keys in 'colors'
            # We'll replace each arg's repr() in the message with colored version
            for idx, color in colors.items():
                try:
                    value = record.args[idx]
                    colored = f"{self.COLORS.get(color, '')}{repr(value)}{self.COLORS['reset']}"  # noqa: E501
                    msg = msg.replace(repr(value), colored, 1)
                except (IndexError, AttributeError):
                    continue
        print(f"{self.COLORS['blue']}[afnio]{self.COLORS['reset']} {msg}")


def in_notebook():
    """
    Detect if the current Python environment is a Jupyter notebook.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


# TODO: Fix notebook behaviour, that prints logs in the wrong cell when exception occours
def configure_logging(verbosity: str = "info"):
    """
    Configure logging for the afnio library.

    Sets up logging format and levels for CLI, scripts, and Jupyter notebooks.
    In a notebook, adds a custom handler for INFO-level logs to display
    user-facing messages with color and formatting.

    Args:
        verbosity (str): Logging level as a string ("info", "debug", etc.).
    """
    if not isinstance(verbosity, str):
        raise TypeError("verbosity must be a string like 'info', 'debug', etc.")
    level = getattr(logging, verbosity.upper(), logging.INFO)
    if level <= logging.DEBUG:
        fmt = "%(asctime)s - %(name)-35s - %(levelname)-9s - %(message)s"
    else:
        fmt = "%(levelname)-9s: %(message)s"

    # Set root logger to WARNING (default for all libraries)
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        force=True,
    )

    # Set only afnio logs to the desired level
    logging.getLogger("afnio").setLevel(level)

    # Add notebook handler for INFO logs if in notebook
    if in_notebook() and level == logging.INFO:
        handler = NotebookInfoHandler()
        handler.setLevel(logging.INFO)
        handler.addFilter(lambda record: record.levelno == logging.INFO)
        logging.getLogger().addHandler(handler)

        class NoInfoFilter(logging.Filter):
            """
            Logging filter that suppresses INFO-level log records.
            Used to prevent duplicate INFO messages in notebook environments.
            """

            def filter(self, record):
                """
                Filter method to suppress INFO-level log records.

                Args:
                    record (logging.LogRecord): The log record to filter.

                Returns:
                    bool: True if the record should be logged, False otherwise.
                """
                return record.levelno != logging.INFO

        for h in logging.getLogger().handlers:
            if not isinstance(h, NotebookInfoHandler):
                h.addFilter(NoInfoFilter())
