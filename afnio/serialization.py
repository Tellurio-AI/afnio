import os
import pickle
import zipfile
from typing import IO, Any, BinaryIO, Type, Union

from typing_extensions import TypeAlias, TypeGuard

DEFAULT_PROTOCOL = 2

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def _is_path(name_or_buffer) -> TypeGuard[Union[str, os.PathLike]]:
    return isinstance(name_or_buffer, (str, os.PathLike))


class _opener:
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_zipfile_writer_file(_opener):
    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # ZipFile only supports ASCII filenames.
            # Use Python's file handling for non-ASCII.
            self.file_stream = open(self.name, mode="wb")
            super().__init__(
                zipfile.ZipFile(
                    self.file_stream, mode="w", compression=zipfile.ZIP_DEFLATED
                )
            )
        else:
            super().__init__(
                zipfile.ZipFile(self.name, mode="w", compression=zipfile.ZIP_DEFLATED)
            )

    def __exit__(self, *args) -> None:
        self.file_like.close()
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener):
    def __init__(self, buffer) -> None:
        if not callable(getattr(buffer, "write", None)):
            msg = (
                f"Buffer of {str(type(buffer)).strip('<>')} "
                f"has no callable attribute 'write'"
            )
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(
            zipfile.ZipFile(self.buffer, mode="w", compression=zipfile.ZIP_DEFLATED)
        )

    def __exit__(self, *args) -> None:
        self.file_like.close()
        self.buffer.flush()


class _open_zipfile_reader(_opener):
    def __init__(self, name_or_buffer) -> None:
        if _is_path(name_or_buffer):
            self.file_like = open(name_or_buffer, "rb")
            self.zipfile = zipfile.ZipFile(self.file_like, "r")
        else:
            self.zipfile = zipfile.ZipFile(name_or_buffer, "r")
        super().__init__(self.zipfile)

    def __exit__(self, *args) -> None:
        self.zipfile.close()
        if _is_path(self.file_like):
            self.file_like.close()


def _open_zipfile_writer(name_or_buffer):
    container: Type[_opener]
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)


def _save(obj, zip_file, pickle_protocol):
    """Helper function to save objects into a zip file."""
    with zip_file.open("data.pkl", "w") as f:
        pickle.dump(obj, f, protocol=pickle_protocol)


def _check_save_filelike(f):
    if not _is_path(f) and not hasattr(f, "write"):
        raise AttributeError(
            "Expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute"
        )


def save(
    obj: object,
    f: FILE_LIKE,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    """
    Saves an object to a disk file using zip compression and pickle serialization.

    Args:
        obj: The object to be saved.
        f: A file-like object (must implement write/flush) or a string or
           os.PathLike object containing a file name.
        pickle_protocol: Pickle protocol version.

    .. note::
        A common Afnio convention is to save variables using .hf file extension.

    Example:
        >>> # Save to file
        >>> x = hf.Variable(data="You are a doctor.", role="system prompt")
        >>> hf.save(x, 'variable.hf')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> hf.save(x, buffer)
    """
    _check_save_filelike(f)

    with _open_zipfile_writer(f) as opened_zipfile:
        _save(
            obj,
            opened_zipfile,
            pickle_protocol,
        )
        return


def load(f: FILE_LIKE) -> Any:
    """
    Loads an object from a disk file using zip compression and pickle serialization.

    Args:
        f: A file-like object (must implement `read`) or a string or os.PathLike
            object containing a file name.

    Returns:
        The deserialized object.

    Example:
        >>> # Load from file
        >>> obj = hf.load('model.hf')
        >>> # Load from io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> obj = hf.load(buffer)
    """
    with _open_zipfile_reader(f) as zip_reader:
        if "data.pkl" not in zip_reader.namelist():
            raise RuntimeError(
                "Missing 'data.pkl' in archive. File might be corrupted."
            )

        # Read the serialized object
        with zip_reader.open("data.pkl", "r") as f:
            obj = pickle.load(f)

    return obj
