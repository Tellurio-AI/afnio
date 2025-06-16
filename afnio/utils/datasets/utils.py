import hashlib
import os
import pathlib
import sys
import urllib
import urllib.request
from typing import Any, Optional, Union

import tqdm

USER_AGENT = "afnio"


def _urlretrieve(
    url: str, filename: Union[str, pathlib.Path], chunk_size: int = 1024 * 32
) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ) as response:
        with open(filename, "wb") as fh, tqdm.tqdm(total=response.length) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


def calculate_md5(
    fpath: Union[str, pathlib.Path], chunk_size: int = 1024 * 1024
) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the
    # functionality, but indicates that we are not using the MD5 checksum for
    # cryptography. This enables its usage in restricted environments like FIPS.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: Union[str, pathlib.Path], md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: Union[str, pathlib.Path], md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url(
    url: str,
    root: Union[str, pathlib.Path],
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the
            basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead. Downloading "
                + url
                + " to "
                + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def download(
    url: str,
    download_root: Union[str, pathlib.Path],
    extract_root: Optional[Union[str, pathlib.Path]] = None,
    filename: Optional[Union[str, pathlib.Path]] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
