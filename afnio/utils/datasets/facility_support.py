"""Meta's Facility Support Analyzer dataset."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Union
from urllib.error import URLError

from afnio._variable import Variable
from afnio.utils.data.dataset import Dataset
from afnio.utils.datasets.utils import check_integrity, download


class FacilitySupport(Dataset):
    """The Meta Facility Support Analyzer dataset consists of 200 real-world emails or
    messages sent in enterprise settings related to facility maintenance or support
    requests. Each example is annotated with:
      - urgency (low, medium, high)
      - sentiment (negative, neutral, positive)
      - relevant service request categories (e.g., cleaning, IT support, maintenance)

    The dataset is split into train, validation, and test sets with a 33%/33%/34%
    ratio. The split is deterministic, ensuring reproducibility across different runs.

    Args:
        split (str): The dataset split to load. Must be either "train", "val",
          or "test".
        root (Union[str, Path], optional): The root directory where JSON files are
          stored. Defaults to None.
    """

    mirrors = [
        "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/"  # noqa: E501
    ]

    resources = [
        ("dataset.json", "530dc66b1b07c9b15b19f08891e9bfa0"),
    ]

    _repr_indent = 4

    def __init__(self, split: str, root: Union[str, Path] = None) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"FacilitySupport Dataset: expected split in ['train', 'val', 'test'], "
                f"but got split={split}"
            )

        if isinstance(root, str):
            root = os.path.expanduser(root)

        self.split = split
        self.root = root

        self.download()

        # Load dataset from JSON
        file_path = os.path.join(self.raw_folder, self.resources[0][0])
        with open(file_path, "r", encoding="utf-8") as f:
            dataset: List[Dict] = json.load(f)

        # Shuffle deterministically
        random.Random(0).shuffle(dataset)

        n = len(dataset)
        n_train = int(n * 0.33)
        n_val = int(n * 0.33)

        if split == "train":
            self.data = dataset[:n_train]
        elif split == "val":
            self.data = dataset[n_train : n_train + n_val]  # noqa: E203
        else:  # test
            self.data = dataset[n_train + n_val :]  # noqa: E203

    def __getitem__(self, index: int) -> Dict:
        """
        Fetches a data sample (message, (urgency, sentiment, categories))
        for a given index.
        """
        if not (0 <= index < len(self.data)):
            raise IndexError("Index out of range.")

        item = self.data[index]

        answer: dict = json.loads(item["answer"])
        urgency = answer.get("urgency", None)
        sentiment = answer.get("sentiment", None)
        categories = answer.get("categories", None)

        message = Variable(
            data=item["fields"]["input"],
            role="input email or message",
        )
        urgency = Variable(data=urgency, role="output urgency")
        sentiment = Variable(data=sentiment, role="output sentiment")
        categories = Variable(data=json.dumps(categories), role="output categories")
        return message, (urgency, sentiment, categories)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def extra_repr(self) -> str:
        """Returns additional information about the dataset."""
        split_map = {"train": "Train", "val": "Validation", "test": "Test"}

        try:
            split = split_map[self.split]
        except KeyError:
            raise ValueError(
                f"Invalid split value: {self.split}. "
                f"Expected one of ['train', 'val', 'test']."
            )

        return f"Split: {split}"

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        return all(
            check_integrity(
                os.path.join(
                    self.raw_folder, os.path.splitext(os.path.basename(url))[0]
                )
            )
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the Facility Support data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    download(
                        url, download_root=self.raw_folder, filename=filename, md5=md5
                    )
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")
