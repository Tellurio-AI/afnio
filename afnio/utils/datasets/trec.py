"""The Text REtrieval Conference (TREC) Question Classification dataset."""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.error import URLError

from afnio.utils.data.dataset import Dataset
from afnio.utils.datasets.utils import check_integrity, download


class TREC(Dataset):
    """The Text REtrieval Conference (TREC) Question Classification dataset contains
    5452 labeled questions in the training set (before removing duplicates) and 5382
    unique labeled questions (after removing duplicates), along with another 500
    questions for the test set.

    The dataset has 6 coarse class labels and 50 fine class labels. Average length of
    each sentence is 10, vocabulary size of 8700.

    Data are collected from four sources: 4,500 English questions published by USC
    (Hovy et al., 2001), about 500 manually constructed questions for a few rare
    classes, 894 TREC 8 and TREC 9 questions, and also 500 questions from TREC 10 which
    serves as the test set. These questions were manually labeled.

    ``TREC`` provides a stratified train set and validation set, ensuring that both
    splits maintain the same class distribution proportions as in the original dataset.

    Args:
        task (str, optional): Defines the classes to classify between
            ``["coarse", "fine"]``. Defaults to None.
        split (str, optional): The dataset split in ``["train", "val", "test"]``.
            Defaults to None.
        validation_split (Optional[float], optional): Float between 0 and 1. Fraction
            of the training data to be used as validation data. Defaults to 0.0.
        root (Union[str, Path], optional): Root directory of dataset where
            ``TREC/raw/train_5500.label`` and  ``TREC/raw/TREC_10.label`` exist.
            Defaults to None.
    """

    mirrors = ["https://cogcomp.seas.upenn.edu/Data/QA/QC/"]

    resources = [
        ("train_5500.label", "073462e3fcefaae31e00edb1f18d2d02"),
        ("TREC_10.label", "323a3554401d86e650717e2d2f942589"),
    ]

    _repr_indent = 4

    def __init__(
        self,
        task: str = None,
        split: str = None,
        validation_split: Optional[float] = 0.0,
        root: Union[str, Path] = None,
    ) -> None:
        if task not in {"coarse", "fine"}:
            raise ValueError(
                f"TREC Dataset: expected classification task in ['coarse', 'fine'], "
                f"but got task={task}"
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"TREC Dataset: expected split in ['train', 'val', 'test'], "
                f"but got split={split}"
            )

        if validation_split < 0.0 or validation_split > 1.0:
            raise ValueError(
                f"TREC Dataset: expected validation_split in [0.0, 1.0], "
                f"but got validation_split={validation_split}"
            )

        if isinstance(root, str):
            root = os.path.expanduser(root)

        self.task = task
        self.split = split
        self.root = root

        self.download()

        if split == "train":
            (self.data, self.targets), (_, _) = self._load_train_and_val_data(
                task=self.task, validation_split=validation_split
            )
        elif split == "val":
            (_, _), (self.data, self.targets) = self._load_train_and_val_data(
                task=self.task, validation_split=validation_split
            )
        elif split == "test":
            self.data, self.targets = self._load_test_data()
        else:
            self.data, self.targets = None, None

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def extra_repr(self) -> str:
        split_map = {"train": "Train", "val": "Validation", "test": "Test"}
        task_map = {"coarse": "Classify Coarse Labels", "fine": "Classify Fine Labels"}

        try:
            split = split_map[self.split]
        except KeyError:
            raise ValueError(
                f"Invalid split value: {self.split}. "
                f"Expected one of ['train', 'val', 'test']."
            )
        try:
            task = task_map[self.task]
        except KeyError:
            raise ValueError(
                f"Invalid task value: {self.task}. Expected one of ['coarse', 'fine']."
            )

        return f"Split: {split}\nTask: {task}"

    def __repr__(self) -> str:
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
        """Download the TREC data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
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

    def _load_train_and_val_data(self, task: str = None, validation_split: float = 0.0):
        train_file_path = os.path.join(self.raw_folder, self.resources[0][0])

        data = []
        targets = []
        unique_samples = set()  # A set to track unique samples

        with open(train_file_path, "rb") as f:
            for row in f:
                # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
                fine_label, _, text = (
                    row.replace(b"\xf0", b" ").strip().decode().partition(" ")
                )
                coarse_label = fine_label.split(":")[0]
                sample = (text, (fine_label, coarse_label))

                # Only add unique samples
                if sample not in unique_samples:
                    unique_samples.add(sample)
                    data.append(text)
                    targets.append((fine_label, coarse_label))

        # Group data by either fine_label or coarse_label based on the task
        label_to_data = defaultdict(list)
        for text, (fine_label, coarse_label) in zip(data, targets):
            label = fine_label if task == "fine" else coarse_label
            label_to_data[label].append((text, fine_label, coarse_label))

        # Split the data based on validation_split
        train_data = []
        train_targets = []
        val_data = []
        val_targets = []

        random.seed(42)

        for label, samples in label_to_data.items():
            # Ensure there are enough samples to split
            if len(samples) < (len(samples) * validation_split):
                raise ValueError(
                    f"Not enough data for label '{label}' to respect the validation split."  # noqa: E501
                )

            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - validation_split))

            if len(samples[:split_idx]) == 0:
                raise ValueError(f"Label {label} missing from the training set.")
            if len(samples[split_idx:]) == 0 and validation_split > 0.0:
                raise ValueError(f"Label {label} missing from the validation set.")

            # Add to training set
            for sample in samples[:split_idx]:
                train_data.append(sample[0])
                train_targets.append((sample[1], sample[2]))

            # Add to validation set
            for sample in samples[split_idx:]:
                val_data.append(sample[0])
                val_targets.append((sample[1], sample[2]))

        return (train_data, train_targets), (val_data, val_targets)

    def _load_test_data(self):
        test_file_path = os.path.join(self.raw_folder, self.resources[1][0])

        data = []
        targets = []

        with open(test_file_path, "rb") as f:
            for row in f:
                # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
                fine_label, _, text = (
                    row.replace(b"\xf0", b" ").strip().decode().partition(" ")
                )
                coarse_label = fine_label.split(":")[0]
                data.append(text)
                targets.append((fine_label, coarse_label))

        return data, targets


class TRECTmp(Dataset):
    mirrors = ["https://cogcomp.seas.upenn.edu/Data/QA/QC/"]

    resources = [
        ("train_5500.label", "073462e3fcefaae31e00edb1f18d2d02"),
        ("TREC_10.label", "323a3554401d86e650717e2d2f942589"),
    ]

    _repr_indent = 4

    def __init__(
        self,
        task: str = None,
        split: str = None,
        validation_split: Optional[float] = 0.0,
        root: Union[str, Path] = None,
    ) -> None:
        if task not in {"coarse", "fine"}:
            raise ValueError(
                f"TREC Dataset: expected classification task in ['coarse', 'fine'], "
                f"but got task={task}"
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"TREC Dataset: expected split in ['train', 'val', 'test'], "
                f"but got split={split}"
            )

        if validation_split < 0.0 or validation_split > 1.0:
            raise ValueError(
                f"TREC Dataset: expected validation_split in [0.0, 1.0], "
                f"but got validation_split={validation_split}"
            )

        if isinstance(root, str):
            root = os.path.expanduser(root)

        self.task = task
        self.split = split
        self.root = root

        self.download()

        if split == "train":
            self.data, self.targets = self._load_train_data()
        elif split == "val":
            (self.data, self.targets), (_, _) = self._load_val_and_test_data(
                task=self.task, validation_split=validation_split
            )
        elif split == "test":
            (_, _), (self.data, self.targets) = self._load_val_and_test_data(
                task=self.task, validation_split=validation_split
            )
        else:
            self.data, self.targets = None, None

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

    def extra_repr(self) -> str:
        split_map = {"train": "Train", "val": "Validation", "test": "Test"}
        task_map = {"coarse": "Classify Coarse Labels", "fine": "Classify Fine Labels"}

        try:
            split = split_map[self.split]
        except KeyError:
            raise ValueError(
                f"Invalid split value: {self.split}. "
                f"Expected one of ['train', 'val', 'test']."
            )
        try:
            task = task_map[self.task]
        except KeyError:
            raise ValueError(
                f"Invalid task value: {self.task}. Expected one of ['coarse', 'fine']."
            )

        return f"Split: {split}\nTask: {task}"

    def __repr__(self) -> str:
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
        """Download the TREC data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
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

    def _load_train_data(self):
        train_file_path = os.path.join(self.raw_folder, self.resources[0][0])

        data = []
        targets = []
        unique_samples = set()  # A set to track unique samples
        label_to_samples = defaultdict(list)

        with open(train_file_path, "rb") as f:
            for row in f:
                # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
                fine_label, _, text = (
                    row.replace(b"\xf0", b" ").strip().decode().partition(" ")
                )
                coarse_label = fine_label.split(":")[0]
                sample = (text, (fine_label, coarse_label))

                # Only add unique samples
                if sample not in unique_samples:
                    unique_samples.add(sample)
                    data.append(text)
                    targets.append((fine_label, coarse_label))
                    label_to_samples[coarse_label].append(sample)

        random.seed(42)

        selected_data = []
        selected_targets = []
        for label, samples in label_to_samples.items():
            random.shuffle(samples)
            selected_samples = samples[:20]  # Select 20 samples per class
            for text, (fine_label, coarse_label) in selected_samples:
                selected_data.append(text)
                selected_targets.append((fine_label, coarse_label))

        return selected_data, selected_targets

    def _load_val_and_test_data(self, task: str = None, validation_split: float = 0.0):
        test_file_path = os.path.join(self.raw_folder, self.resources[1][0])

        data = []
        targets = []
        unique_samples = set()  # A set to track unique samples

        with open(test_file_path, "rb") as f:
            for row in f:
                # One non-ASCII byte: sisterBADBYTEcity. We replace it with a space
                fine_label, _, text = (
                    row.replace(b"\xf0", b" ").strip().decode().partition(" ")
                )
                coarse_label = fine_label.split(":")[0]
                sample = (text, (fine_label, coarse_label))

                # Only add unique samples
                if sample not in unique_samples:
                    unique_samples.add(sample)
                    data.append(text)
                    targets.append((fine_label, coarse_label))

        # Group data by either fine_label or coarse_label based on the task
        label_to_data = defaultdict(list)
        for text, (fine_label, coarse_label) in zip(data, targets):
            label = fine_label if task == "fine" else coarse_label
            label_to_data[label].append((text, fine_label, coarse_label))

        # Split the data based on validation_split
        val_data = []
        val_targets = []
        test_data = []
        test_targets = []

        random.seed(42)

        for label, samples in label_to_data.items():
            # Ensure there are enough samples to split
            if len(samples) < (len(samples) * validation_split):
                raise ValueError(
                    f"Not enough data for label '{label}' to respect the validation split."  # noqa: E501
                )

            random.shuffle(samples)
            split_idx = int(len(samples) * (1 - validation_split))

            if len(samples[:split_idx]) == 0:
                raise ValueError(f"Label {label} missing from the training set.")
            if len(samples[split_idx:]) == 0 and validation_split > 0.0:
                raise ValueError(f"Label {label} missing from the validation set.")

            # Add to validation set
            for sample in samples[:split_idx]:
                val_data.append(sample[0])
                val_targets.append((sample[1], sample[2]))

            # Add to test set
            for sample in samples[split_idx:]:
                test_data.append(sample[0])
                test_targets.append((sample[1], sample[2]))

        return (val_data, val_targets), (test_data, test_targets)
