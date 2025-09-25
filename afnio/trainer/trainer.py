import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from IPython import get_ipython
from IPython.display import clear_output
from rich.console import Console
from rich.progress import (
    BarColumn,
    Column,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

import afnio as hf
import afnio.cognitive as cog
from afnio._variable import Variable
from afnio.logging_config import configure_logging, set_logger_level
from afnio.models.model import BaseModel
from afnio.tellurio import log
from afnio.utils.data import DataLoader

_PATH = Union[str, Path]
TRAIN_DATALOADER = Iterable[Any]
EVAL_DATALOADER = Iterable[Any]

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class MinutesPerStepColumn(ProgressColumn):
    """
    Show average minutes per step as Xm/step, styled like TimeElapsedColumn,
    only for training.
    """

    def render(self, task):
        # Only show for training task (assume description contains '[Training]')
        if "[Training]" not in (task.fields.get("desc") or ""):
            return Text("")
        # Use final values if present (after training)
        training_step_times = task.fields.get("training_step_times", [])
        if not training_step_times:
            return Text("----m/step", style="progress.elapsed")
        min_per_step = sum(training_step_times) / len(training_step_times) / 60
        return Text(f"{min_per_step:.1f}m/step", style="progress.elapsed")


class Trainer:
    def __init__(
        self,
        *,
        max_epochs: Optional[int] = None,
        # min_epochs: Optional[int] = None,
        # max_steps: int = -1,
        # min_steps: Optional[int] = None,
        # max_time: Optional[Union[str, timedelta, dict[str, int]]] = None,
        # max_cost: Optional[float] = None,
        enable_checkpointing: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = True,
        enable_agent_summary: Optional[bool] = True,
        default_root_dir: Optional[_PATH] = None,
    ) -> None:
        r"""Customize every aspect of training via flags.

        Args:
            max_epochs: Stop training once this number of epochs is reached. Disabled
                by default (None). If both max_epochs and max_steps are not specified,
                defaults to ``max_epochs = 10``.
            enable_checkpointing: If ``True``, enable checkpointing.
                Default: ``True``.
            enable_progress_bar: Whether to enable to progress bar by default.
                Default: ``True``.
            enable_agent_summary: Whether to enable agent summarization by default.
                Default: ``True``.
            default_root_dir: Default path for logs, checkpoints and other artifacts.
                Default: ``os.getcwd()``.
        """

        # r"""Customize every aspect of training via flags.

        # Args:
        #     max_epochs: Stop training once this number of epochs is reached. Disabled
        #         by default (None). If both max_epochs and max_steps are not specified,
        #         defaults to ``max_epochs = 10``.
        #     min_epochs: Force training for at least these many epochs.
        #         Disabled by default (None).
        #     max_steps: Stop training after this number of steps. Disabled by default
        #         (-1). If ``max_steps = -1`` and ``max_epochs = None``, will default to
        #         ``max_epochs = 10``.
        #     min_steps: Force training for at least these number of steps.
        #         Disabled by default (``None``).
        #     max_time: Stop training after this amount of time has passed. Disabled by
        #         default (``None``). The time duration can be specified in the format
        #         DD:HH:MM:SS (days, hours, minutes seconds), as a
        #         :class:`datetime.timedelta`, or a dictionary with keys that will be
        #         passed to :class:`datetime.timedelta`.
        #     max_cost: Stop training after this amount (USD) of cost has been incurred.
        #         Disabled by default (``None``). The cost is a float value that can be
        #         used to limit the computational resources consumed during training.
        #         This is useful for budgeted training scenarios.
        #     enable_checkpointing: If ``True``, enable checkpointing.
        #         Default: ``True``.
        #     enable_progress_bar: Whether to enable to progress bar by default.
        #         Default: ``True``.
        #     enable_agent_summary: Whether to enable agent summarization by default.
        #         Default: ``True``.
        #     default_root_dir: Default path for logs, checkpoints and other artifacts.
        #         Default: ``os.getcwd()``.
        # """
        self.max_epochs = max_epochs
        # self.min_epochs = min_epochs
        # self.max_steps = max_steps
        # self.min_steps = min_steps
        # self.max_time = max_time
        # self.max_cost = max_cost
        self.enable_checkpointing = enable_checkpointing
        self.enable_progress_bar = enable_progress_bar
        self.enable_agent_summary = enable_agent_summary
        self._is_notebook = self._in_notebook()
        self.buffer = [] if self._is_notebook else None
        self.total_cost = 0.0

        # TODO: Re-enable once `max_steps` is implemented
        # if not max_epochs and max_steps == -1:
        #     # If max_epochs is not set and max_steps is -1, default to 10 epochs
        #     self.max_epochs = 10

        if not max_epochs:
            self.max_epochs = 10

        self.default_root_dir = (
            os.getcwd() if default_root_dir is None else os.fspath(default_root_dir)
        )

    def _in_notebook(self):
        """Check if the code is running in a Jupyter notebook."""
        try:
            ip = get_ipython()
            if ip is None:
                return False
            shell = ip.__class__.__name__
            # VS Code, Colab, Jupyter
            return shell in ("ZMQInteractiveShell", "Shell")
        except Exception:
            return False

    def _in_notebook_cell_changed(self):
        """Check if the notebook cell has changed."""
        try:
            ip = get_ipython()
            count = ip.execution_count
            if not hasattr(self, "_last_execution_count"):
                self._last_execution_count = count
                return False
            if self._last_execution_count != count:
                self._last_execution_count = count
                return True
            return False
        except Exception:
            return False

    def _setup_progress(self, mode, console):
        """Setup the progress bar for training or validation."""
        progress = Progress(
            TextColumn(
                "{task.fields[desc]}",
                justify="right",
                table_column=Column(justify="left"),
            ),
            BarColumn(table_column=Column(max_width=20)),
            TimeElapsedColumn(),
            *([MinutesPerStepColumn()] if mode == "train" else []),
            TextColumn("{task.fields[metrics]}", table_column=Column(justify="left")),
            refresh_per_second=5,
            transient=False,
            console=console,
        )
        progress.start()
        return progress

    def _teardown_progress(self, progress, refresher_stop, refresher_thread):
        """Stop the progress bar and any background threads."""
        if refresher_stop:
            refresher_stop.set()
            refresher_thread.join()
        if progress:
            progress.stop()

    def _run_fn_with_retries(
        self, func, batch, batch_idx, max_retries=3, step_name="step"
    ):
        """
        Run a function with retries in case of failure.
        Raises RuntimeError if all retries fail.
        """
        for retry_count in range(1, max_retries + 1):
            try:
                return func(batch, batch_idx)
            except Exception as e:
                if retry_count == max_retries:
                    raise RuntimeError(
                        f"Forward pass in {step_name}() failed after {max_retries} retries."  # noqa: E501
                    ) from e
                logger.warning(
                    f"Retry {retry_count}/{max_retries}: Forward pass in {step_name}() failed, retrying...",  # noqa: E501
                )

    def _run_training_step_with_retries(
        self,
        func,
        batch,
        batch_idx,
        optimizer,
        max_retries=3,
        automatic=True,
    ):
        for retry_count in range(1, max_retries + 1):
            try:
                if automatic:
                    optimizer.clear_grad()
                    step_out = self._run_fn_with_retries(
                        func,
                        batch,
                        batch_idx,
                        max_retries=3,
                        step_name="training_step",
                    )
                # Retrials should be handled by the user in manual mode
                else:
                    step_out = func(batch, batch_idx)
                batch_metrics = self._parse_step_metrics(step_out, "training_step")
                if automatic:
                    _, explanation = batch_metrics["loss"]
                    explanation.backward()
                    optimizer.step()
                return batch_metrics
            except Exception as e:
                if retry_count == max_retries:
                    raise RuntimeError(
                        f"training_step() failed after {max_retries} retries: {e}"
                    ) from e
                logger.warning(
                    f"Retry {retry_count}/{max_retries}: training_step() failed with error '{e}', retrying...",  # noqa: E501
                )

    def _parse_step_metrics(self, step_out, step_name="training_step"):
        """
        Parse step output and return a dict of metrics.
        """
        metrics = {}

        # Handle dict with 'loss' key
        if isinstance(step_out, dict):
            metrics.update(step_out)
        # Handle tuple (score, explanation)
        elif isinstance(step_out, tuple) and len(step_out) == 2:
            metrics["loss"] = (step_out[0], step_out[1])
        else:
            raise ValueError(
                f"{step_name}() must return either a tuple of Variables "
                f"(score, explanation) or a dict with 'loss' key and "
                f"containing a tuple of two Variables (score, explanation), "
                f"but got {type(step_out)}"
            )

        # Ensure 'loss' is a tuple of two Variables
        loss = metrics.get("loss")
        if not isinstance(loss, tuple) or not len(loss) == 2:
            raise ValueError(
                f"{step_name}() must return a loss which is a tuple of two Variables "
                f"(score, explanation), but got {type(loss)}"
            )
        score, explanation = loss
        if not (isinstance(score, Variable) and isinstance(explanation, Variable)):
            raise TypeError(
                f"Both score and explanation must be afnio.Variable, "
                f"got {type(score)} and {type(explanation)} in {step_name}()"
            )

        return metrics

    def _collect_metrics(self, batch_metrics, metrics_dict):
        """Collect metrics from a batch into the provided metrics_dict."""
        for k, v in batch_metrics.items():
            if k == "loss" and isinstance(v, tuple) and len(v) == 2:
                score = v[0]
                metrics_dict[k].append(
                    score.data if isinstance(score, Variable) else score
                )
            else:
                metrics_dict[k].append(v.data if isinstance(v, Variable) else v)

    def _average_metrics(self, metrics_dict):
        """Compute the average for each metric key."""
        return {k: sum(vs) / len(vs) for k, vs in metrics_dict.items() if len(vs) > 0}

    def _ordered_metrics(self, metrics_dict):
        """
        Return a list of (key, value) pairs with 'loss' key first,
        followed by sorted keys.
        """
        keys = list(metrics_dict.keys())
        ordered = []
        if "loss" in keys:
            ordered.append("loss")
            keys.remove("loss")
        ordered += sorted(keys)
        return [(k, metrics_dict[k]) for k in ordered]

    def _save_checkpoint(self, agent, optimizer, epoch, batch=None):
        """
        Save agent and optimizer state at the specified location.
        """
        ckpt_dir = os.path.join(self.default_root_dir, "checkpoints")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if batch is not None:
            epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            ckpt_path = os.path.join(
                epoch_dir, f"checkpoint_batch{batch}_{timestamp}.hf"
            )
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir, f"checkpoint_epoch{epoch}_{timestamp}.hf"
            )
        checkpoint = {
            "epoch": epoch,
            "batch": batch,
            "agent_state_dict": agent.state_dict(keep_vars=True),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        hf.save(checkpoint, ckpt_path)

    def _progress_update(
        self,
        progress,
        train_task,
        val_task,
        train_samples_so_far,
        train_len,
        val_samples_so_far,
        val_len,
        avg_metrics,
        avg_val_metrics,
        width,
        phase="train",
        console=None,
    ):
        """
        Update the progress bar with current training/validation status.
        """
        if phase == "test":
            val_label = "[bold green][Test]"
            val_metric_appendix = "test_"
        else:
            val_label = "[bold magenta][Validation]"
            val_metric_appendix = "val_"
        train_desc = (
            f"[bold blue][Training] {str(train_samples_so_far).rjust(width)}/"
            f"{str(train_len).ljust(width)}"
        )
        val_desc = (
            f"{val_label} {str(val_samples_so_far).rjust(width)}/"
            f"{str(val_len).ljust(width)}"
            if val_task is not None
            else ""
        )
        metrics_str = f"tot_cost: ${self.total_cost:.4f} "
        metrics_str += " - ".join(
            f"train_{k}: {v:.4f}" for k, v in self._ordered_metrics(avg_metrics)
        )
        if avg_val_metrics:
            metrics_str += " - " + " - ".join(
                f"{val_metric_appendix}{k}: {v:.4f}"
                for k, v in self._ordered_metrics(avg_val_metrics)
            )
        if phase == "train" and train_task is not None:
            progress.update(
                train_task,
                completed=train_samples_so_far,
                metrics=metrics_str,
                desc=train_desc,
            )
            if val_task is not None:
                progress.update(
                    val_task,
                    completed=val_samples_so_far,
                    metrics="",
                    desc=val_desc,
                )
        elif phase == "val" and train_task is not None and val_task is not None:
            progress.update(
                val_task,
                completed=val_samples_so_far,
                metrics="",
                desc=val_desc,
            )
            progress.update(
                train_task,
                metrics=metrics_str,
                desc=train_desc,
            )
        elif phase in ["val", "test"] and train_task is None and val_task is not None:
            progress.update(
                val_task,
                completed=val_samples_so_far,
                metrics=metrics_str,
                desc=val_desc,
            )

        # Notebook-specific: capture and print
        if self.buffer is not None and console is not None:
            with console.capture() as capture:
                progress.refresh()
            self.buffer[-1] = capture.get()
            clear_output(wait=True)
            print("\n".join(self.buffer))
        else:
            progress.refresh()

    def _progress_refresh(self, stop_event, progress, console):
        """Refresh the progress bar in a background thread."""
        while not stop_event.is_set():
            with console.capture() as capture:
                progress.refresh()
            self.buffer[-1] = capture.get()
            clear_output(wait=True)
            print("\n".join(self.buffer))
            stop_event.wait(0.5)

    def _start_refresher(self, progress, console):
        """Start a background thread to refresh the progress bar in a notebook."""
        refresher_stop = threading.Event()
        refresher_thread = threading.Thread(
            target=self._progress_refresh,
            args=(refresher_stop, progress, console),
            daemon=True,
        )
        refresher_thread.start()
        return refresher_stop, refresher_thread

    # TODO: Implement ckpt_path
    def fit(
        self,
        agent: cog.Module,
        train_dataloader: Optional[Union[TRAIN_DATALOADER, DataLoader]] = None,
        val_dataloader: Optional[Union[EVAL_DATALOADER, DataLoader]] = None,
        ckpt_path: Optional[_PATH] = None,
        llm_clients: Optional[List[BaseModel]] = [],
    ) -> None:
        r"""Runs the full optimization routine.

        Args:
            agent: AI agent (or flow) to fit.
            train_dataloader: An iterable or :class:`~afnio.util.data.DataLoader`
                specifying training samples.
            val_dataloader: An iterable or or :class:`~afnio.util.data.DataLoader`
                specifying validation samples.
            ckpt_path: Path of the checkpoint from which training is resumed. Otherwise,
                if there is no checkpoint file at the path, an exception is raised.
            llm_clients: Optional list of LLM clients used during training. If provided
                this list is used to calculate the total cost of training (in USD).

        Raises:
            TypeError: If ``agent`` is not :class:`~afnio.cognitive.modules.Module`.
        """
        if not isinstance(agent, cog.Module):
            raise TypeError(
                f"Expected agent to be an instance of cog.Module, but got {type(agent)}"
            )
        if train_dataloader is None:
            raise ValueError("train_dataloader must be provided.")
        if not isinstance(train_dataloader, (DataLoader, Iterable)):
            raise TypeError(
                f"Expected train_dataloader to be DataLoader or Iterable, "
                f"but got {type(train_dataloader)}"
            )
        if val_dataloader is not None and not isinstance(
            val_dataloader, (DataLoader, Iterable)
        ):
            raise TypeError(
                f"Expected val_dataloader to be DataLoader or Iterable, "
                f"but got {type(val_dataloader)}"
            )
        if ckpt_path is not None and not isinstance(ckpt_path, _PATH):
            raise TypeError(
                f"Expected ckpt_path to be str or Path, but got {type(ckpt_path)}"
            )

        if not hasattr(agent, "training_step"):
            raise AttributeError("Your agent must implement training_step().")
        if not hasattr(agent, "configure_optimizers"):
            raise AttributeError("Your agent must implement configure_optimizers().")
        if val_dataloader is not None and not hasattr(agent, "validation_step"):
            raise AttributeError(
                "Your agent must implement validation_step() "
                "when using a `val_dataloader`."
            )

        # If running in a notebook, clear the buffer if the cell has changed
        if self._is_notebook and self._in_notebook_cell_changed():
            self.buffer = []

        if self.enable_agent_summary:
            if self._is_notebook:
                self.buffer.append(str(agent) + "\n")
            else:
                print(str(agent) + "\n")

        optimizer = agent.configure_optimizers()
        # Set the optimizer(s) on the agent for manual optimization support
        agent._optimizers = optimizer

        console = Console()
        train_len = len(train_dataloader.dataset)
        val_len = len(val_dataloader.dataset) if val_dataloader is not None else 0
        width = (
            max(len(str(train_len)), len(str(val_len)))
            if val_len
            else len(str(train_len))
        )

        # TODO: Use also `self.min_epochs`, `self.max_steps`, `self.min_steps`,
        #       `self.max_time`, and `self.max_cost` in the training loop.
        for epoch_idx, epoch in enumerate(range(self.max_epochs)):
            metrics = defaultdict(list)
            val_metrics = defaultdict(list)
            train_start_time = time.time()
            training_step_times = []
            train_samples_so_far = 0
            val_samples_so_far = 0

            header = f"Epoch {epoch+1}/{self.max_epochs}"
            if self._is_notebook:
                self.buffer.append(header)
                self.buffer.append("")  # Placeholder for progress bar
            else:
                print(header)

            progress, train_task, val_task = None, None, None
            refresher_stop, refresher_thread = None, None
            if self.enable_progress_bar:
                progress = self._setup_progress("train", console)
                train_task = progress.add_task(
                    "", total=train_len, metrics="", desc="", visible=True
                )
                val_task = (
                    progress.add_task(
                        "", total=val_len, metrics="", desc="", visible=True
                    )
                    if val_dataloader is not None
                    else None
                )
                # Initial progress bar(s) render
                self._progress_update(
                    progress,
                    train_task,
                    val_task,
                    0,
                    train_len,
                    0,
                    val_len,
                    {},
                    {},
                    width,
                    console=console if self._is_notebook else None,
                )

                # Start refresher thread for notebook progress bar
                if self._is_notebook and self.enable_progress_bar and progress:
                    refresher_stop, refresher_thread = self._start_refresher(
                        progress, console
                    )

            try:
                # --- Training ---
                agent.train()
                for batch_idx, batch in enumerate(train_dataloader):
                    num_samples = get_batch_size(batch)
                    train_samples_so_far += num_samples
                    batch_metrics = self._run_training_step_with_retries(
                        agent.training_step,
                        batch,
                        batch_idx,
                        optimizer,
                        max_retries=3,
                        automatic=agent.automatic_optimization,
                    )

                    # Collect training metrics and clear LM models usage
                    train_elapsed = time.time() - train_start_time
                    training_step_times.append(train_elapsed)
                    self._collect_metrics(batch_metrics, metrics)
                    avg_metrics = self._average_metrics(metrics)
                    for client in llm_clients:
                        usage = client.get_usage()
                        self.total_cost += usage["cost"]["amount"]
                        client.clear_usage()

                    # Display progress
                    if self.enable_progress_bar and progress:
                        progress.update(
                            train_task, training_step_times=training_step_times
                        )
                        self._progress_update(
                            progress,
                            train_task,
                            val_task,
                            train_samples_so_far,
                            train_len,
                            0,
                            val_len,
                            avg_metrics,
                            {},
                            width,
                            phase="train",
                            console=console if self._is_notebook else None,
                        )

                    # Save checkpoint (batch)
                    if self.enable_checkpointing:
                        self._save_checkpoint(
                            agent, optimizer, epoch + 1, batch=batch_idx
                        )

                # Log the training results to Tellurio Studio
                with set_logger_level("afnio.tellurio.run", logging.WARNING):
                    for name, value in avg_metrics.items():
                        log(name=f"train_{name}", value=value, step=epoch_idx + 1)

                # --- Validation ---
                if val_dataloader is not None:
                    agent.eval()
                    with hf.no_grad():
                        for val_idx, val_batch in enumerate(val_dataloader):
                            num_val_samples = get_batch_size(val_batch)
                            val_samples_so_far += num_val_samples
                            val_step_out = self._run_fn_with_retries(
                                agent.validation_step,
                                val_batch,
                                val_idx,
                                max_retries=3,
                                step_name="validation_step",
                            )
                            val_batch_metrics = self._parse_step_metrics(
                                val_step_out, "validation_step"
                            )

                            # Collect validation metrics and clear LM models usage
                            self._collect_metrics(val_batch_metrics, val_metrics)
                            avg_val_metrics = self._average_metrics(val_metrics)
                            for client in llm_clients:
                                usage = client.get_usage()
                                self.total_cost += usage["cost"]["amount"]
                                client.clear_usage()

                            # Display progress
                            if self.enable_progress_bar and progress:
                                self._progress_update(
                                    progress,
                                    train_task,
                                    val_task,
                                    train_samples_so_far,
                                    train_len,
                                    val_samples_so_far,
                                    val_len,
                                    avg_metrics,
                                    avg_val_metrics,
                                    width,
                                    phase="val",
                                    console=console if self._is_notebook else None,
                                )

                    # Log the validation results to Tellurio Studio
                    with set_logger_level("afnio.tellurio.run", logging.WARNING):
                        for name, value in avg_val_metrics.items():
                            log(name=f"val_{name}", value=value, step=epoch_idx + 1)

                # Log total cost
                with set_logger_level("afnio.tellurio.run", logging.WARNING):
                    log(name="total_cost($)", value=self.total_cost, step=epoch_idx + 1)

                # Save checkpoint (epoch)
                if self.enable_checkpointing:
                    self._save_checkpoint(agent, optimizer, epoch + 1)

            finally:
                self._teardown_progress(progress, refresher_stop, refresher_thread)

    def validate(
        self,
        agent: cog.Module,
        val_dataloader: Optional[Union[EVAL_DATALOADER, DataLoader]] = None,
        llm_clients: Optional[List[BaseModel]] = [],
    ) -> dict:
        if val_dataloader is None:
            raise ValueError("val_dataloader must be provided.")
        if not isinstance(val_dataloader, (DataLoader, Iterable)):
            raise TypeError(
                f"Expected val_dataloader to be DataLoader or Iterable, "
                f"but got {type(val_dataloader)}"
            )
        if not hasattr(agent, "validation_step"):
            raise AttributeError("Your agent must implement validation_step().")

        # If running in a notebook, clear the buffer if the cell has changed
        if self._is_notebook and self._in_notebook_cell_changed():
            self.buffer = []

        if self._is_notebook:
            self.buffer.append("Validation")
            self.buffer.append("")  # Placeholder for progress bar
        else:
            print("Validation")

        console = Console()
        val_len = len(val_dataloader.dataset)
        width = len(str(val_len))
        val_metrics = defaultdict(list)
        val_samples_so_far = 0

        progress, val_task = None, None
        refresher_stop, refresher_thread = None, None
        if self.enable_progress_bar:
            progress = self._setup_progress("val", console)
            val_task = progress.add_task(
                "", total=val_len, metrics="", desc="", visible=True
            )
            # Initial progress bar render
            self._progress_update(
                progress,
                None,
                val_task,
                0,
                0,
                0,
                val_len,
                {},
                {},
                width,
                phase="val",
                console=console if self._is_notebook else None,
            )

            # Start refresher thread for notebook progress bar
            if self._is_notebook and self.enable_progress_bar and progress:
                refresher_stop, refresher_thread = self._start_refresher(
                    progress, console
                )

        try:
            agent.eval()
            with hf.no_grad():
                for val_idx, val_batch in enumerate(val_dataloader):
                    num_val_samples = get_batch_size(val_batch)
                    val_samples_so_far += num_val_samples
                    val_step_out = self._run_fn_with_retries(
                        agent.validation_step,
                        val_batch,
                        val_idx,
                        max_retries=3,
                        step_name="validation_step",
                    )
                    val_batch_metrics = self._parse_step_metrics(
                        val_step_out, "validation_step"
                    )

                    # Collect validation metrics and clear LM models usage
                    self._collect_metrics(val_batch_metrics, val_metrics)
                    avg_val_metrics = self._average_metrics(val_metrics)
                    for client in llm_clients:
                        usage = client.get_usage()
                        self.total_cost += usage["cost"]["amount"]
                        client.clear_usage()

                    # Display progress
                    if self.enable_progress_bar and progress:
                        self._progress_update(
                            progress,
                            None,
                            val_task,
                            0,
                            0,
                            val_samples_so_far,
                            val_len,
                            {},
                            avg_val_metrics,
                            width,
                            phase="val",
                            console=console if self._is_notebook else None,
                        )
        finally:
            self._teardown_progress(progress, refresher_stop, refresher_thread)

        # Log the validation results to Tellurio Studio
        with set_logger_level("afnio.tellurio.run", logging.WARNING):
            for name, value in avg_val_metrics.items():
                log(name=f"val_{name}", value=value)

            log(name="total_cost($)", value=self.total_cost)

        # Return averaged metrics
        return avg_val_metrics

    def test(
        self,
        agent: cog.Module,
        test_dataloader: Optional[Union[EVAL_DATALOADER, DataLoader]] = None,
        llm_clients: Optional[List[BaseModel]] = [],
    ) -> dict:
        if test_dataloader is None:
            raise ValueError("test_dataloader must be provided.")
        if not isinstance(test_dataloader, (DataLoader, Iterable)):
            raise TypeError(
                f"Expected test_dataloader to be DataLoader or Iterable, "
                f"but got {type(test_dataloader)}"
            )
        if not hasattr(agent, "test_step"):
            raise AttributeError("Your agent must implement test_step().")

        # If running in a notebook, clear the buffer if the cell has changed
        if self._is_notebook and self._in_notebook_cell_changed():
            self.buffer = []

        if self._is_notebook:
            self.buffer.append("Testing")
            self.buffer.append("")  # Placeholder for progress bar
        else:
            print("Testing")

        console = Console()
        test_len = len(test_dataloader.dataset)
        width = len(str(test_len))
        test_metrics = defaultdict(list)
        test_samples_so_far = 0

        progress, test_task = None, None
        refresher_stop, refresher_thread = None, None
        if self.enable_progress_bar:
            progress = self._setup_progress("test", console)
            test_task = progress.add_task(
                "", total=test_len, metrics="", desc="", visible=True
            )
            # Initial progress bar render
            self._progress_update(
                progress,
                None,
                test_task,
                0,
                0,
                0,
                test_len,
                {},
                {},
                width,
                phase="test",
                console=console if self._is_notebook else None,
            )

            # Start refresher thread for notebook progress bar
            if self._is_notebook and self.enable_progress_bar and progress:
                refresher_stop, refresher_thread = self._start_refresher(
                    progress, console
                )

        try:
            agent.eval()
            with hf.no_grad():
                for test_idx, test_batch in enumerate(test_dataloader):
                    num_test_samples = get_batch_size(test_batch)
                    test_samples_so_far += num_test_samples
                    test_step_out = self._run_fn_with_retries(
                        agent.test_step,
                        test_batch,
                        test_idx,
                        max_retries=3,
                        step_name="test_step",
                    )
                    test_batch_metrics = self._parse_step_metrics(
                        test_step_out, "test_step"
                    )

                    # Collect test metrics and clear LM models usage
                    self._collect_metrics(test_batch_metrics, test_metrics)
                    avg_test_metrics = self._average_metrics(test_metrics)
                    for client in llm_clients:
                        usage = client.get_usage()
                        self.total_cost += usage["cost"]["amount"]
                        client.clear_usage()

                    # Display progress
                    if self.enable_progress_bar and progress:
                        self._progress_update(
                            progress,
                            None,
                            test_task,
                            0,
                            0,
                            test_samples_so_far,
                            test_len,
                            {},
                            avg_test_metrics,
                            width,
                            phase="test",
                            console=console if self._is_notebook else None,
                        )
        finally:
            self._teardown_progress(progress, refresher_stop, refresher_thread)

        # Log the test results to Tellurio Studio
        with set_logger_level("afnio.tellurio.run", logging.WARNING):
            for name, value in avg_test_metrics.items():
                log(name=f"test_{name}", value=value)

            log(name="total_cost($)", value=self.total_cost)

        # Return averaged metrics
        return avg_test_metrics

    # TODO: Finalize this method
    def predict(self) -> None:
        pass


def get_batch_size(batch):
    """
    Returns the number of samples in a batch, supporting all DataLoader output formats:
    - If batch is a dict: returns the length of the first value if it's a
      list/tuple/Variable, else 1.
    - If batch is a tuple or list: recursively checks the first element.
    - If batch is a Variable: returns the length of its .data attribute if possible,
      else 1.
    - Otherwise: returns 1 (single sample).
    Raises:
        ValueError: If the batch is empty (size 0).
    """
    if isinstance(batch, dict):
        if not batch:
            raise ValueError("Batch is empty (size 0), which is not allowed.")
        first_key = next(iter(batch))
        first_value = batch[first_key]
        if isinstance(first_value, (list, tuple)):
            size = len(first_value)
        elif isinstance(first_value, Variable):
            try:
                size = len(first_value.data)
            except TypeError:
                size = 1
        else:
            size = 1
    elif isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("Batch is empty (size 0), which is not allowed.")
        size = get_batch_size(batch[0])
    elif isinstance(batch, Variable):
        try:
            size = len(batch.data)
        except TypeError:
            size = 1
    else:
        size = 1

    if size == 0:
        raise ValueError("Batch is empty (size 0), which is not allowed.")
    return size
