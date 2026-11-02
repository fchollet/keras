import os
import re
import warnings

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.monitor_callback import MonitorCallback
from keras.src.utils import file_utils
from keras.src.utils import io_utils


@keras_export("keras.callbacks.ModelCheckpoint")
class ModelCheckpoint(MonitorCallback):
    """Callback to save the Keras model or model weights at some frequency.

    `ModelCheckpoint` callback is used in conjunction with training using
    `model.fit()` to save a model or weights (in a checkpoint file) at some
    interval, so the model or weights can be loaded later to continue the
    training from the state saved.

    A few options this callback provides include:

    - Whether to only keep the model that has achieved the "best performance" so
      far, or whether to save the model at the end of every epoch regardless of
      performance.
    - Definition of "best"; which quantity to monitor and whether it should be
      maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving
      at the end of every epoch, or after a fixed number of training batches.
    - Whether only weights are saved, or the whole model is saved.

    Example:

    ```python
    # Define a learning rate schedule
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True,
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model is saved at the end of every epoch, if it's the best seen so far.
    model.fit(
        x_train, y_train, epochs=10, callbacks=[model_checkpoint_callback]
    )

    # The model (that are considered the best) can be loaded as -
    keras.models.load_model(checkpoint_filepath)

    # Alternatively, one could checkpoint just the model weights as -
    checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(
        x_train, y_train, epochs=10, callbacks=[model_checkpoint_callback]
    )

    # The model weights (that are considered the best) can be loaded as -
    model.load_weights(checkpoint_filepath)
    ```

    Resuming training from weight-only checkpoints:

    When using `save_weights_only=True`, the weights file includes the state
    of the optimizer (including iteration count and learning rate state)
    if the model is compiled at the time of saving.

    To correctly resume training and restore the optimizer state (e.g., to
    continue a learning rate schedule without resetting it), you must
    **compile the model before loading the weights**.

    ```python
    # Define a learning rate schedule
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True,
    )

    # 1. Create a fresh model instance
    model = get_model()

    # 2. Compile the model *before* loading weights
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 3. Load weights (optimizer state is restored automatically)
    model.load_weights(checkpoint_filepath)

    # 4. Continue training
    model.fit(x_train, y_train, epochs=10)
    ```

    Args:
        filepath: string or `PathLike`, path to save the model file.
            `filepath` can contain named formatting options,
            which will be filled the value of `epoch` and keys in `logs`
            (passed in `on_epoch_end`).
            The `filepath` name needs to end with `".weights.h5"` when
            `save_weights_only=True` or should end with `".keras"` or `".h5"`
            when checkpoint saving the whole model (default).
            For example:
            if `filepath` is `"{epoch:02d}-{val_loss:.2f}.keras"` or
            "{epoch:02d}-{val_loss:.2f}.weights.h5"`, then the model
            checkpoints will be saved with the epoch number and the validation
            loss in the filename. The directory of the filepath
            should not be reused by any other callbacks to avoid conflicts.
        monitor: The metric name to monitor. Typically the metrics are set by
            the `Model.compile` method. Note:
            * Prefix the name with `"val_"` to monitor validation metrics.
            * Use `"loss"` or `"val_loss"` to monitor the model's total loss.
            * If you specify metrics as strings, like `"accuracy"`, pass the
                same string (with or without the `"val_"` prefix).
            * If you pass `metrics.Metric` objects, `monitor` should be set to
                `metric.name`
            * If you're not sure about the metric names you can check the
                contents of the `history.history` dictionary returned by
                `history = model.fit()`
            * Multi-output models set additional prefixes on the metric names.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only: if `save_best_only=True`, it only saves when the model
            is considered the "best" and the latest best model according to the
            quantity monitored will not be overwritten. If `filepath` doesn't
            contain formatting options like `{epoch}` then `filepath` will be
            overwritten by each new better model.
        mode: one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `"max"`, for `val_loss` this should be
            `"min"`, etc. In `"auto"` mode, the direction is automatically
            inferred from the name of the monitored quantity.
        save_weights_only: if `True`, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model is
            saved (`model.save(filepath)`).
        save_freq: `"epoch"` or integer. When using `"epoch"`, the callback
            saves the model after each epoch. When using integer, the callback
            saves the model at end of this many batches. If the `Model` is
            compiled with `steps_per_execution=N`, then the saving criteria will
            be checked every Nth batch. Note that if the saving isn't aligned to
            epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.
        initial_value_threshold: Floating point initial "best" value of the
            metric to be monitored. Only applies if `save_best_value=True`. Only
            overwrites the model weights already saved if the performance of
            current model is better than this value.
    """

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    ):
        super().__init__(monitor, mode, initial_value_threshold)
        self.verbose = verbose
        self.filepath = file_utils.path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = None

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

        if save_weights_only:
            if not self.filepath.endswith(".weights.h5"):
                raise ValueError(
                    "When using `save_weights_only=True` in `ModelCheckpoint`"
                    ", the filepath provided must end in `.weights.h5` "
                    "(Keras weights format). Received: "
                    f"filepath={self.filepath}"
                )
        else:
            if not any(
                self.filepath.endswith(ext) for ext in (".keras", ".h5")
            ):
                raise ValueError(
                    "The filepath provided must end in `.keras` "
                    "(Keras model format). Received: "
                    f"filepath={self.filepath}"
                )

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            # Delay setup until the model's metrics are all built
            self._set_monitor_op()

        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == "epoch":
            return False
        if self._last_batch_seen is None or batch <= self._last_batch_seen:
            # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _should_save_model(self, epoch, batch, logs, filepath):
        """Determines whether the model should be saved.

        The model should be saved in the following cases:

        - self.save_best_only is False
        - self.save_best_only is True and `monitor` is a numpy array or
          backend tensor (falls back to `save_best_only=False`)
        - self.save_best_only is True and `self.monitor_op(current, self.best)`
          evaluates to True.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or
                `on_epoch_end`.
            filepath: the path where the model would be saved
        """
        logs = logs or {}
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn(
                    f"Can save best model only with {self.monitor} available.",
                    stacklevel=2,
                )
                return True
            elif (
                isinstance(current, np.ndarray)
                or backend.ops.is_tensor(current)
            ) and len(current.shape) > 0:
                warnings.warn(
                    "Can save best model only when `monitor` is "
                    f"a scalar value. Received: {current}. "
                    "Falling back to `save_best_only=False`."
                )
                return True
            else:
                best_str = "None" if self.best is None else f"{self.best:.5f}"
                if self._is_improvement(current, self.best):
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f"\nEpoch {epoch + 1}: {self.monitor} "
                            f"improved from {best_str} to {current:.5f}, "
                            f"saving model to {filepath}"
                        )
                    self.best = current
                    return True
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f"\nEpoch {epoch + 1}: "
                            f"{self.monitor} did not improve from {best_str}"
                        )
                    return False
        else:
            if self.verbose > 0:
                io_utils.print_msg(
                    f"\nEpoch {epoch + 1}: saving model to {filepath}"
                )
            return True

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        filepath = self._get_file_path(epoch, batch, logs)

        try:
            if self._should_save_model(epoch, batch, logs, filepath):
                # Create host directory if it doesn't exist.
                dirname = os.path.dirname(filepath)
                if dirname and not file_utils.exists(dirname):
                    file_utils.makedirs(dirname)

                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                if self.verbose > 0:
                    io_utils.print_msg(
                        f"\nEpoch {epoch + 1}: "
                        f"finished saving model to {filepath}"
                    )
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ModelCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of
            # `e.args[0]`.
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: f{filepath}"
                )
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return file_path

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        return file_utils.exists(filepath)

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.

        In the rare case where there are more than one pattern-matching file
        having the same modified time that is most recent among all, return the
        filepath that is largest (by `>` operator, lexicographically using the
        numeric equivalents). This provides a tie-breaker when multiple files
        are most recent. Note that a larger `filepath` can sometimes indicate a
        later time of modification (for instance, when epoch/batch is used as
        formatting option), but not necessarily (when accuracy or loss is used).
        The tie-breaker is put in the logic as best effort to return the most
        recent, and to avoid nondeterministic result.

        Modified time of a file is obtained with `file_utils.getmtime()`.

        This utility function is best demonstrated via an example:

        ```python
        file_pattern = 'batch{batch:02d}epoch{epoch:02d}.keras'
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name) for file_name in
            ['batch03epoch02.keras',
             'batch02epoch02.keras', 'batch01epoch01.keras']
        ]
        for file_path in file_paths:
            # Write something to each of the files
            ...
        self.assertEqual(
            _get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1])
        ```

        Args:
            pattern: The file pattern that may optionally contain python
                placeholder such as `{epoch:02d}`.

        Returns:
            The most recently modified file's full filepath matching `pattern`.
            If `pattern` does not contain any placeholder, this returns the
            filepath that exactly matches `pattern`. Returns `None` if no match
            is found.
        """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = f"^{re.sub(r'{.*}', r'.*', base_name)}$"

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if file_utils.exists(dir_name):
            for file_name in file_utils.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = file_utils.join(dir_name, file_name)
                    mod_time = file_utils.getmtime(file_path)
                    if (
                        file_path_with_largest_file_name is None
                        or file_path > file_path_with_largest_file_name
                    ):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found,
                        # reset the counter for the number of files with latest
                        # modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the
                        # most recent, increment the counter for the number of
                        # files with latest modified time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time,
            # return the file path with the largest file name.
            return file_path_with_largest_file_name
