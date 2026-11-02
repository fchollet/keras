import contextlib
import functools
import warnings

import tensorflow as tf
from tensorflow.python.eager import context as tf_context

from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.losses import loss as loss_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs


class Trainer(base_trainer.BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Specifies how many steps of the step_per_execution loop to unroll.
        # Increasing this value can reduce kernel launch overhead,
        # but will increase memory usage and compilation time.
        self.unrolled_steps_per_execution = 1

        # Model must be created under scope of DistStrat it will be trained
        # with.
        if tf.distribute.has_strategy():
            self._distribute_strategy = tf.distribute.get_strategy()
        else:
            self._distribute_strategy = None

    @property
    def distribute_strategy(self):
        return self._distribute_strategy or tf.distribute.get_strategy()

    @property
    def distribute_reduction_method(self):
        return self._distribute_reduction_method or "auto"

    @distribute_reduction_method.setter
    def distribute_reduction_method(self, value):
        self._distribute_reduction_method = value

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self._compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
            self._loss_tracker.update_state(
                loss_module.unscale_loss_for_distribution(loss),
                sample_weight=tf.shape(
                    next(i for i in tree.flatten(x) if i is not None)
                )[0],
            )
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss_module.unscale_loss_for_distribution(loss),
            sample_weight=tf.shape(
                next(i for i in tree.flatten(x) if i is not None)
            )[0],
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def _autoconvert_optionals(self, step_func):
        # Wrapper converting (nested) TF Optional in input data to None
        @functools.wraps(step_func)
        def wrapper(data):
            converted_data = tree.map_structure(
                lambda i: (
                    None if isinstance(i, tf.experimental.Optional) else i
                ),
                data,
            )
            result = step_func(converted_data)
            return result

        return wrapper

    def _make_function(self, step_function):
        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            outputs = self.distribute_strategy.run(step_function, args=(data,))
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction="auto",
            )
            return outputs

        if not self.run_eagerly:
            one_step_on_data = tf.function(
                one_step_on_data,
                reduce_retracing=True,
                jit_compile=self.jit_compile,
            )
        one_step_on_data = self._autoconvert_optionals(one_step_on_data)

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_iterator(iterator):
            # the spec is set lazily during the tracing of `tf.while_loop`
            empty_outputs = tf.experimental.Optional.empty(None)

            def cond(execution_step, optional_outputs, next_optional_inputs):
                return tf.logical_and(
                    tf.less(execution_step, self.steps_per_execution),
                    next_optional_inputs.has_value(),
                )

            def inner_body(
                execution_step, optional_outputs, next_optional_inputs
            ):
                def has_next():
                    next_optional_outputs = tf.experimental.Optional.from_value(
                        one_step_on_data(next_optional_inputs.get_value())
                    )
                    empty_outputs._element_spec = (
                        next_optional_outputs.element_spec
                    )
                    return next_optional_outputs

                def no_has_next():
                    optional_outputs._element_spec = empty_outputs._element_spec
                    return optional_outputs

                next_optional_outputs = tf.cond(
                    tf.logical_and(
                        tf.less(execution_step, self.steps_per_execution),
                        next_optional_inputs.has_value(),
                    ),
                    has_next,
                    no_has_next,
                )

                return (
                    execution_step + 1,
                    next_optional_outputs,
                    # We don't want to iterate if we have reached
                    # `steps_per_execution` steps
                    tf.cond(
                        tf.less(execution_step + 1, self.steps_per_execution),
                        lambda: iterator.get_next_as_optional(),
                        lambda: next_optional_inputs,
                    ),
                )

            def body(execution_step, optional_outputs, next_optional_inputs):
                for _ in range(
                    min(
                        self.unrolled_steps_per_execution,
                        self.steps_per_execution,
                    )
                ):
                    execution_step, optional_outputs, next_optional_inputs = (
                        inner_body(
                            execution_step,
                            optional_outputs,
                            next_optional_inputs,
                        )
                    )

                return (execution_step, optional_outputs, next_optional_inputs)

            execution_step = tf.constant(0)
            next_optional_inputs = iterator.get_next_as_optional()

            # Run the while loop
            _, final_optional_outputs, _ = tf.while_loop(
                cond,
                body,
                loop_vars=[execution_step, empty_outputs, next_optional_inputs],
            )
            final_optional_outputs._element_spec = empty_outputs.element_spec
            return final_optional_outputs

        if not self.run_eagerly:
            multi_step_on_iterator = tf.function(
                multi_step_on_iterator, reduce_retracing=True
            )

        def function(iterator):
            if isinstance(
                iterator, (tf.data.Iterator, tf.distribute.DistributedIterator)
            ):
                if self.steps_per_execution == 1:
                    next_optional_inputs = iterator.get_next_as_optional()
                    if not next_optional_inputs.has_value():
                        raise StopIteration
                    return one_step_on_data(next_optional_inputs.get_value())

                opt_outputs = multi_step_on_iterator(iterator)
                if not opt_outputs.has_value():
                    raise StopIteration
                return opt_outputs.get_value()
            else:
                for _, data in zip(
                    range(self.steps_per_execution),
                    iterator,
                ):
                    outputs = one_step_on_data(data)
                return outputs

        return function

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function
        self.train_function = self._make_function(self.train_step)

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function
        self.test_function = self._make_function(self.test_step)

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(
                one_step_on_data, reduce_retracing=True, jit_compile=True
            )
        one_step_on_data = self._autoconvert_optionals(one_step_on_data)

        @tf.autograph.experimental.do_not_convert
        def one_step_on_data_distributed(data):
            data = data[0]
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction="concat",
            )
            return outputs

        @tf.autograph.experimental.do_not_convert
        def multi_step_on_data(data):
            if len(data) == 1:
                return one_step_on_data_distributed(data[:1])
            all_outputs = [one_step_on_data_distributed([d]) for d in data]
            return tree.map_structure(
                lambda *args: concat(list(args), axis=0), *all_outputs
            )

        if self.steps_per_execution > 1:
            predict_function = multi_step_on_data
        else:
            predict_function = one_step_on_data_distributed

        if not self.run_eagerly:
            predict_function = tf.function(
                predict_function, reduce_retracing=True
            )

        self.predict_function = predict_function

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        self._assert_compile_called("fit")
        # Possibly cap epochs for debugging runs.
        max_epochs = config.max_epochs()
        if max_epochs and max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % max_epochs)
            epochs = max_epochs

        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        self._maybe_symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = {}
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for begin_step, end_step, iterator in epoch_iterator:
                    callbacks.on_train_batch_begin(begin_step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(end_step, logs)
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        distribute_strategy=self.distribute_strategy,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    f"val_{name}": val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        self._assert_compile_called("evaluate")
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TFEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                distribute_strategy=self.distribute_strategy,
                steps_per_execution=self.steps_per_execution,
            )

        self._maybe_symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(begin_step)
                logs = self.test_function(iterator)
                callbacks.on_test_batch_end(end_step, logs)
                if self.stop_evaluating:
                    break
        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TFEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            distribute_strategy=self.distribute_strategy,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        def get_data(iterator):
            """Returns data for the next execution."""
            data = []
            for _ in range(self.steps_per_execution):
                try:
                    single_step_data = next(iterator)
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    if hasattr(data, "__len__") and len(data) > 0:
                        # Suppress the error when still have remaining data.
                        return data
                    else:
                        # Re-raise the error for
                        # EpochIterator.catch_stop_iteration() to catch when
                        # no data left.
                        raise e
                data.append(single_step_data)
            return data

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_predict_batch_begin(begin_step)
                data = get_data(iterator)
                batch_outputs = self.predict_function(data)
                outputs = append_to_outputs(batch_outputs, outputs)
                callbacks.on_predict_batch_end(
                    end_step, {"outputs": batch_outputs}
                )
                if self.stop_predicting:
                    break
        callbacks.on_predict_end()
        outputs = tree.map_structure_up_to(
            batch_outputs, potentially_ragged_concat, outputs
        )
        return tree.map_structure(convert_to_np_if_not_ragged, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        # Maybe build model
        self._maybe_symbolic_build(data_batch=(x, y, sample_weight))
        self.make_train_function()
        self.reset_metrics()

        def data():
            yield (x, y, sample_weight)

        logs = self.train_function(data())
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        def data():
            yield (x, y, sample_weight)

        # Maybe build model
        self._maybe_symbolic_build(data_batch=(x, y, sample_weight))
        self.make_test_function()
        self.reset_metrics()

        logs = self.test_function(data())
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            convert_to_np_if_not_ragged, batch_outputs
        )
        return batch_outputs

    # Backwards compatibility shims.
    @property
    def compiled_metrics(self):
        class DeprecatedCompiledMetric:
            def update_state(_, y, y_pred, sample_weight=None):
                return self._compiled_metrics_update_state(
                    y, y_pred, sample_weight=sample_weight
                )

        return DeprecatedCompiledMetric()

    def _compiled_metrics_update_state(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.compiled_metrics()` is deprecated. "
            "Instead, use e.g.:\n"
            "```\n"
            "for metric in self.metrics:\n"
            "    metric.update_state(y, y_pred)\n"
            "```\n",
            stacklevel=2,
        )
        for metric in self.metrics:
            if isinstance(metric, metrics_module.Mean):
                metric.update_state(y_pred, sample_weight=sample_weight)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

    def compiled_loss(
        self, y, y_pred, sample_weight=None, regularization_losses=None
    ):
        warnings.warn(
            "`model.compiled_loss()` is deprecated. Instead, use "
            "`model.compute_loss(x, y, y_pred, sample_weight, training)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )

    def loss(self, y, y_pred, sample_weight=None):
        warnings.warn(
            "`model.loss()` is deprecated. Instead, use "
            "`model.compute_loss(x, y, y_pred, sample_weight, training)`.",
        )
        return self.compute_loss(
            x=None, y=y, y_pred=y_pred, sample_weight=sample_weight
        )

    def _maybe_symbolic_build(self, iterator=None, data_batch=None):
        # Only symbolic build when distribute strategy is created in tf trainer
        if self._distribute_strategy is None:
            # When no distribution strategy is set, defer building
            # to when the train/test/predict function gets traced.
            # This maximizes backwards compatibility.
            return

        # Unlike jax/torch iterator, tf iterator returns an iterator instead
        # of data batch in `iterator`.
        if iterator is not None:
            for _, _, it in iterator:
                maybe_distributed_data_batch = next(it)
                has_distributed_values = tree.map_structure(
                    lambda x: isinstance(x, tf.distribute.DistributedValues),
                    maybe_distributed_data_batch,
                )
                if all(tree.flatten(has_distributed_values)):
                    data_batch = self.distribute_strategy.reduce(
                        "MEAN",
                        maybe_distributed_data_batch,
                        axis=None,
                    )
                else:
                    data_batch = maybe_distributed_data_batch
                break
        with self.distribute_strategy.scope():
            self._symbolic_build(data_batch=data_batch)

    def _aggregate_additional_loss(self, loss):
        loss = super()._aggregate_additional_loss(loss)
        return loss_module.scale_loss_for_distribution(loss)


class TFEpochIterator(EpochIterator):
    def __init__(self, distribute_strategy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribute_strategy = distribute_strategy
        dataset = self.data_adapter.get_tf_dataset()
        if not isinstance(dataset, tf.distribute.DistributedDataset):
            dataset = self._distribute_strategy.experimental_distribute_dataset(
                dataset
            )
        self._distributed_dataset = dataset

    def _get_iterator(self):
        return self._distributed_dataset

    def tf_sync(self):
        tf_context.async_wait()

    def __next__(self):
        return next(self._epoch_iterator)

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        with super().catch_stop_iteration():
            try:
                yield
                self.tf_sync()
            except tf.errors.OutOfRangeError:
                raise StopIteration


def reduce_per_replica(values, strategy, reduction):
    """Attempt to reduce the structure `values` to single values.

    Given `values` (a `tf.Tensor` or a `PerReplica` structure),
    which represents the values across all the replicas, `reduce_per_replica`
    attempts to "reduce" those values and returns the corresponding structure
    that represents only single values.

    Currently, `reduce_per_replica` is only used for reducing the metric results
    from `tf.distribute.Strategy.run()`. Depending on the underlying
    `Strategy` implementation, `values` may be a `PerReplica` object,
    which can be thought of as a collection of values across the replicas,
    or a `tf.Tensor`, if the strategy has already conducted the reduction
    for the downstream library.

    There are five possible outcomes of reduction:

    1) if the `values` is a structure of simple `tf.Tensor`s, meaning that
       reduction is not actually needed, `reduce_per_replica` returns the
       structure as-is.
    2) else, if `reduction="auto"`, then the best reduction strategy is
       chosen based on the current environment. This should only be used
       for training cases (`fit()`).
    3) else, if `reduction="first"`, then `reduce_per_replica`
       returns the values of the first replica. This is used in the case of
       training and evaluation, where `values` is expected to hold the same
       value across the replicas as a result of `Strategy`'s synchronization
       across the replicas.
       `reduce_per_replica` does not synchronize the values.
    4) else, if `reduction="sum"`, then `reduce_per_replica` returns the sum
       of values for all replicas. This may be used in the custom training loop
       case, where each replica contain different values which are not
       synchronized.
    5) else, if `reduction="concat"`, then `reduce_per_replica`
       returns the concatenation of the values across the replicas, along the
       axis of dimension 0. This is used in the inference case (`predict()`).

    Args:
        values: Structure of `PerReplica` objects or `tf.Tensor`s.
            `tf.Tensor`s are returned as-is.
        strategy: `tf.distribute.Strategy` object.
        reduction: One of `"auto"`, `"first"`, `"concat"`, `"mean"`, or `"sum"`.
            `"auto"` will select `"first"` when used under a TPUStrategy, or
            `"mean"` otherwise.

    Returns:
        Structure of `Tensor`s, representing the result of reduction.
    """

    if reduction == "auto":
        if isinstance(strategy, tf.distribute.TPUStrategy):
            reduction = "first"
        else:
            reduction = "mean"

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if _collective_all_reduce_multi_worker(strategy):
            if reduction == "concat":
                return _multi_worker_concat(v, strategy)
            elif reduction == "sum":
                return strategy.reduce("SUM", v)
            elif reduction == "mean":
                return strategy.reduce("MEAN", v, axis=0)

        if not _is_per_replica_instance(v):
            return v
        elif reduction == "first":
            return strategy.experimental_local_results(v)[0]
        elif reduction == "concat":
            if _is_tpu_multi_host(strategy):
                return _tpu_multi_host_concat(v, strategy)
            else:
                return concat(strategy.experimental_local_results(v))
        elif reduction == "sum":
            return tf.reduce_sum(strategy.experimental_local_results(v))
        elif reduction == "mean":
            return tf.reduce_mean(
                strategy.experimental_local_results(v), axis=0
            )
        else:
            raise ValueError(
                "`reduction` must be one of "
                '"first", "concat", "mean", "sum", or "auto". '
                f"Received: reduction={reduction}."
            )

    return tree.map_structure(_reduce, values)


def _multi_worker_concat(v, strategy):
    """Order PerReplica objects for CollectiveAllReduceStrategy and concat."""
    replicas = strategy.gather(v, axis=0)
    # v might not have the same shape on different replicas
    if _is_per_replica_instance(v):
        shapes = tf.concat(
            [
                tf.expand_dims(tf.shape(single_value)[0], axis=0)
                for single_value in v.values
            ],
            axis=0,
        )
        all_shapes = strategy.gather(shapes, axis=0)
    else:
        # v is a tensor. This may happen when, say, we have 2x1 multi-worker.
        all_shapes = strategy.gather(
            tf.expand_dims(tf.shape(v)[0], axis=0), axis=0
        )

    replicas = tf.split(
        replicas,
        num_or_size_splits=all_shapes,
        num=strategy.num_replicas_in_sync,
    )
    ordered_replicas = []
    num_replicas_per_worker = len(strategy.extended.worker_devices)
    for replica_id in range(num_replicas_per_worker):
        ordered_replicas += replicas[replica_id::num_replicas_per_worker]
    return concat(ordered_replicas)


def concat(tensors, axis=0):
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], tf.SparseTensor):
        return tf.sparse.concat(axis=axis, sp_inputs=tensors)
    elif _is_scalar(tensors[0]):
        return tf.stack(tensors, axis=axis)
    else:
        return tf.concat(tensors, axis=axis)


def _tpu_multi_host_concat(v, strategy):
    """Correctly order TPU PerReplica objects."""
    replicas = strategy.experimental_local_results(v)
    # When distributed datasets are created from Tensors / NumPy,
    # TPUStrategy.experimental_distribute_dataset shards data in
    # (Replica, Host) order, and TPUStrategy.experimental_local_results returns
    # it in (Host, Replica) order.
    num_replicas_per_host = strategy.extended.num_replicas_per_host
    ordered_replicas = []
    for replica_id in range(num_replicas_per_host):
        ordered_replicas += replicas[replica_id::num_replicas_per_host]
    return concat(ordered_replicas)


def _collective_all_reduce_multi_worker(strategy):
    return (
        isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy)
    ) and strategy.extended._in_multi_worker_mode()


def _is_per_replica_instance(obj):
    return isinstance(obj, tf.distribute.DistributedValues) and isinstance(
        obj, tf.__internal__.CompositeTensor
    )


def _is_scalar(x):
    return isinstance(x, (tf.Tensor, tf.Variable)) and x.shape.rank == 0


def _is_tpu_multi_host(strategy):
    return _is_tpu_strategy(strategy) and strategy.extended.num_hosts > 1


def _is_tpu_strategy(strategy):
    return _is_tpu_strategy_class(strategy.__class__)


def _is_tpu_strategy_class(clz):
    def is_tpu_strat(k):
        return k.__name__.startswith("TPUStrategy")

    if is_tpu_strat(clz):
        return True
    return any(map(_is_tpu_strategy_class, clz.__bases__))


def convert_to_np_if_not_ragged(x):
    if isinstance(x, tf.RaggedTensor):
        return x
    elif isinstance(x, tf.SparseTensor):
        return x
    return x.numpy()


def potentially_ragged_concat(tensors):
    """Concats `Tensor`s along their first dimension.

    Args:
        tensors: List of `Tensor`s.

    Returns:
        Concatenation of the inputs along the first dimension -- of type
        `np.ndarray` if all input shapes are compatible, or `tf.RaggedTensor`
        if not.
    """
    if len(tensors) == 1:
        return tensors[0]
    elif isinstance(tensors[0], tf.SparseTensor):
        return tf.sparse.concat(axis=0, sp_inputs=tensors)
    elif isinstance(tensors[0], tf.RaggedTensor):
        return tf.concat(tensors, axis=0)

    non_batch_shapes = tf.stack([tf.shape(tensor)[1:] for tensor in tensors])
    constant_dims = tf.math.reduce_all(
        non_batch_shapes == non_batch_shapes[:1], axis=0
    )
    if tf.math.reduce_all(constant_dims).numpy().item():
        # All non-batch dims are constant
        if _is_scalar(tensors[0]):
            return tf.stack(tensors, axis=0)
        else:
            return tf.concat(tensors, axis=0)

    # First, identify constant inner dimensions by finding the
    # rightmost dimension that is not constant
    constant_inner_dimensions = (
        constant_dims.numpy().tolist()[::-1].index(False)
    )
    # If there are constant inner dimensions, define a constant inner shape
    if constant_inner_dimensions == 0:
        constant_inner_shape = None
    else:
        constant_inner_shape = tensors[0].shape[-constant_inner_dimensions:]
    return tf.ragged.constant(
        [tensor.numpy() for tensor in tensors], inner_shape=constant_inner_shape
    ).merge_dims(0, 1)
