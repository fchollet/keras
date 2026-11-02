import itertools
import warnings
from functools import partial

import jax
import numpy as np

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.backend.config import is_nnx_enabled
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs

if is_nnx_enabled():
    from flax import nnx

    jit = nnx.jit
else:
    jit = jax.jit


def _concatenate_outputs(outputs):
    if not outputs:
        return
    if len(outputs) == 1:
        return outputs[0]
    return tree.map_structure(
        lambda *args: jax.numpy.concatenate(args, axis=0),
        *outputs,
    )


def _concat_leading_dims(x):
    if x is None:
        return None
    shape = (x.shape[0] * x.shape[1],) + x.shape[2:]
    return jax.numpy.reshape(x, shape)


def _build_multi_step_scan(
    step_function, steps_per_execution, concatenate_outputs
):
    def multi_step_scan(state, super_batch):
        def scan_body(current_state, step_data):
            outputs, next_state = step_function(current_state, step_data)
            return next_state, outputs

        final_state, stacked_outputs = jax.lax.scan(
            scan_body,
            state,
            super_batch,
            length=steps_per_execution,
        )
        if concatenate_outputs:
            outputs = tree.map_structure(_concat_leading_dims, stacked_outputs)
        else:
            outputs = tree.map_structure(
                lambda x: x[-1] if x is not None else None,
                stacked_outputs,
            )
        return outputs, final_state

    return multi_step_scan


def _unroll_steps(
    step_function, state, batches, concatenate_outputs, concat_fn
):
    outputs_list = []
    outputs = None
    for b in batches:
        outputs, state = step_function(state, b)
        if concatenate_outputs:
            outputs_list.append(outputs)
    if concatenate_outputs:
        return concat_fn(outputs_list), state
    return outputs, state


def _build_multi_step_iterator_step(
    step_function,
    multi_step_fn,
    steps_per_execution,
    concatenate_outputs,
    concat_fn,
):
    def iterator_step(state, iterator):
        batch = next(iterator)

        # Case 1: List remainder from super_batch_iterator
        if isinstance(batch, data_adapter_utils.PartialBatchList):
            return _unroll_steps(
                step_function, state, batch, concatenate_outputs, concat_fn
            )

        leaf = tree.flatten(batch)[0]

        # Case 2: Steady-state full super-batch on-device
        if multi_step_fn is not None and leaf.shape[0] == steps_per_execution:
            return multi_step_fn(state, batch)

        # Case 3: Partial super-batch or eager/unjitted fallback
        sliced_batches = [
            tree.map_structure(lambda x, i=i: x[i], batch)
            for i in range(leaf.shape[0])
        ]
        return _unroll_steps(
            step_function,
            state,
            sliced_batches,
            concatenate_outputs,
            concat_fn,
        )

    return iterator_step


class Trainer(base_trainer.BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._jax_state_synced = True

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        metrics_variables,
        x,
        y,
        sample_weight,
        training=False,
        optimizer_variables=None,
    ):
        """This method is stateless and is intended for use with jax.grad."""
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = training

        # Run stateless forward pass
        y_pred, non_trainable_variables, losses = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            return_losses=True,
            **kwargs,
        )
        if losses:
            # Make forward pass losses available to compute_loss.
            self._losses_override.clear()
            self._losses_override = losses

        loss, variables = self.stateless_compute_loss(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=training,
        )
        if losses:
            self._losses_override.clear()
        (trainable_variables, non_trainable_variables, metrics_variables) = (
            variables
        )

        # Handle loss scaling
        unscaled_loss = loss
        if training and self.optimizer is not None:
            # Scale loss with a StatelessScope, to use an update scale variable.
            mapping = list(zip(self.optimizer.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                loss = self.optimizer.scale_loss(loss)
        return loss, (
            unscaled_loss,
            y_pred,
            non_trainable_variables,
            metrics_variables,
        )

    def _update_metrics_variables(
        self, metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
    ):
        with backend.StatelessScope(
            state_mapping=[
                (ref_v, v)
                for ref_v, v in zip(self.metrics_variables, metrics_variables)
            ]
        ) as scope:
            self._loss_tracker.update_state(
                unscaled_loss,
                sample_weight=next(
                    i for i in tree.flatten(x) if i is not None
                ).shape[0],
            )
            logs = self.compute_metrics(x, y, y_pred, sample_weight)

        new_metrics_variables = []
        for ref_v in self.metrics_variables:
            new_v = scope.get_current_value(ref_v)
            if new_v is None:
                new_v = ref_v.value
            new_metrics_variables.append(new_v)
        return logs, new_metrics_variables

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )
        (loss, aux), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=True,
            optimizer_variables=optimizer_variables,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        return logs, state

    def test_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        loss, aux = self.compute_loss_and_updates(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=False,
        )
        (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
            aux
        )

        logs, metrics_variables = self._update_metrics_variables(
            metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
        )

        state = (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )
        return logs, state

    def predict_step(self, state, data):
        trainable_variables, non_trainable_variables = state
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = False

        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        outputs, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x, **kwargs
        )
        return outputs, non_trainable_variables

    def _make_function(
        self,
        step_function,
        out_shardings=None,
        concatenate_outputs=False,
    ):
        if not self.run_eagerly and self.jit_compile:
            step_function = jit(
                step_function,
                donate_argnums=0,
                out_shardings=out_shardings,
            )

        if self.steps_per_execution > 1:
            multi_step_fn = None
            if not self.run_eagerly and self.jit_compile:
                scan_fn = _build_multi_step_scan(
                    step_function,
                    self.steps_per_execution,
                    concatenate_outputs,
                )
                multi_step_fn = jit(
                    scan_fn,
                    donate_argnums=0,
                    out_shardings=out_shardings,
                )

            concat_fn = (
                jit(_concatenate_outputs)
                if (self.jit_compile and not self.run_eagerly)
                else _concatenate_outputs
            )
            return _build_multi_step_iterator_step(
                step_function,
                multi_step_fn,
                self.steps_per_execution,
                concatenate_outputs,
                concat_fn,
            )

        return lambda state, iterator: step_function(state, next(iterator))

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return
        out_shardings = None
        if not self.run_eagerly and self.jit_compile:
            if distribution_lib.distribution() is not None:
                state_shardings = self._get_state_sharding_spec()
                out_shardings = (None, state_shardings)
        if is_nnx_enabled():
            train_step = lambda state, data: type(self).train_step(
                self, state, data
            )
        else:
            train_step = self.train_step

        self.train_function = self._make_function(
            train_step,
            out_shardings=out_shardings,
        )

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return
        out_shardings = None
        if not self.run_eagerly and self.jit_compile:
            if distribution_lib.distribution() is not None:
                (
                    trainable_shardings,
                    non_trainable_shardings,
                    _,  # optimizer_shardings
                    metrics_shardings,
                ) = self._get_state_sharding_spec()
                state_shardings = (
                    trainable_shardings,
                    non_trainable_shardings,
                    metrics_shardings,
                )
                out_shardings = (None, state_shardings)
        if is_nnx_enabled():
            test_step = lambda state, data: type(self).test_step(
                self, state, data
            )
        else:
            test_step = self.test_step

        self.test_function = self._make_function(
            test_step,
            out_shardings=out_shardings,
        )

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return

        def predict_step(state, data):
            outputs, non_trainable_variables = self.predict_step(state, data)
            return outputs, (state[0], non_trainable_variables)

        out_shardings = None
        if not self.run_eagerly and self.jit_compile:
            if distribution_lib.distribution() is not None:
                (
                    trainable_shardings,
                    non_trainable_shardings,
                    _,  # optimizer_shardings
                    _,  # metrics_shardings
                ) = self._get_state_sharding_spec()
                state_shardings = (
                    trainable_shardings,
                    non_trainable_shardings,
                )
                out_shardings = (None, state_shardings)
        self.predict_function = self._make_function(
            predict_step,
            out_shardings=out_shardings,
            concatenate_outputs=True,
        )

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
        # TODO: respect compiled trainable state
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
        epoch_iterator = JAXEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
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

        self.make_train_function()
        self.stop_training = False
        training_logs = {}
        training_finished = False
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        try:
            for epoch in range(initial_epoch, epochs):
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)

                self._jax_state_synced = True
                with epoch_iterator.catch_stop_iteration():
                    for begin_step, end_step, iterator in epoch_iterator:
                        # Callbacks
                        callbacks.on_train_batch_begin(begin_step)

                        # Train step
                        if self._jax_state_synced:
                            # The state may have been synced by a callback.
                            state = self._get_jax_state(
                                trainable_variables=True,
                                non_trainable_variables=True,
                                optimizer_variables=True,
                                metrics_variables=True,
                                purge_model_variables=True,
                            )
                            self._jax_state_synced = False

                        logs, state = self.train_function(state, iterator)
                        (
                            trainable_variables,
                            non_trainable_variables,
                            optimizer_variables,
                            metrics_variables,
                        ) = state

                        # Setting _jax_state enables callbacks to force a state
                        # sync if they need to.
                        self._jax_state = {
                            "trainable_variables": trainable_variables,
                            "non_trainable_variables": non_trainable_variables,
                            "optimizer_variables": optimizer_variables,
                            "metrics_variables": metrics_variables,
                        }
                        # Dispatch callbacks. This takes care of async dispatch.
                        callbacks.on_train_batch_end(end_step, logs)

                        if self.stop_training:
                            # Stop training if a callback has set
                            # this flag in on_(train_)batch_end.
                            break

                # Reattach state to the model
                # (if not already done by a callback).
                # NOTE: doing this after each step would be a big performance
                # bottleneck.
                self.state_sync()

                # Override with model metrics instead of last step logs if
                # needed.
                epoch_logs = dict(self._get_metrics_result_or_logs(logs))

                # Run validation.
                if validation_data is not None and self._should_eval(
                    epoch, validation_freq
                ):
                    # Create JAXEpochIterator for evaluation and cache it.
                    if getattr(self, "_eval_epoch_iterator", None) is None:
                        self._eval_epoch_iterator = JAXEpochIterator(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
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
            training_finished = True

        finally:
            self.state_sync()
            if (
                isinstance(self.optimizer, optimizers_module.Optimizer)
                and epochs > 0
            ):
                self.optimizer.finalize_variable_values(self.trainable_weights)

            # If _eval_epoch_iterator exists, delete it after all epochs
            # are done.
            if getattr(self, "_eval_epoch_iterator", None) is not None:
                del self._eval_epoch_iterator
            if training_finished:
                callbacks.on_train_end(logs=training_logs)
            self._jax_state = None
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
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of
            # input/target data.
            epoch_iterator = JAXEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
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

        self._jax_state_synced = True
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_test_batch_begin(begin_step)

                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        metrics_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False

                logs, state = self.test_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                    metrics_variables,
                ) = state

                # Setting _jax_state enables callbacks to force a state sync
                # if they need to.
                self._jax_state = {
                    # I wouldn't recommend modifying non-trainable model state
                    # during evaluate(), but it's allowed.
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "metrics_variables": metrics_variables,
                }

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_test_batch_end(end_step, logs)

                if self.stop_evaluating:
                    break

        # Reattach state back to model (if not already done by a callback).
        self.state_sync()

        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)
        self._jax_state = None

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = JAXEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            for _, _, iterator in epoch_iterator:
                # Build model
                x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(
                    next(iterator)
                )
                if is_nnx_enabled():
                    self(x)
                else:
                    with backend.StatelessScope():
                        self(x)
                break
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

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()

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

        self._jax_state_synced = True
        outputs = None
        non_trainable_variables = None
        with epoch_iterator.catch_stop_iteration():
            for begin_step, end_step, iterator in epoch_iterator:
                callbacks.on_predict_batch_begin(begin_step)
                if self._jax_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_jax_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        purge_model_variables=True,
                    )
                    self._jax_state_synced = False
                batch_outputs, state = self.predict_function(state, iterator)
                (
                    trainable_variables,
                    non_trainable_variables,
                ) = state
                self._jax_state = {
                    "trainable_variables": trainable_variables,
                    # I wouldn't recommend modifying non-trainable model state
                    # during predict(), but it's allowed.
                    "non_trainable_variables": non_trainable_variables,
                }
                outputs = append_to_outputs(batch_outputs, outputs)

                # Dispatch callbacks. This takes care of async dispatch.
                callbacks.on_predict_batch_end(
                    end_step, {"outputs": batch_outputs}
                )

                if self.stop_predicting:
                    break

        self.state_sync()
        callbacks.on_predict_end()
        self._jax_state = None
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

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

        def data():
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self.make_train_function()
        self.reset_metrics()

        # Train step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.train_function(state, data())

        # State sync
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "optimizer_variables": optimizer_variables,
            "metrics_variables": metrics_variables,
        }
        self.state_sync()

        # Format return values
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
            yield _distribute_data((x, y, sample_weight))

        # Maybe build model
        self._symbolic_build(data_batch=next(data()))
        self.make_test_function()
        self.reset_metrics()

        # Test step
        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._jax_state_synced = False
        logs, state = self.test_function(state, data())

        # State sync
        trainable_variables, non_trainable_variables, metrics_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "metrics_variables": metrics_variables,
        }
        self.state_sync()

        # Format return values.
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        if not all(layer.built for layer in self._flatten_layers()):
            # Build model
            with backend.StatelessScope():
                self(x)
        self.make_predict_function()

        state = self._get_jax_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=False,
            purge_model_variables=False,
        )
        self._jax_state_synced = False

        def data():
            yield (x,)

        batch_outputs, state = self.predict_function(state, data())
        trainable_variables, non_trainable_variables = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
        }
        self.state_sync()
        batch_outputs = tree.map_structure(lambda x: np.array(x), batch_outputs)
        return batch_outputs

    def _maybe_symbolic_build(self, iterator=None, data_batch=None):
        if not any(self._get_unbuilt_components()):
            return
        if data_batch is None and iterator is not None:
            try:
                data_batch = next(
                    iter(iterator.data_adapter.get_jax_iterator())
                )
            except StopIteration:
                raise ValueError(
                    "The symbolic build failed because train or validation "
                    "dataset is empty."
                )
        self._symbolic_build(data_batch=data_batch)

    def state_sync(self):
        if not getattr(self, "_jax_state", None) or self._jax_state_synced:
            return

        trainable_variables = self._jax_state.get("trainable_variables", None)
        non_trainable_variables = self._jax_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._jax_state.get("optimizer_variables", None)
        metrics_variables = self._jax_state.get("metrics_variables", None)
        if trainable_variables:
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
        if non_trainable_variables:
            for ref_v, v in zip(
                self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
        if optimizer_variables:
            for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
                ref_v.assign(v)
        if metrics_variables:
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)
        self._jax_state_synced = True

    def jax_state_sync(self):
        # Kept for backwards compatibility, use `state_sync` instead.
        return self.state_sync()

    def _get_state_sharding_spec(self):
        trainable_shardings = [
            v.value.sharding for v in self.trainable_variables
        ]
        non_trainable_shardings = [
            v.value.sharding for v in self.non_trainable_variables
        ]
        if hasattr(self, "optimizer") and self.optimizer is not None:
            optimizer_shardings = [
                v.value.sharding for v in self.optimizer.variables
            ]
        else:
            optimizer_shardings = []
        metrics_shardings = [v.value.sharding for v in self.metrics_variables]

        self._check_sharding_consistency(
            trainable_shardings,
            non_trainable_shardings,
            optimizer_shardings,
            metrics_shardings,
        )

        return (
            trainable_shardings,
            non_trainable_shardings,
            optimizer_shardings,
            metrics_shardings,
        )

    def _check_sharding_consistency(
        self,
        trainable_shardings,
        non_trainable_shardings,
        optimizer_shardings,
        metrics_shardings,
    ):
        """Warn if there is a mix of local and distributed variable shardings.

        When some variables have SingleDeviceSharding (created outside the
        distribution scope) and others have mesh-aware shardings (created
        inside), passing them together as `out_shardings` to `jax.jit`
        raises ``ValueError: Received incompatible devices for jitted
        computation``. This helper detects the mismatch early and emits
        an actionable warning.
        """
        if distribution_lib.distribution() is None:
            return

        var_shard_pairs = itertools.chain(
            zip(self.trainable_variables, trainable_shardings),
            zip(self.non_trainable_variables, non_trainable_shardings),
            zip(
                (
                    self.optimizer.variables
                    if hasattr(self, "optimizer") and self.optimizer
                    else []
                ),
                optimizer_shardings,
            ),
            zip(self.metrics_variables, metrics_shardings),
        )

        first_local_var_path = None
        has_mesh = False
        for v, s in var_shard_pairs:
            if isinstance(s, jax.sharding.SingleDeviceSharding):
                if first_local_var_path is None:
                    first_local_var_path = v.path
            else:
                has_mesh = True
            # Early exit: we know there is a mix as soon as we have
            # seen at least one of each kind.
            if first_local_var_path and has_mesh:
                break

        if not (first_local_var_path and has_mesh):
            return

        warnings.warn(
            "Detected a mix of local (SingleDeviceSharding) and "
            "distributed (mesh-aware) variables. This will cause "
            "a 'ValueError: Received incompatible devices for "
            "jitted computation' when JAX tries to compile the "
            "training step.\n\n"
            f"First local variable found: {first_local_var_path!r}\n\n"
            "This typically happens when the model is built or "
            "weights are loaded before the distribution is set. "
            "To fix this, call set_distribution() before creating "
            "any Keras objects:\n\n"
            "    import keras\n"
            "    keras.distribution.set_distribution(distribution)\n"
            "    model = create_model()\n"
            "    model.compile(...)\n"
            "    model.fit(...)\n\n"
            "Alternatively, use the distribution scope context "
            "manager:\n\n"
            "    with distribution.scope():\n"
            "        model = create_model()\n"
            "        model.compile(...)\n"
            "        model.fit(...)\n",
            stacklevel=3,
        )

    def _purge_model_variables(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
    ):
        """Remove all the model variable for memory saving.

        During JAX training, since the training function is stateless, we have
        to pass in and get the model weights over and over, during which the
        copy of the weights that attached to the Variable are still and
        occupying extra memory. We remove those variable to save memory (for
        better memory utilization) at the beginning of the epoch, and reattach
        the value back to variables at the end of the epoch, via
        `state_sync()`.
        """
        if trainable_variables:
            for v in self.trainable_variables:
                v._value = None
        if non_trainable_variables:
            for v in self.non_trainable_variables:
                v._value = None
        if optimizer_variables:
            for v in self.optimizer.variables:
                v._value = None
        if metrics_variables:
            for v in self.metrics_variables:
                v._value = None

    def _get_jax_state(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
        purge_model_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizer.variables])
        if metrics_variables:
            state.append([v.value for v in self.metrics_variables])
        if purge_model_variables:
            self._purge_model_variables(
                trainable_variables=trainable_variables,
                non_trainable_variables=non_trainable_variables,
                optimizer_variables=optimizer_variables,
                metrics_variables=metrics_variables,
            )
        return tuple(state)


def get_layout(d, distribution=None, is_super_batch=False):
    """Computes backend layout for a single batch or super-batch tensor."""
    if d is None or distribution is None:
        return None
    shape = d.shape[1:] if is_super_batch else d.shape
    base_layout = distribution.get_data_layout(shape)
    if base_layout is None:
        return None
    if is_super_batch:
        new_axes = (None,) + base_layout.axes
        return distribution_lib.TensorLayout(
            new_axes, base_layout.device_mesh
        ).backend_layout
    return base_layout.backend_layout


def _distribute_data(data, layouts=None, is_super_batch=False):
    distribution = distribution_lib.distribution()

    if distribution is not None:
        if layouts is None:
            layouts = tree.map_structure(
                lambda d: get_layout(d, distribution, is_super_batch),
                data,
            )
        jax_dist_data_input = partial(
            jax_distribution_lib.distribute_data_input,
            batch_dim_name=distribution.batch_dim_name,
        )
        return tree.map_structure(jax_dist_data_input, data, layouts)

    return tree.map_structure(jax.device_put, data)


class JAXEpochIterator(EpochIterator):
    def __next__(self):
        return next(self._epoch_iterator)

    def _get_iterator(self):
        super_batch = (
            self.steps_per_execution if self.steps_per_execution > 1 else None
        )
        raw_iterator = self.data_adapter.get_jax_iterator(
            super_batch=super_batch
        )

        def _distributed_iterator():
            for data in raw_iterator:
                if isinstance(data, data_adapter_utils.PartialBatchList):
                    yield data_adapter_utils.PartialBatchList(
                        [_distribute_data(b) for b in data]
                    )
                else:
                    yield _distribute_data(
                        data, is_super_batch=super_batch is not None
                    )

        iterator = _distributed_iterator()
        if jax.default_backend() == "cpu":
            return iterator
        return self._one_batch_ahead_iterator(iterator)

    def _one_batch_ahead_iterator(self, iterator):
        """Initiate transfers to the device one batch ahead.

        This utility takes an iterator and returns a new iterator which
        initiates the transfer to device one step ahead. This can improve the
        performance of training loops significantly by overlapping compute and
        data transfer.

        Args:
            iterator: Source data iterator yielding batches.

        Yields:
            Batches one step ahead of the source iterator.
        """
        next_batch = None
        for batch in iterator:
            if next_batch is None:
                next_batch = batch
            else:
                current_batch = next_batch
                next_batch = batch
                yield current_batch

        if next_batch is not None:
            yield next_batch
