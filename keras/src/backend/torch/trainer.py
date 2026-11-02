import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.nn.parallel import DistributedDataParallel

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend.torch.distribution_lib import distribute_data_input
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import ModelParallel
from keras.src.distribution.distribution_lib import distribution
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.trainers.trainer import BaseTrainer
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils.python_utils import pythonify_logs


class Trainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self.ddp_model = None

    @tracking.no_automatic_dependency_tracking
    def _initialize_ddp(self):
        if torch.distributed.is_initialized() and self.ddp_model is None:
            active_distribution = distribution()

            if active_distribution is None or isinstance(
                active_distribution, DataParallel
            ):
                device = get_device()
                device_str = str(device)
                if device_str.startswith("cuda") or device_str.startswith(
                    "xpu"
                ):
                    if ":" in device_str:
                        device_ids = [int(device_str.split(":")[-1])]
                    else:
                        if device_str.startswith("cuda"):
                            device_ids = [torch.cuda.current_device()]
                        else:
                            device_ids = [torch.xpu.current_device()]
                else:
                    device_ids = None

                process_group = None
                if active_distribution is not None:
                    backend_mesh = active_distribution.device_mesh.backend_mesh
                    process_group = backend_mesh.get_group(
                        active_distribution.batch_dim_name
                    )

                # Bypass PyTorch submodule registration, which does not handle
                # circular dependencies
                object.__setattr__(
                    self,
                    "ddp_model",
                    DistributedDataParallel(
                        self,
                        device_ids=device_ids,
                        process_group=process_group,
                        find_unused_parameters=False,
                    ),
                )

    def _use_torch_compile(self):
        if self.run_eagerly or not self.jit_compile:
            return False
        active_distribution = distribution()
        if active_distribution is not None and not isinstance(
            active_distribution, DataParallel
        ):
            # The distribution entered `KerasDTensorPromotionMode`, a
            # `TorchDispatchMode`. TorchDynamo skips every frame while a
            # dispatch mode is active, and checking every frame makes the
            # compiled step slower than the eager one.
            warnings.warn(
                "`jit_compile=True` is not supported with "
                f"`{active_distribution.__class__.__name__}` on the torch "
                "backend. Running without `torch.compile`."
            )
            return False
        return True

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Compute predictions
        model = self.ddp_model or self
        if self._call_has_training_arg:
            y_pred = model(x, training=True)
        else:
            y_pred = model(x)

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        model.zero_grad()

        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True
        )
        self._loss_tracker.update_state(
            loss.detach() if isinstance(loss, torch.Tensor) else loss,
            sample_weight=(
                x.shape[0]
                if isinstance(x, torch.Tensor)
                else next(i for i in tree.flatten(x) if i is not None).shape[0]
            ),
        )
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            trainable_weights = self.trainable_weights[:]
            if (
                not self.run_eagerly
                and self.jit_compile
                and self.ddp_model is None
            ):
                trainable_tensors = [v.value for v in trainable_weights]
                if isinstance(loss, torch.Tensor) and not loss.requires_grad:
                    gradients = [None] * len(trainable_weights)
                elif all(t.requires_grad for t in trainable_tensors):
                    gradients = torch.autograd.grad(
                        loss, trainable_tensors, allow_unused=True
                    )
                else:
                    inputs = [t for t in trainable_tensors if t.requires_grad]
                    grads = (
                        iter(
                            torch.autograd.grad(loss, inputs, allow_unused=True)
                        )
                        if inputs
                        else iter(())
                    )
                    gradients = [
                        next(grads) if t.requires_grad else None
                        for t in trainable_tensors
                    ]
            else:
                # Call torch.Tensor.backward() on the loss to compute gradients
                # for the weights (required for DDP bucket all-reduce hooks).
                loss.backward()
                gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        (
            x,
            y,
            sample_weight,
        ) = data_adapter_utils.unpack_x_y_sample_weight(data)
        model = self.ddp_model or self
        if self._call_has_training_arg:
            y_pred = model(x, training=False)
        else:
            y_pred = model(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=(
                x.shape[0]
                if isinstance(x, torch.Tensor)
                else next(i for i in tree.flatten(x) if i is not None).shape[0]
            ),
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        model = self.ddp_model or self
        if self._call_has_training_arg:
            y_pred = model(x, training=False)
        else:
            y_pred = model(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        self._initialize_ddp()

        train_step = self.train_step
        if self._use_torch_compile():
            train_step = torch.compile(train_step)
            if (
                type(self).train_step is Trainer.train_step
                and self.ddp_model is None
                and hasattr(torch._dynamo.config, "trace_autograd_ops")
            ):
                # Let Dynamo trace `torch.autograd.grad` into the graph. Scope
                # the flag to this step: with the flag on, Dynamo also traces
                # `Tensor.backward()`, which breaks `train_step` overrides.
                train_step = torch._dynamo.config.patch(
                    trace_autograd_ops=True
                )(train_step)

        def train_function(data):
            """Runs training steps on a list of batches of data."""
            logs = {}
            for step_data in data:
                logs = train_step(step_data)
            return logs

        self.train_function = train_function

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        self._initialize_ddp()

        test_step = self.test_step
        if self._use_torch_compile():
            test_step = torch.compile(test_step)

        def test_function(data):
            """Runs test steps on a list of batches of data."""
            logs = {}
            with torch.no_grad():
                for step_data in data:
                    logs = test_step(step_data)
            return logs

        self.test_function = test_function

    def get_metrics_result(self):
        if not torch.distributed.is_initialized():
            return super().get_metrics_result()

        # This also runs inside a compiled `train_step`, through
        # `compute_metrics()`. The `with` blocks are not nested: TorchDynamo
        # in torch 2.13 fails to inline a call inside nested `with` blocks on
        # Python 3.11.
        with torch.no_grad():
            active_distribution = distribution()
            process_group = None
            if active_distribution is not None:
                # Reuse the cached mesh: TorchDynamo fails to trace building
                # a new one.
                backend_mesh = active_distribution.device_mesh.backend_mesh
                process_group = backend_mesh.get_group(
                    active_distribution.batch_dim_name
                )

            reduced_vars = []
            for v in self.metrics_variables:
                # Use a copy for reduction to avoid modifying
                # the original variable.
                val = v.value.clone()
                if isinstance(val, DTensor):
                    # DTensor indicates we're using ModelParallel.
                    placements = [Replicate()] * len(val.placements)
                    val = val.redistribute(
                        val.device_mesh, placements
                    ).to_local()
                dist.all_reduce(
                    val,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                )
                reduced_vars.append(val)

        # The reduced values have no gradient, so neither do the results.
        with backend.StatelessScope(
            state_mapping=list(zip(self.metrics_variables, reduced_vars))
        ):
            results = super().get_metrics_result()

        return {
            k: val.clone() if isinstance(val, torch.Tensor) else val
            for k, val in results.items()
        }

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        self._initialize_ddp()

        predict_step = self.predict_step
        if self._use_torch_compile():
            predict_step = torch.compile(predict_step)

        def predict_function(data):
            """Runs predict steps on a list of batches of data."""
            outputs = []
            with torch.no_grad():
                for step_data in data:
                    outputs.append(
                        _gather_non_batch_axes(predict_step(step_data))
                    )

            def concat_outputs(outputs):
                if not outputs:
                    return []
                if len(outputs) == 1:
                    return outputs[0]
                return tree.map_structure(
                    lambda *args: torch.cat(args, dim=0),
                    *outputs,
                )

            return concat_outputs(outputs)

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

        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            # TODO: Support torch tensors for validation data.
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
        epoch_iterator = TorchEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        self._symbolic_build(iterator=epoch_iterator)
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
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            # Switch the torch Module to training mode. Inform torch layers to
            # do training behavior in case the user did not use `self.training`
            # when implementing a custom layer with torch layers.
            self.train()

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                # Callbacks
                callbacks.on_train_batch_begin(begin_step)

                logs = self.train_function(data)

                # Callbacks
                callbacks.on_train_batch_end(end_step, logs)
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Switch the torch Module back to testing mode.
            self.eval()

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create TorchEpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TorchEpochIterator(
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
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TorchEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(iterator=epoch_iterator)
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

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_test_batch_begin(begin_step)
            logs = self.test_function(data)
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
        epoch_iterator = TorchEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
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

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_predict_batch_begin(begin_step)
            batch_outputs = self.predict_function(data)
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        outputs = tree.map_structure(backend.ops.convert_to_numpy, outputs)
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

        data = (x, y, sample_weight)
        data = _distribute_data(data)

        # Maybe build model
        self._symbolic_build(data_batch=data)
        self.make_train_function()
        self.reset_metrics()

        logs = self.train_function([data])
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

        data = (x, y, sample_weight)
        data = _distribute_data(data)

        # Maybe build model
        self._symbolic_build(data_batch=data)
        self.make_test_function()
        self.reset_metrics()

        logs = self.test_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        x = _distribute_data(x)
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.ops.convert_to_numpy, batch_outputs
        )
        return batch_outputs


def _gather_non_batch_axes(data):
    """Gathers sharded outputs along every mesh axis but the batch axis.

    This is the inverse of `_distribute_data`: each process gets every
    feature of the rows that it fed in.
    """
    active_distribution = distribution()
    if active_distribution is None:
        return data
    batch_dim_name = active_distribution.batch_dim_name

    def gather(x):
        if not isinstance(x, DTensor):
            return x
        mesh = x.device_mesh
        names = mesh.mesh_dim_names or (None,) * mesh.ndim
        placements = [
            Shard(0) if x.ndim and name == batch_dim_name else Replicate()
            for name in names
        ]
        return x.redistribute(mesh, placements).to_local()

    return tree.map_structure(gather, data)


def _distribute_data(data, layouts=None):
    active_distribution = distribution()
    if active_distribution is not None and isinstance(
        active_distribution, ModelParallel
    ):
        if layouts is None:

            def get_layout(d):
                if d is None:
                    return None
                return active_distribution.get_data_layout(d.shape)

            layouts = tree.map_structure(get_layout, data)
        return tree.map_structure(
            lambda d, l: distribute_data_input(d, l),
            data,
            layouts,
        )
    return tree.map_structure(
        lambda x: convert_to_tensor(x) if x is not None else None, data
    )


class TorchEpochIterator(EpochIterator):
    def _get_iterator(self):
        active_distribution = distribution()
        if active_distribution is not None and isinstance(
            active_distribution, ModelParallel
        ):
            return self._get_distributed_iterator(active_distribution)
        return self.data_adapter.get_torch_dataloader()

    def _get_distributed_iterator(self, active_distribution):
        layouts = None
        for data in self.data_adapter.get_numpy_iterator():
            if layouts is None:

                def get_layout(d):
                    if d is None:
                        return None
                    return active_distribution.get_data_layout(d.shape)

                layouts = tree.map_structure(get_layout, data)
            yield _distribute_data(data, layouts)
