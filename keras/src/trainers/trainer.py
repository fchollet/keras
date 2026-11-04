import inspect
import platform
import warnings

from keras.src import backend
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import optimizers
from keras.src import tree
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.optimizers.loss_scale_optimizer import LossScaleOptimizer
from keras.src.saving import serialization_lib
from keras.src.trainers.compile_utils import CompileLoss
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking


class BaseTrainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self._jit_compile = None
        self.compiled = False
        self.loss = None
        self.steps_per_execution = 1
        # Can be set by callbacks in on_train_begin
        self._initial_epoch = None
        self._compute_loss_has_training_arg = (
            "training" in inspect.signature(self.compute_loss).parameters
        )

        # Placeholders used in `compile`
        self._compile_loss = None
        self._compile_metrics = None
        self._loss_tracker = None

    @traceback_utils.filter_traceback
    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        """Configures the model for training.

        Example:

        ```python
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.FalseNegatives(),
            ],
        )
        ```

        Args:
            optimizer: String (name of optimizer) or optimizer instance. See
                `keras.optimizers`.
            loss: Loss function. May be a string (name of loss function), or
                a `keras.losses.Loss` instance. See `keras.losses`. A
                loss function is any callable with the signature
                `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth
                values, and `y_pred` are the model's predictions.
                `y_true` should have shape `(batch_size, d0, .. dN)`
                (except in the case of sparse loss functions such as
                sparse categorical crossentropy which expects integer arrays of
                shape `(batch_size, d0, .. dN-1)`).
                `y_pred` should have shape `(batch_size, d0, .. dN)`.
                The loss function should return a float tensor.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions of
                different model outputs. The loss value that will be minimized
                by the model will then be the *weighted sum* of all individual
                losses, weighted by the `loss_weights` coefficients.  If a list,
                it is expected to have a 1:1 mapping to the model's outputs. If
                a dict, it is expected to map output names (strings) to scalar
                coefficients.
            metrics: List of metrics to be evaluated by the model during
                training and testing. Each of this can be a string (name of a
                built-in function), function or a `keras.metrics.Metric`
                instance. See `keras.metrics`. Typically you will use
                `metrics=['accuracy']`. A function is any callable with the
                signature `result = fn(y_true, _pred)`. To specify different
                metrics for different outputs of a multi-output model, you could
                also pass a dictionary, such as
                `metrics={'a':'accuracy', 'b':['accuracy', 'mse']}`.
                You can also pass a list to specify a metric or a list of
                metrics for each output, such as
                `metrics=[['accuracy'], ['accuracy', 'mse']]`
                or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass
                the strings 'accuracy' or 'acc', we convert this to one of
                `keras.metrics.BinaryAccuracy`,
                `keras.metrics.CategoricalAccuracy`,
                `keras.metrics.SparseCategoricalAccuracy` based on the
                shapes of the targets and of the model output. A similar
                conversion is done for the strings `"crossentropy"`
                and `"ce"` as well.
                The metrics passed here are evaluated without sample weighting;
                if you would like sample weighting to apply, you can specify
                your metrics via the `weighted_metrics` argument instead.
            weighted_metrics: List of metrics to be evaluated and weighted by
                `sample_weight` or `class_weight` during training and testing.
            run_eagerly: Bool. If `True`, this model's forward pass
                 will never be compiled. It is recommended to leave this
                 as `False` when training (for best performance),
                 and to set it to `True` when debugging.
            steps_per_execution: Int. The number of batches to run
                during each a single compiled function call. Running multiple
                batches inside a single compiled function call can
                greatly improve performance on TPUs or small models with a large
                Python overhead. At most, one full epoch will be run each
                execution. If a number larger than the size of the epoch is
                passed, the execution will be truncated to the size of the
                epoch. Note that if `steps_per_execution` is set to `N`,
                `Callback.on_batch_begin` and `Callback.on_batch_end` methods
                will only be called every `N` batches (i.e. before/after
                each compiled function execution).
                Not supported with the PyTorch backend.
            jit_compile: Bool or `"auto"`. Whether to use XLA compilation when
                compiling a model. For `jax` and `tensorflow` backends,
                `jit_compile="auto"` enables XLA compilation if the model
                supports it, and disabled otherwise.
                For `torch` backend, `"auto"` will default to eager
                execution and `jit_compile=True` will run with `torch.compile`
                with the `"inductor"` backend.
                On the JAX backend, compilation happens on the first step and
                can take several seconds. JAX can persist the compiled result
                to disk and reuse it across runs and processes, so the same
                model only pays that cost once. Enable it before the first
                step with:
                ```python
                import jax
                jax.config.update("jax_compilation_cache_dir", "/path/to/cache")
                ```
                See the JAX persistent compilation cache docs for the
                available thresholds.
            auto_scale_loss: Bool. If `True` and the model dtype policy is
                `"mixed_float16"`, the passed optimizer will be automatically
                wrapped in a `LossScaleOptimizer`, which will dynamically
                scale the loss to prevent underflow.

        Note:
            Trainable variables are determined at `compile()` time. If you
            modify the `trainable` property of a layer after calling
            `compile()`, those changes will not take effect during `fit()`
            unless `compile()` is called again.

            Recommended workflow when changing trainable variables:
            ```python
            # Initial training with some layers
            model.compile(optimizer="adam", loss="mse")
            model.fit(x_train, y_train)

            # Change trainable flags
            layer.trainable = False  # or True

            # Recompile for the change to take effect
            model.compile(optimizer="adam", loss="mse")
            model.fit(x_train, y_train)
            ```

            This behavior applies to all Keras backends and is also documented
            in the transfer learning guide.
        """
        optimizer = optimizers.get(optimizer)
        self.optimizer = optimizer
        if (
            auto_scale_loss
            and self.dtype_policy.name == "mixed_float16"
            and self.optimizer
            and not isinstance(self.optimizer, LossScaleOptimizer)
        ):
            self.optimizer = LossScaleOptimizer(
                self.optimizer, name="loss_scale_optimizer"
            )
        if hasattr(self, "output_names"):
            output_names = self.output_names
        else:
            output_names = None
        if loss is not None:
            self._compile_loss = CompileLoss(
                loss, loss_weights, output_names=output_names
            )
            self.loss = loss
        if metrics is not None or weighted_metrics is not None:
            self._compile_metrics = CompileMetrics(
                metrics, weighted_metrics, output_names=output_names
            )
        if jit_compile == "auto":
            if run_eagerly:
                jit_compile = False
            else:
                jit_compile = self._resolve_auto_jit_compile()
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` "
                "cannot also be True. Disabling `jit_compile`.",
                stacklevel=2,
            )

        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self._compiled_trainable_state = {
            layer: layer.trainable for layer in self._flatten_layers()
        }
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")
        self.steps_per_execution = steps_per_execution

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
        )

    @property
    def jit_compile(self):
        if self._jit_compile is None:
            # Value was never set. Resolve it now.
            self._jit_compile = self._resolve_auto_jit_compile()
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value):
        if value and not model_supports_jit(self):
            warnings.warn(
                "Model doesn't support `jit_compile=True`. "
                "Proceeding with `jit_compile=False`."
            )
            self._jit_compile = False
        else:
            self._jit_compile = value

    def _resolve_auto_jit_compile(self):
        if backend.backend() == "torch":
            # jit_compile = "auto" with the pytorch backend defaults to eager
            return False

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            devices = tf.config.list_physical_devices()
            if not list(filter(lambda x: x.device_type != "CPU", devices)):
                # Disable XLA on CPU-only machines.
                return False

            if self._distribute_strategy:
                # Disable XLA with tf.distribute
                return False

        if model_supports_jit(self):
            return True
        return False

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    @property
    def metrics(self):
        # Order: loss tracker, individual loss trackers, compiled metrics,
        # custom metrics, sublayer metrics.
        metrics = []
        if self.compiled:
            if self._loss_tracker is not None:
                metrics.append(self._loss_tracker)
            if self._compile_metrics is not None:
                metrics.append(self._compile_metrics)
            if self._compile_loss is not None:
                metrics.extend(self._compile_loss.metrics)
        metrics.extend(self._metrics)
        for layer in self._flatten_layers(include_self=False):
            if isinstance(layer, BaseTrainer):
                # All Trainer-related metrics in sublayers should be ignored
                # because a new Trainer has been instantiated.
                continue
            metrics.extend(layer.metrics)
        return metrics

    @property
    def metrics_names(self):
        return [m.name for m in self.metrics]

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def _get_own_metrics(self):
        metrics = []
        if self._loss_tracker is not None:
            metrics.append(self._loss_tracker)
        if self._compile_metrics is not None:
            metrics.append(self._compile_metrics)
        if self._compile_loss is not None:
            metrics.extend(self._compile_loss.metrics)
        metrics.extend(self._metrics)
        return metrics

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        """Compute the total loss, validate it, and return it.

        Subclasses can optionally override this method to provide custom loss
        computation logic.

        Example:

        ```python
        class MyModel(Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_tracker = metrics.Mean(name='loss')

            def compute_loss(self, x, y, y_pred, sample_weight, training=True):
                loss = ops.mean((y_pred - y) ** 2)
                loss += ops.sum(self.losses)
                self.loss_tracker.update_state(loss)
                return loss

            def reset_metrics(self):
                self.loss_tracker.reset_state()

            @property
            def metrics(self):
                return [self.loss_tracker]

        inputs = layers.Input(shape=(10,), name='my_input')
        outputs = layers.Dense(10)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(ops.sum(outputs))

        optimizer = SGD()
        model.compile(optimizer, loss='mse', steps_per_execution=10)
        dataset = ...
        model.fit(dataset, epochs=2, steps_per_epoch=10)
        print(f"Custom loss: {model.loss_tracker.result()}")
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`)
            sample_weight: Sample weights for weighting the loss function.
            training: Whether we are training or evaluating the model.

        Returns:
            The total loss as a scalar tensor, or `None` if no loss results
            (which is the case when called by `Model.test_step`).
        """
        # The default implementation does not use `x` or `training`.
        del x
        del training
        losses = []
        if self._compile_loss is not None:
            loss = self._compile_loss(y, y_pred, sample_weight)
            if loss is not None:
                losses.append(loss)
        for loss in self.losses:
            losses.append(self._aggregate_additional_loss(loss))
        if backend.backend() != "jax" and len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        elif len(losses) == 0:
            total_loss = ops.zeros(())
        else:
            total_loss = ops.sum(losses)
        return total_loss

    def _compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        """Backwards compatibility wrapper for `compute_loss`.

        This should be used instead `compute_loss` within `train_step` and
        `test_step` to support overrides of `compute_loss` that may not have
        the `training` argument, as this argument was added in Keras 3.3.
        """
        if self._compute_loss_has_training_arg:
            return self.compute_loss(
                x, y, y_pred, sample_weight, training=training
            )
        else:
            return self.compute_loss(x, y, y_pred, sample_weight)

    def _aggregate_additional_loss(self, loss):
        """Aggregates losses from `add_loss`, regularizers and sublayers.

        Args:
            loss: A tensor representing the additional loss to aggregate.

        Returns:
            A tensor representing the summed loss, cast to the `floatx()` if
            necessary.
        """
        if not is_float_dtype(loss.dtype):
            loss = ops.cast(loss, dtype=backend.floatx())
        return ops.sum(loss)

    def stateless_compute_loss(
        self,
        trainable_variables,
        non_trainable_variables,
        metrics_variables,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        var_mapping = list(zip(self.trainable_variables, trainable_variables))
        var_mapping.extend(
            zip(self.non_trainable_variables, non_trainable_variables)
        )
        var_mapping.extend(zip(self.metrics_variables, metrics_variables))
        with backend.StatelessScope(state_mapping=var_mapping) as scope:
            # Note that this is needed for the regularization loss, which need
            # the latest value of train/non-trainable variables.
            loss = self._compute_loss(
                x,
                y,
                y_pred,
                sample_weight=sample_weight,
                training=training,
            )

        # Update non trainable vars (may have been updated in compute_loss)
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            non_trainable_variables.append(new_v)

        # Update metrics vars (may have been updated in compute_loss)
        metrics_variables = []
        for v in self.metrics_variables:
            new_v = scope.get_current_value(v)
            metrics_variables.append(new_v)
        return loss, (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic. Custom metrics are not passed in
        `compile()`, they can be created in `__init__` or `build`. They are
        automatically tracked and returned by `self.metrics`.

        Example:

        ```python
        class MyModel(Sequential):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.custom_metric = MyMetric(name="custom_metric")

            def compute_metrics(self, x, y, y_pred, sample_weight):
                # This super call updates metrics from `compile` and returns
                # results for all metrics listed in `self.metrics`.
                metric_results = super().compute_metrics(
                    x, y, y_pred, sample_weight)

                # `metric_results` contains the previous result for
                # `custom_metric`, this is where we update it.
                self.custom_metric.update_state(x, y, y_pred, sample_weight)
                metric_results['custom_metric'] = self.custom_metric.result()
                return metric_results
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model output of `model.call(x)`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            A `dict` containing values that will be passed to
            `keras.callbacks.CallbackList.on_train_batch_end()`. Typically,
            the values of the metrics listed in `self.metrics` are returned.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            self._compile_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
        """Returns the model's metrics values as a dict.

        If any of the metric result is a dict (containing multiple metrics),
        each of them gets added to the top level returned dict of this method.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics`.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

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
        """Trains the model for a fixed number of epochs (dataset iterations).

        Args:
            x: Input data. It can be:
                - A NumPy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
                - A backend-native tensor, or a list of tensors
                (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
                - A `keras.utils.PyDataset` returning `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
                - A `tf.data.Dataset` yielding `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
                - A `torch.utils.data.DataLoader` yielding `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
                - A Python generator function yielding `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`, it can be either NumPy
                array(s) or backend-native tensor(s). If `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or a Python generator function,
                `y` should not be specified since targets will be obtained from
                `x`.
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                "auto" becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g., in a production environment). Defaults to `"auto"`.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `keras.callbacks`. Note
                `keras.callbacks.ProgbarLogger` and
                `keras.callbacks.History` callbacks are created
                automatically and need not be passed to `model.fit()`.
                `keras.callbacks.ProgbarLogger` is created
                or not based on the `verbose` argument in `model.fit()`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model
                metrics on this data at the end of each epoch. The validation
                data is selected from the last samples in the `x` and `y` data
                provided, before shuffling.
                This argument is only supported when `x` and `y` are made of
                NumPy arrays or tensors.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using
                `validation_split` or `validation_data` is not affected by
                regularization layers like noise and dropout.
                `validation_data` will override `validation_split`.
                It can be:
                - A tuple `(x_val, y_val)` of NumPy arrays or tensors.
                - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                arrays.
                - A `keras.utils.PyDataset`, a `tf.data.Dataset`, a
                `torch.utils.data.DataLoader` yielding `(inputs, targets)` or a
                Python generator function yielding `(x_val, y_val)` or
                `(inputs, targets, sample_weights)`.
            shuffle: Boolean, whether to shuffle the training data before each
                epoch. This argument is ignored when `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must be
                one-hot encoded, or an explicit final dimension of `1` must
                be included for sparse class labels.
            sample_weight: Optional NumPy array or tensor of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                NumPy array or tensor with the same length as the input samples
                (1:1 mapping between weights and samples), or in the case of
                temporal data, you can pass a 2D NumPy array or tensor with
                shape `(samples, sequence_length)` to apply a different weight
                to every timestep of every sample.
                This argument is not supported when `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function.
                Instead, provide `sample_weights` as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples) before declaring one
                epoch finished and starting the next epoch. When training with
                input tensors or NumPy arrays, the default `None` means that the
                value used is the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
                If `x` is a `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function, the
                epoch will run until the input dataset is exhausted. When
                passing an infinitely repeating dataset, you must specify the
                `steps_per_epoch` argument, otherwise the training will run
                indefinitely.
            validation_steps: Integer or `None`.
                Only relevant if `validation_data` is provided.
                Total number of steps (batches of samples) to draw before
                stopping when performing validation at the end of every epoch.
                If `validation_steps` is `None`, validation will run until the
                `validation_data` dataset is exhausted. In the case of an
                infinitely repeating dataset, it will run indefinitely. If
                `validation_steps` is specified and only part of the dataset
                is consumed, the evaluation will start from the beginning of the
                dataset at each epoch. This ensures that the same validation
                samples are used every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            validation_freq: Only relevant if validation data is provided.
                Specifies how many training epochs to run
                before a new validation run is performed,
                e.g. `validation_freq=2` runs validation every 2 epochs.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass an iterator like object such as a
            `tf.data.Dataset` or a `keras.utils.PyDataset` to `fit()`,
            which will in fact yield not only features (`x`)
            but optionally targets (`y`) and sample weights (`sample_weight`).
            Keras requires that the output of such iterator-likes be
            unambiguous. The iterator should return a tuple
            of length 1, 2, or 3, where the optional second and third elements
            will be used for `y` and `sample_weight` respectively.
            Any other type provided will be wrapped in
            a length-one tuple, effectively treating everything as `x`. When
            yielding dicts, they should still adhere to the top-level tuple
            structure,
            e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
            features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the `namedtuple`. The reason is
            that it behaves like both an ordered datatype (tuple) and a mapping
            datatype (dict). So given a namedtuple of the form:
            `namedtuple("example_tuple", ["y", "x"])`
            it is ambiguous whether to reverse the order of the elements when
            interpreting the value. Even worse is a tuple of the form:
            `namedtuple("other_tuple", ["x", "y", "z"])`
            where it is unclear if the tuple was intended to be unpacked
            into `x`, `y`, and `sample_weight` or passed through
            as a single element to `x`.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        raise NotImplementedError

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
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches (see the `batch_size` arg.)

        Args:
            x: Input data. It can be:
                - A NumPy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
                - A backend-native tensor, or a list of tensors
                (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
                - A `keras.utils.PyDataset` returning `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
                - A `tf.data.Dataset` yielding `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
                - A `torch.utils.data.DataLoader` yielding `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
                - A Python generator function yielding `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`, it can be either NumPy
                array(s) or backend-native tensor(s). If `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or a Python generator function,
                `y` should not be specified since targets will be obtained from
                `x`.
            batch_size: Integer or `None`.
                Number of samples per batch of computation.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file, so `verbose=2` is
                recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            sample_weight: Optional NumPy array or tensor of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                NumPy array or tensor with the same length as the input samples
                (1:1 mapping between weights and samples), or in the case of
                temporal data, you can pass a 2D NumPy array or tensor with
                shape `(samples, sequence_length)` to apply a different weight
                to every timestep of every sample.
                This argument is not supported when `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function.
                Instead, provide `sample_weights` as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            steps: Integer or `None`.
                Total number of steps (batches of samples) to draw before
                declaring the evaluation round finished. If `steps` is `None`,
                it will run until `x` is exhausted. In the case of an infinitely
                repeating dataset, it will run indefinitely.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric.
                If `False`, they are returned as a list.

        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics).

        Note: When using compiled metrics, `evaluate()` may return multiple
        submetric values, while `model.metrics_names` often lists only
        top-level names (e.g., 'loss', 'compile_metrics'), leading to a
        length mismatch. The order of the `evaluate()` output corresponds
        to the order of metrics specified during `model.compile()`. You can
        use this order to map the `evaluate()` results to the intended
        metric. `model.metrics_names` itself will still return only the
        top-level names.
        """
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        """Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `BatchNormalization` that behave differently during
        inference.

        Note: See [this FAQ entry](
        https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
        for more details about the difference between `Model` methods
        `predict()` and `__call__()`.

        Args:
            x: Input data. It can be:
                - A NumPy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
                - A backend-native tensor, or a list of tensors
                (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
                - A `keras.utils.PyDataset`.
                - A `tf.data.Dataset`.
                - A `torch.utils.data.DataLoader`.
                - A Python generator function.
            batch_size: Integer or `None`.
                Number of samples per batch of computation.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                `keras.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            steps: Total number of steps (batches of samples) to draw before
                declaring the prediction round finished. If `steps` is `None`,
                it will run until `x` is exhausted. In the case of an infinitely
                repeating dataset, it will run indefinitely.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell the
                model to "pay more attention" to samples from an
                under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must
                be one-hot encoded, or an explicit final dimension of 1
                must be included for sparse class labels.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        """Test the model on a single batch of samples.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def get_compile_config(self):
        """Returns a serialized config with information for compiling the model.

        This method returns a config dictionary containing all the information
        (optimizer, loss, metrics, etc.) with which the model was compiled.

        Returns:
            A dict containing information for compiling the model.
        """
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()
        return {}

    def compile_from_config(self, config):
        """Compiles the model with the information given in config.

        This method uses the information in the config (optimizer, loss,
        metrics, etc.) to compile the model.

        Args:
            config: Dict containing information for compiling the model.
        """
        has_overridden_compile = self.__class__.compile != BaseTrainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _get_metrics_result_or_logs(self, logs):
        """Returns model metrics as a dict if the keys match with input logs.

        When the training / evaluation is performed with an asynchronous steps,
        the last scheduled `train / test_step` may not give the latest metrics
        because it is not guaranteed to be executed the last. This method gets
        metrics from the model directly instead of relying on the return from
        last step function.

        When the user has custom train / test step functions, the metrics
        returned may be different from `Model.metrics`. In those instances,
        this function will be no-op and return the logs passed in.

        Args:
            logs: A `dict` of metrics returned by train / test step function.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics`
            when logs and model metrics keys match. Otherwise it returns input
            `logs`.
        """
        metric_logs = self.get_metrics_result()
        # Verify that train / test step logs passed and metric logs have
        # matching keys. It could be different when using custom step functions,
        # in which case we return the logs from the last step.
        if isinstance(logs, dict) and set(logs.keys()) == set(
            metric_logs.keys()
        ):
            return metric_logs
        return logs

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = []
        for metric in self.metrics:
            if isinstance(metric, CompileMetrics):
                metric_names += [
                    sub_metric.name for sub_metric in metric.metrics
                ]
            else:
                metric_names.append(metric.name)
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _warn_if_trainable_state_changed(self):
        if not self.compiled or not hasattr(self, "_compiled_trainable_state"):
            return

        current_state = {
            layer: layer.trainable for layer in self._flatten_layers()
        }
        changed_layers = []
        for layer, trainable in current_state.items():
            if layer not in self._compiled_trainable_state:
                changed_layers.append(layer.name)
            elif trainable != self._compiled_trainable_state[layer]:
                changed_layers.append(layer.name)

        if changed_layers:
            warnings.warn(
                "A layer's `trainable` state has changed since the model was "
                "compiled. This can result in unexpected behavior, as the "
                "compiled optimizer and loss functions may not track the "
                "correct variables. Please recompile your model "
                "(e.g. `model.compile(...)`) before calling `fit()` or "
                "`evaluate()` so the change takes effect. "
                f"Changed layers: {changed_layers}",
                stacklevel=2,
            )
            # Update the state so we only warn once per compile
            self._compiled_trainable_state = current_state

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)
            self._warn_if_trainable_state_changed()

    def _get_unbuilt_components(self):
        """Returns a 4-tuple of booleans for unbuilt trainer components."""
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        compile_loss_unbuilt = (
            self._compile_loss is not None and not self._compile_loss.built
        )
        optimizer_unbuilt = (
            self.optimizer is not None and not self.optimizer.built
        )
        return (
            model_unbuilt,
            compile_metrics_unbuilt,
            compile_loss_unbuilt,
            optimizer_unbuilt,
        )

    def _symbolic_build(self, iterator=None, data_batch=None):
        (
            model_unbuilt,
            compile_metrics_unbuilt,
            compile_loss_unbuilt,
            optimizer_unbuilt,
        ) = self._get_unbuilt_components()
        if not (
            model_unbuilt
            or compile_metrics_unbuilt
            or compile_loss_unbuilt
            or optimizer_unbuilt
        ):
            return
        if model_unbuilt or compile_metrics_unbuilt or compile_loss_unbuilt:
            # Create symbolic tensors matching an input batch.

            def to_symbolic_input(v):
                if v is None:
                    return None
                return backend.KerasTensor(v.shape, standardize_dtype(v.dtype))

            if data_batch is None:
                for _, _, data_or_iterator in iterator:
                    if isinstance(data_or_iterator, (list, tuple)):
                        data_batch = data_or_iterator[0]
                    else:
                        data_batch = next(data_or_iterator)
                    break
            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)

            # Build all model state with `backend.compute_output_spec`.
            try:
                y_pred = backend.compute_output_spec(self, x, training=False)
            except Exception as e:
                # If the underlying failure is a ValueError (e.g. invalid
                # inputs provided by the user), propagate it so callers
                # (and tests) can observe the original error type. For
                # other unexpected exceptions, raise a RuntimeError with
                # context to guide the user.
                if isinstance(e, ValueError):
                    raise
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it.\n"
                    "Exception encountered:\n"
                    f"'{e}'"
                )
            if compile_metrics_unbuilt:
                # Build all metric state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
            if compile_loss_unbuilt:
                # Build `CompileLoss` state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self._compute_loss,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    training=False,
                )
        if optimizer_unbuilt:
            # Build optimizer
            self.optimizer.build(self.trainable_variables)
        self._post_build()


def model_supports_jit(model):
    # XLA not supported with TF on MacOS GPU
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        if backend.backend() == "tensorflow":
            from keras.src.utils.module_utils import tensorflow as tf

            if tf.config.list_physical_devices("GPU"):
                return False
    # XLA not supported by some layers
    if all(x.supports_jit for x in model._flatten_layers()):
        if backend.backend() == "tensorflow":
            from tensorflow.python.framework.config import (
                is_op_determinism_enabled,
            )

            if is_op_determinism_enabled():
                # disable XLA with determinism enabled since not all ops are
                # supported by XLA with determinism enabled.
                return False
        return True
    return False
