import inspect
import json
import typing
import warnings
from collections.abc import Callable

from keras.src import backend
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.layer import Layer
from keras.src.models.variable_mapping import map_saveable_variables
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.calibration_run import find_layers_in_block
from keras.src.quantizers.quantization_config import validate_and_resolve_config
from keras.src.quantizers.report import QuantizationReport
from keras.src.quantizers.utils import should_quantize_layer
from keras.src.saving import saving_api
from keras.src.utils import io_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils.module_utils import get_pluggable_backend_module

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.trainer import Trainer
elif backend.backend() == "jax":
    from keras.src.backend.jax.trainer import Trainer
elif backend.backend() == "torch":
    from keras.src.backend.torch.trainer import Trainer
elif backend.backend() == "numpy":
    from keras.src.backend.numpy.trainer import Trainer
else:
    backend_trainer_module = get_pluggable_backend_module("src.trainer")
    Trainer = getattr(backend_trainer_module, "Trainer")


@keras_export(["keras.Model", "keras.models.Model"])
class Model(Trainer, Layer):
    """A model grouping layers into an object with training/inference features.

    There are three ways to instantiate a `Model`:

    ## With the "Functional API"

    You start from `Input`,
    you chain layer calls to specify the model's forward pass,
    and finally, you create your model from inputs and outputs:

    ```python
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    ```

    Note: Only dicts, lists, and tuples of input tensors are supported. Nested
    inputs are not supported (e.g. lists of list or dicts of dict).

    A new Functional API model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.

    Example:

    ```python
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
    conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)

    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    ```

    Note that the `backbone` and `activations` models are not
    created with `keras.Input` objects, but with the tensors that originate
    from `keras.Input` objects. Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard Functional API models that support
    all the existing APIs.

    ## By subclassing the `Model` class

    In that case, you should define your
    layers in `__init__()` and you should implement the model's forward pass
    in `call()`.

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    model = MyModel()
    ```

    If you subclass `Model`, you can optionally have
    a `training` argument (boolean) in `call()`, which you can use to specify
    a different behavior in training and inference:

    ```python
    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(32, activation="relu")
            self.dense2 = keras.layers.Dense(5, activation="softmax")
            self.dropout = keras.layers.Dropout(0.5)

        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dropout(x, training=training)
            return self.dense2(x)

    model = MyModel()
    ```

    Once the model is created, you can config the model with losses and metrics
    with `model.compile()`, train the model with `model.fit()`, or use the model
    to do prediction with `model.predict()`.

    ## With the `Sequential` class

    In addition, `keras.Sequential` is a special case of model where
    the model is purely a stack of single-input, single-output layers.

    ```python
    model = keras.Sequential([
        keras.Input(shape=(None, None, 3)),
        keras.layers.Conv2D(filters=32, kernel_size=3),
    ])
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Model` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == Model:
            from keras.src.models.functional import Functional

            return Functional.__new__(Functional, *args, **kwargs)
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        from keras.src.models import functional

        # Signature detection for usage of a `Model` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_model_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)
        else:
            Layer.__init__(self, *args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @property
    def layers(self):
        return list(self._flatten_layers(include_self=False, recursive=False))

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    @traceback_utils.filter_traceback
    def get_layer(self, name=None, index=None):
        """Retrieves a layer based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of horizontal graph traversal (bottom-up).

        Args:
            name: String, name of layer.
            index: Integer, index of layer.

        Returns:
            A layer instance.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a layer name or a layer index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError(
                    f"Was asked to retrieve layer at index {index}"
                    f" but model only has {len(self.layers)}"
                    " layers."
                )
            else:
                return self.layers[index]

        if name is not None:
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError(
                f"No such layer: {name}. Existing layers are: "
                f"{list(layer.name for layer in self.layers)}."
            )
        raise ValueError(
            "Provide either a layer name or layer index at `get_layer`."
        )

    @traceback_utils.filter_traceback
    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
    ):
        """Prints a string summary of the network.

        Args:
            line_length: Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions: Relative or absolute positions of log elements
                in each line. If not provided, becomes
                `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
            print_fn: Print function to use. By default, prints to `stdout`.
                If `stdout` doesn't work in your environment, change to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested: Whether to expand the nested models.
                Defaults to `False`.
            show_trainable: Whether to show if a layer is trainable.
                Defaults to `False`.
            layer_range: a list or tuple of 2 strings,
                which is the starting layer name and ending layer name
                (both inclusive) indicating the range of layers to be printed
                in summary. It also accepts regex patterns instead of exact
                names. In this case, the start predicate will be
                the first element that matches `layer_range[0]`
                and the end predicate will be the last element
                that matches `layer_range[1]`.
                By default `None` considers all layers of the model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        summary_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            layer_range=layer_range,
        )

    @traceback_utils.filter_traceback
    def save(self, filepath, overwrite=True, zipped=None, **kwargs):
        """Saves a model as a `.keras` file.

        Note that `model.save()` is an alias for `keras.saving.save_model()`.

        The saved `.keras` file contains:

        - The model's configuration (architecture)
        - The model's weights
        - The model's optimizer's state (if any)

        Thus models can be reinstantiated in the exact same state.

        Args:
            filepath: `str` or `pathlib.Path` object.
                The path where to save the model. Must end in `.keras`
                (unless saving the model as an unzipped directory
                via `zipped=False`).
            overwrite: Whether we should overwrite any existing model at
                the target location, or instead ask the user via
                an interactive prompt.
            zipped: Whether to save the model as a zipped `.keras`
                archive (default when saving locally), or as an
                unzipped directory (default when saving on the
                Hugging Face Hub).

        Example:

        ```python
        model = keras.Sequential(
            [
                keras.layers.Dense(5, input_shape=(3,)),
                keras.layers.Softmax(),
            ],
        )
        model.save("model.keras")
        loaded_model = keras.saving.load_model("model.keras")
        x = keras.random.uniform((10, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))
        ```
        """
        return saving_api.save_model(
            self, filepath, overwrite=overwrite, zipped=zipped, **kwargs
        )

    @traceback_utils.filter_traceback
    def save_weights(self, filepath, overwrite=True, max_shard_size=None):
        """Saves all weights to a single file or sharded files.

        By default, the weights will be saved in a single `.weights.h5` file.
        If sharding is enabled (`max_shard_size` is not `None`), the weights
        will be saved in multiple files, each with a size at most
        `max_shard_size` (in GB). Additionally, a configuration file
        `.weights.json` will contain the metadata for the sharded files.

        The saved sharded files contain:

        - `*.weights.json`: The configuration file containing 'metadata' and
            'weight_map'.
        - `*_xxxxxx.weights.h5`: The sharded files containing only the
            weights.

        Args:
            filepath: `str` or `pathlib.Path` object. Path where the weights
                will be saved.  When sharding, the filepath must end in
                `.weights.json`. If `.weights.h5` is provided, it will be
                overridden.
            overwrite: Whether to overwrite any existing weights at the target
                location or instead ask the user via an interactive prompt.
            max_shard_size: `int` or `float`. Maximum size in GB for each
                sharded file. If `None`, no sharding will be done. Defaults to
                `None`.

        Example:

        ```python
        # Instantiate a EfficientNetV2L model with about 454MB of weights.
        model = keras.applications.EfficientNetV2L(weights=None)

        # Save the weights in a single file.
        model.save_weights("model.weights.h5")

        # Save the weights in sharded files. Use `max_shard_size=0.25` means
        # each sharded file will be at most ~250MB.
        model.save_weights("model.weights.json", max_shard_size=0.25)

        # Load the weights in a new model with the same architecture.
        loaded_model = keras.applications.EfficientNetV2L(weights=None)
        loaded_model.load_weights("model.weights.h5")
        x = keras.random.uniform((1, 480, 480, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))

        # Load the sharded weights in a new model with the same architecture.
        loaded_model = keras.applications.EfficientNetV2L(weights=None)
        loaded_model.load_weights("model.weights.json")
        x = keras.random.uniform((1, 480, 480, 3))
        assert np.allclose(model.predict(x), loaded_model.predict(x))
        ```
        """
        return saving_api.save_weights(
            self, filepath, overwrite=overwrite, max_shard_size=max_shard_size
        )

    @traceback_utils.filter_traceback
    def load_weights(self, filepath, skip_mismatch=False, **kwargs):
        """Load the weights from a single file or sharded files.

        Weights are loaded based on the network's topology. This means the
        architecture should be the same as when the weights were saved. Note
        that layers that don't have weights are not taken into account in the
        topological ordering, so adding or removing layers is fine as long as
        they don't have weights.

        **Partial weight loading**

        If you have modified your model, for instance by adding a new layer
        (with weights) or by changing the shape of the weights of a layer, you
        can choose to ignore errors and continue loading by setting
        `skip_mismatch=True`. In this case any layer with mismatching weights
        will be skipped. A warning will be displayed for each skipped layer.

        **Sharding**

        When loading sharded weights, it is important to specify `filepath` that
        ends with `*.weights.json` which is used as the configuration file.
        Additionally, the sharded files `*_xxxxx.weights.h5` must be in the same
        directory as the configuration file.

        Args:
            filepath: `str` or `pathlib.Path` object. Path where the weights
                will be saved.  When sharding, the filepath must end in
                `.weights.json`.
            skip_mismatch: Boolean, whether to skip loading of layers where
                there is a mismatch in the number of weights, or a mismatch in
                the shape of the weights.

        Example:

        ```python
        # Load the weights in a single file.
        model.load_weights("model.weights.h5")

        # Load the weights in sharded files.
        model.load_weights("model.weights.json")
        ```
        """
        saving_api.load_weights(
            self,
            filepath,
            skip_mismatch=skip_mismatch,
            **kwargs,
        )

    def get_quantization_layer_structure(self, mode=None):
        """Returns the quantization structure for the model.

        This method is intended to be overridden by model authors to provide
        topology information required for structure-aware quantization modes
        like 'gptq'.

        Args:
            mode: The quantization mode.

        Returns:
            A dictionary describing the topology, e.g.:
            `{'pre_block_layers': [list], 'sequential_blocks': [list]}`
            or `None` if the mode does not require structure or is not
            supported. `'pre_block_layers'` is a list of layers that
            the inputs should be passed through, before being passed to
            the sequential blocks. For example, inputs to an LLM must
            first be passed through an embedding layer, followed by
            the transformer. `'pre_block_layers'` may be left out, and
            each value is a list or a tuple. A structure with any other
            key is refused with a `ValueError`.
        """
        del mode  # Unused.
        return None

    def quantize(
        self, mode=None, config=None, filters=None, verbose=True, **kwargs
    ):
        """Quantize the weights of the model.

        Note that the model must be built first before calling this method.
        `quantize` will recursively call `quantize(...)` in all layers and
        will be skipped if the layer doesn't implement the function.

        This method can be called by passing a `mode` string, which uses the
        default configuration for that mode. Alternatively, a `config` object
        can be passed to customize the behavior of the quantization (e.g. to
        use specific quantizers for weights or activations).

        For the `"gptq"` and `"awq"` modes, quantization is restricted to
        the `Dense` and `EinsumDense` layers inside the structure's
        `"sequential_blocks"` (provided via
        `config.quantization_layer_structure` or the model's
        `get_quantization_layer_structure(mode)` hook). All other layers
        are left in their original precision.

        The call collects a `QuantizationReport` describing which layers were
        quantized and which were skipped (and why). The report is returned and
        also stored on the model as `model._quantization_report`. A single
        summary warning is emitted if any layers could not be quantized because
        they do not support quantization. For a per-layer view of storage sizes,
        call `model.quantization_summary()`.

        Args:
            mode: The mode of the quantization. Supported modes are:
                `"int8"`, `"int4"`, `"float8"`, `"ternary"`, `"gptq"` and
                `"awq"`. This is optional if `config` is provided. Passing a
                bare string uses the default configuration for that mode,
                which is identical to passing the corresponding config object
                with default arguments (e.g. `quantize("int4")` matches
                `quantize(config=Int4QuantizationConfig())`). The activation
                (A) times weight (W) semantics of each mode are:

                -   `"int8"`: **W8A8 dynamic**. Weights and activations are both
                    quantized to `int8` and multiplied with a real integer
                    GEMM; activation scales are computed dynamically per call.
                -   `"int4"`: **W4A16 weight-only**, **grouped** with
                    `block_size=128` by default. Only weights are quantized (to
                    4 bits, packed two per `int8` byte); this is storage-only
                    today (weights are dequantized to float before each
                    matmul). Pass `Int4QuantizationConfig(block_size=-1)` for
                    per-channel.
                -   `"float8"`: **float8 QDQ** mixed-precision training scheme
                    (not post-training compression); weights and activations
                    are dynamically cast to `float8` during training.
                -   `"ternary"`: **weight-only** BitNet b1.58 quantization of
                    `Dense` layers. Weights are quantized to `{-1, 0, +1}`
                    with one scalar scale and packed five per byte; this is
                    storage-only today (weights are unpacked to float before
                    each matmul).
                -   `"gptq"`: **weight-only** post-training quantization to
                    2, 3, 4 or 8 bits (`GPTQConfig.weight_bits`, 4 by
                    default); requires a `GPTQConfig` passed via `config`.
                -   `"awq"`: **W4A16 weight-only** post-training quantization;
                    requires an `AWQConfig` passed via `config`.
            config: The configuration object specifying additional
                quantization options. This argument allows to configure
                the weight and activation quantizers. It must be an instance of
                `keras.quantizers.QuantizationConfig`.
            filters: Optional filters to apply to the quantization. Can be a
                regex string, a list of regex strings, or a callable. Only the
                layers which match the filter conditions will be quantized.
            verbose: Whether to print the quantization report. Defaults to
                `True`.
            **kwargs: Additional keyword arguments.

        Returns:
            A `QuantizationReport` describing the outcome of the call.

        Example:

        Quantize a model to int8 with default configuration:

        ```python
        # Build the model
        model = keras.Sequential([
            keras.Input(shape=(10,)),
            keras.layers.Dense(10),
        ])
        model.build((None, 10))

        # Quantize with default int8 config
        model.quantize("int8")
        ```

        Quantize a model to int8 with a custom configuration:

        ```python
        from keras.quantizers import Int8QuantizationConfig
        from keras.quantizers import AbsMaxQuantizer

        # Build the model
        model = keras.Sequential([
            keras.Input(shape=(10,)),
            keras.layers.Dense(10),
        ])
        model.build((None, 10))

        # Create a custom config
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(
                axis=0,
                value_range=(-127, 127)
            ),
            activation_quantizer=AbsMaxQuantizer(
                axis=-1,
                value_range=(-127, 127)
            ),
        )

        # Quantize with custom config
        model.quantize(config=config)
        ```
        """
        # Resolve `mode`/`config` into a concrete mode string and
        # `QuantizationConfig`, and validate that the model can be quantized,
        # before validating the remaining arguments (mirroring
        # `Layer.quantize`'s validation order).
        config = validate_and_resolve_config(mode, config)
        mode = config.mode
        self._check_quantize_args(mode, self.compute_dtype)

        # Validate inputs.
        type_check = kwargs.pop("type_check", True)
        if kwargs:
            raise ValueError(
                "Unrecognized keyword arguments "
                f"passed to {self.__class__.__name__}: {kwargs}"
            )

        if filters is not None:
            if not isinstance(filters, (str, Callable, list, tuple)):
                raise ValueError(
                    "The `filters` argument must be a regex string, a list of "
                    "regex strings, or a callable. Received: "
                    f"{type(filters)}"
                )

        # For structure-aware modes (`gptq`/`awq`), resolve and validate the
        # layer structure *before* mutating any layer, and restrict
        # quantization to the layers covered by the structure. Resolving it
        # here (rather than after the mutation loop) guarantees that a
        # missing/invalid structure leaves the model completely untouched
        # instead of half-quantized (buffers allocated, dtype policies
        # swapped). Only structure-covered layers are calibrated afterwards;
        # quantizing any other layer would leave it uncalibrated, and its
        # uninitialized quantized weights would silently replace the real
        # ones when the model is saved and reloaded.
        strategy = strategy_registry.get_strategy(mode)
        structure = None
        structure_layer_ids = None
        if strategy.requires_layer_structure:
            # 1. If quantization_layer_structure is provided inside the
            # config, use that.
            structure = config.quantization_layer_structure
            # 2. If no layer structure is provided in the config, try to
            # fetch it using the `get_quantization_layer_structure` hook.
            if structure is None:
                structure = self.get_quantization_layer_structure(mode)

            if structure is None:
                raise ValueError(
                    f"For {mode=}, a valid quantization structure must be "
                    "provided either via `config.quantization_layer_structure` "
                    "or by overriding "
                    "`model.get_quantization_layer_structure(mode)`. The "
                    "structure should be a dictionary with keys "
                    "'pre_block_layers' and 'sequential_blocks'."
                )
            structure_layer_ids = set()
            for block in structure.get("sequential_blocks", []):
                for sub_layer in find_layers_in_block(block).values():
                    structure_layer_ids.add(id(sub_layer))

        report = QuantizationReport(mode=mode)
        graph_modified = False
        for layer in self._flatten_layers():
            # Skip nested models: this walk already visits their layers
            # directly (`_flatten_layers` is recursive), and calling
            # `quantize` on a sub-model would recurse into a second full
            # walk. Owning sub-layers does NOT make a layer unquantizable
            # (e.g. `Dense` with a `Layer` activation), so this is an
            # isinstance test, not a topology test; any other layer that
            # cannot be quantized reports itself by raising
            # `NotImplementedError` below.
            if isinstance(layer, Model):
                continue

            # Input layers carry no weights; they are not a meaningful skip.
            if isinstance(layer, InputLayer):
                continue

            path = layer.path or layer.name

            # 1. For GPTQ/AWQ, layers outside the structure's sequential
            # blocks are never calibrated, so they must not be quantized at
            # all.
            if (
                structure_layer_ids is not None
                and id(layer) not in structure_layer_ids
            ):
                report.add_skipped(
                    path, QuantizationReport.SKIP_OUTSIDE_STRUCTURE
                )
                continue
            # 2. Excluded by `filters`.
            if not should_quantize_layer(layer, filters):
                report.add_skipped(path, QuantizationReport.SKIP_FILTERED)
                continue
            # 3. Already quantized (e.g. a previously quantized layer).
            if layer._is_quantized:
                report.add_skipped(
                    path, QuantizationReport.SKIP_ALREADY_QUANTIZED
                )
                continue
            # 4. Attempt to quantize. Only `NotImplementedError` means the
            # layer does not support quantization; any other exception is a
            # real bug and is allowed to propagate.
            try:
                layer.quantize(mode, type_check=type_check, config=config)
            except NotImplementedError:
                report.add_skipped(path, QuantizationReport.SKIP_NO_SUPPORT)
                continue
            report.add_quantized(
                path, layer.quantization_mode, layer._own_dtype_policy.name
            )
            graph_modified = True

        # Structure-aware modes run their calibration pass here (a no-op
        # for the other modes).
        strategy.finalize_model_quantization(self, config, structure, filters)

        # Emit a single summary warning in place of the previous per-layer
        # warning storm (one `UserWarning` per non-quantizable leaf).
        warning_message = report.summary_warning()
        if warning_message is not None:
            warnings.warn(warning_message, stacklevel=2)

        if verbose:
            io_utils.print_msg(report.render())

        self._quantization_report = report

        # If any layer was changed, we must rebuild the execution functions.
        if graph_modified:
            self.train_function = None
            self.test_function = None
            self.predict_function = None
            self._post_quantize(mode, **kwargs)

        return report

    def quantization_summary(self, verbose=True):
        """Print a per-quantized-layer summary of the model.

        For every quantized layer this reports the layer path, its own dtype
        policy (its entry, for a layer that holds a `DTypePolicyMap`), the
        storage dtype of its quantized weight, and the bytes the weight
        stores: codes, scale, zero point, group index and input scales. A
        mode that keeps the float weight (float8) reports that weight and
        says so. It also prints totals comparing the quantized weight
        storage against a float32 baseline.

        Args:
            verbose: Whether to print the summary. Defaults to `True`. The
                summary string is always returned.

        Returns:
            The summary as a string.
        """
        return summary_utils.print_quantization_summary(self, verbose=verbose)

    def _post_quantize(self, mode, **kwargs):
        if backend.backend() == "torch":
            # Local import: `sequential` imports this module.
            from keras.src.models.sequential import Sequential

            # We need to manually retrack `torch_params`.
            # The reason is that after quantization, the removed variables are
            # still referenced by `torch_params` and cannot be gc.
            for layer in self._flatten_layers():
                layer._track_variables()
                # A `Sequential` runs its layers through the `Functional` in
                # `_functional`, which `_flatten_layers` does not visit. Its
                # `torch_params` also reference the removed variables.
                if (
                    isinstance(layer, Sequential)
                    and layer._functional is not None
                ):
                    layer._functional._track_variables()

    def build_from_config(self, config):
        if not config:
            return
        status = False
        if "input_shape" in config:
            # Case: all inputs are in the first arg (possibly nested).
            if utils.is_default(self.build):
                status = self._build_by_run_for_single_pos_arg(
                    config["input_shape"]
                )
            else:
                try:
                    self.build(config["input_shape"])
                    status = True
                except:
                    pass
            self._build_shapes_dict = config

        elif "shapes_dict" in config:
            # Case: inputs were recorded as multiple keyword arguments.
            if utils.is_default(self.build):
                status = self._build_by_run_for_kwargs(config["shapes_dict"])
            else:
                try:
                    self.build(**config["shapes_dict"])
                    status = True
                except:
                    pass
            self._build_shapes_dict = config["shapes_dict"]

        if not status:
            warnings.warn(
                f"Model '{self.name}' had a build config, but the model "
                "cannot be built automatically in "
                "`build_from_config(config)`. "
                "You should implement "
                "`def build_from_config(self, config)`, "
                "and you might also want to implement the method "
                " that generates the config at saving time, "
                "`def get_build_config(self)`. "
                "The method `build_from_config()` is meant to "
                "create the state of the model (i.e. its variables) "
                "upon deserialization.",
                stacklevel=2,
            )

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        To load a network from a JSON save file, use
        `keras.models.model_from_json(json_string, custom_objects={...})`.

        Args:
            **kwargs: Additional keyword arguments to be passed to
                `json.dumps()`.

        Returns:
            A JSON string.
        """
        from keras.src.saving import serialization_lib

        model_config = serialization_lib.serialize_keras_object(self)
        return json.dumps(model_config, **kwargs)

    def export(
        self,
        filepath,
        format="tf_saved_model",
        verbose=None,
        input_signature=None,
        **kwargs,
    ):
        """Export the model as an artifact for inference.

        Args:
            filepath: `str` or `pathlib.Path` object. The path to save the
                artifact.
            format: `str`. The export format. Supported values:
                `"tf_saved_model"`, `"onnx"`, `"openvino"`, `"litert"`,
                and `"torch"`. Defaults to `"tf_saved_model"`.
            verbose: `bool`. Whether to print a message during export. Defaults
                to `None`, which uses the default value set by different
                backends and formats.
            input_signature: Optional. Specifies the shape and dtype of the
                model inputs. Can be a structure of `keras.InputSpec`,
                `tf.TensorSpec`, `backend.KerasTensor`, or backend tensor. If
                not provided, it will be automatically computed. Defaults to
                `None`.
                Note: With `format="litert"` and the PyTorch backend, dynamic
                input shapes are not supported. Any dynamic dimensions (i.e.,
                `None` in input shapes) will be automatically replaced with `1`
                during export, which may cause runtime failures for other
                shapes. You must explicitly pass a fixed static
                `input_signature` matching your maximum runtime shape.
            **kwargs: Additional keyword arguments.
                - `is_static`: Optional `bool`. Specific to the JAX backend and
                    `format="tf_saved_model"`. Indicates whether `fn` is static.
                    Set to `False` if `fn` involves state updates (e.g., RNG
                    seeds and counters).
                - `jax2tf_kwargs`: Optional `dict`. Specific to the JAX backend
                    and `format="tf_saved_model"`. Arguments for
                    `jax2tf.convert`. See the documentation for
                    [`jax2tf.convert`](
                        https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
                    If `native_serialization` and `polymorphic_shapes` are not
                    provided, they will be automatically computed.
                - `opset_version`: Optional `int`. Specific to `format="onnx"`.
                    An integer value that specifies the ONNX opset version.
                - LiteRT-specific options: Optional keyword arguments specific
                    to `format="litert"`. These are passed directly to the
                    TensorFlow Lite converter on the TensorFlow backend and
                    include options like `optimizations`,
                    `representative_dataset`, `experimental_new_quantizer`,
                    `allow_custom_ops`, `enable_select_tf_ops`, etc. On the
                    PyTorch backend, LiteRT export accepts `optimizations`
                    plus the installed `litert_torch.convert()` keyword
                    arguments such as `strict_export`, `dynamic_shapes` (note
                    that standard ops do not support dynamic shapes, as noted
                    below), `lightweight_conversion`, `enable_x64`,
                    `runtime_constant_folding`, and `quant_config`.
                - PyTorch export options: Optional keyword arguments specific
                    to `format="torch"`. These are passed directly to
                    `torch.export.export` and include `strict`,
                    `dynamic_shapes`,
                    `prefer_deferred_runtime_asserts_over_guards`, and
                    `preserve_module_call_signature`.

        **Note on LiteRT (TFLite) Export with PyTorch Backend:**
        With the PyTorch backend, LiteRT export (`format="litert"`) does not
        support dynamic input shapes. If no static signature is provided,
        any dynamic dimensions (represented as `None`) are automatically
        replaced with `1` during export. This can lead to runtime failures
        for other shapes. You must explicitly specify a fixed static
        `input_signature` (matching your maximum runtime dimensions) and pad
        your inputs to this static shape at runtime.

        **Note:** This feature is currently supported only with TensorFlow, JAX
        and Torch backends.

        **Note:** Be aware that the exported artifact may contain information
        from the local file system when using `format="onnx"`, `verbose=True`
        and Torch backend.

        Examples:

        Here's how to export a TensorFlow SavedModel for inference.

        ```python
        # Export the model as a TensorFlow SavedModel artifact
        model.export("path/to/location", format="tf_saved_model")

        # Load the artifact in a different process/environment
        reloaded_artifact = tf.saved_model.load("path/to/location")
        predictions = reloaded_artifact.serve(input_data)
        ```

        Here's how to export an ONNX for inference.

        ```python
        # Export the model as a ONNX artifact
        model.export("path/to/location", format="onnx")

        # Load the artifact in a different process/environment
        ort_session = onnxruntime.InferenceSession("path/to/location")
        ort_inputs = {
            k.name: v for k, v in zip(ort_session.get_inputs(), input_data)
        }
        predictions = ort_session.run(None, ort_inputs)
        ```

        Here's how to export a LiteRT (TFLite) for inference.

        ```python
        # Export the model as a LiteRT artifact
        model.export("path/to/location", format="litert")

        # Load the artifact in a different process/environment
        interpreter = tf.lite.Interpreter(model_path="path/to/location")
        interpreter.allocate_tensors()
        interpreter.set_tensor(
            interpreter.get_input_details()[0]['index'], input_data
        )
        interpreter.invoke()
        output_data = interpreter.get_tensor(
            interpreter.get_output_details()[0]['index']
        )
        ```

        Here's how to export a PyTorch ExportedProgram for inference.

        ```python
        # Export the model as a PyTorch ExportedProgram artifact
        model.export("path/to/model.pt2", format="torch")

        # Load the artifact in a different process/environment
        import torch
        loaded_program = torch.export.load("path/to/model.pt2")
        predictions = loaded_program.module()(input_tensor)
        ```
        """
        from keras.src.export import export_litert
        from keras.src.export import export_onnx
        from keras.src.export import export_openvino
        from keras.src.export import export_saved_model
        from keras.src.export import export_torch

        available_formats = (
            "tf_saved_model",
            "onnx",
            "openvino",
            "litert",
            "torch",
        )
        if format not in available_formats:
            raise ValueError(
                f"Unrecognized format={format}. Supported formats are: "
                f"{list(available_formats)}."
            )

        # Check if LiteRT export is available (requires TensorFlow or
        # PyTorch backend)
        if format == "litert" and backend.backend() not in (
            "tensorflow",
            "torch",
        ):
            raise ValueError(
                "LiteRT export requires TensorFlow or PyTorch backend."
            )

        # Check if Torch export is available (requires PyTorch backend)
        if format == "torch" and backend.backend() != "torch":
            raise ValueError("Torch export requires PyTorch backend.")

        if format == "tf_saved_model":
            export_saved_model(
                self,
                filepath,
                verbose,
                input_signature=input_signature,
                **kwargs,
            )
        elif format == "onnx":
            export_onnx(
                self,
                filepath,
                verbose,
                input_signature=input_signature,
                **kwargs,
            )
        elif format == "openvino":
            export_openvino(
                self,
                filepath,
                verbose,
                input_signature=input_signature,
                **kwargs,
            )
        elif format == "litert":
            export_litert(
                self,
                filepath,
                verbose=verbose,
                input_signature=input_signature,
                **kwargs,
            )
        elif format == "torch":
            export_torch(
                self,
                filepath,
                verbose=verbose,
                input_signature=input_signature,
                **kwargs,
            )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.src.models.functional import Functional

        functional_config_keys = [
            "name",
            "layers",
            "input_layers",
            "output_layers",
        ]
        is_functional_config = all(
            key in config for key in functional_config_keys
        )
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[
            1:
        ]
        revivable_as_functional = (
            cls in {Functional, Model}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            from keras.src.models.functional import functional_from_config

            return functional_from_config(
                cls, config, custom_objects=custom_objects
            )

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive model from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    def _get_variable_map(self):
        store = {}
        map_saveable_variables(self, store=store, visited_saveables=set())
        return store

    def get_state_tree(self, value_format="backend_tensor"):
        """Retrieves tree-like structure of model variables.

        This method allows retrieval of different model variables (trainable,
        non-trainable, optimizer, and metrics). The variables are returned in a
        nested dictionary format, where the keys correspond to the variable
        names and the values are the nested representations of the variables.

        Returns:
            dict: A dictionary containing the nested representations of the
                requested variables. The keys are the variable names, and the
                values are the corresponding nested dictionaries.
            value_format: One of `"backend_tensor"`, `"numpy_array"`.
                The kind of array to return as the leaves of the nested
                    state tree.

        Example:

        ```python
        model = keras.Sequential([
            keras.Input(shape=(1,), name="my_input"),
            keras.layers.Dense(1, activation="sigmoid", name="my_dense"),
        ], name="my_sequential")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        model.fit(np.array([[1.0]]), np.array([[1.0]]))
        state_tree = model.get_state_tree()
        ```

        The `state_tree` dictionary returned looks like:

        ```
        {
            'metrics_variables': {
                'loss': {
                    'count': ...,
                    'total': ...,
                },
                'mean_absolute_error': {
                    'count': ...,
                    'total': ...,
                }
            },
            'trainable_variables': {
                'my_sequential': {
                    'my_dense': {
                        'bias': ...,
                        'kernel': ...,
                    }
                }
            },
            'non_trainable_variables': {},
            'optimizer_variables': {
                'adam': {
                        'iteration': ...,
                        'learning_rate': ...,
                        'my_sequential_my_dense_bias_momentum': ...,
                        'my_sequential_my_dense_bias_velocity': ...,
                        'my_sequential_my_dense_kernel_momentum': ...,
                        'my_sequential_my_dense_kernel_velocity': ...,
                    }
                }
            }
        }
        ```
        """
        variables = {}
        variables["trainable_variables"] = self._create_nested_dict(
            self.trainable_variables, value_format
        )
        variables["non_trainable_variables"] = self._create_nested_dict(
            self.non_trainable_variables, value_format
        )
        variables["optimizer_variables"] = self._create_nested_dict(
            self.optimizer.variables, value_format
        )
        variables["metrics_variables"] = self._create_nested_dict(
            self.metrics_variables, value_format
        )
        return variables

    def _create_nested_dict(self, variables, value_format):
        flat_dict = {}
        for v in variables:
            if v.path in flat_dict:
                raise ValueError(
                    "The following variable path is found twice in the model: "
                    f"'{v.path}'. `get_state_tree()` can only be called when "
                    "all variable paths are unique. Make sure to give unique "
                    "names to your layers (and other objects)."
                )
            if value_format == "backend_tensor":
                flat_dict[v.path] = v.value
            elif value_format == "numpy_array":
                flat_dict[v.path] = v.numpy()
            else:
                raise ValueError(
                    "Invalid `value_format` argument. Expected one of "
                    "{'numpy_array', 'backend_tensor'}. Received: "
                    f"value_format={value_format}"
                )

        nested_dict = {}
        for path, value in flat_dict.items():
            parts = path.split("/")
            current_dict = nested_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = value

        return nested_dict

    def set_state_tree(self, state_tree):
        """Assigns values to variables of the model.

        This method takes a dictionary of nested variable values, which
        represents the state tree of the model, and assigns them to the
        corresponding variables of the model. The dictionary keys represent the
        variable names (e.g., `'trainable_variables'`, `'optimizer_variables'`),
        and the values are nested dictionaries containing the variable
        paths and their corresponding values.

        Args:
            state_tree: A dictionary representing the state tree of the model.
                The keys are the variable names, and the values are nested
                dictionaries representing the variable paths and their values.
        """
        for k, v in state_tree.items():
            path_value_dict = self._flatten_nested_dict(v)
            if k == "trainable_variables":
                self._assign_variable_values(
                    self.trainable_variables, path_value_dict
                )
            elif k == "non_trainable_variables":
                self._assign_variable_values(
                    self.non_trainable_variables, path_value_dict
                )
            elif k == "optimizer_variables":
                if hasattr(self, "optimizer") and self.optimizer is not None:
                    self._assign_variable_values(
                        self.optimizer.variables, path_value_dict
                    )
            elif k == "metrics_variables":
                if (
                    hasattr(self, "metrics_variables")
                    and self.metrics_variables
                ):
                    self._assign_variable_values(
                        self.metrics_variables, path_value_dict
                    )
            else:
                raise ValueError(f"Unknown variable name: {k}")

    def _assign_variable_values(self, variables, path_value_dict):
        for path, value in path_value_dict.items():
            for variable in variables:
                if variable.path == path:
                    variable.assign(value)

    def _flatten_nested_dict(self, nested_dict):
        flat_dict = {}

        def _flatten(current_dict, prefix=""):
            for key, value in current_dict.items():
                if isinstance(value, dict):
                    _flatten(value, f"{prefix}{key}/")
                else:
                    flat_dict[f"{prefix}{key}"] = value

        _flatten(nested_dict)
        return flat_dict


@keras_export("keras.models.model_from_json")
def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration string and returns a model instance.

    Example:

    >>> model = keras.Sequential([
    ...     keras.layers.Dense(5, input_shape=(3,)),
    ...     keras.layers.Softmax()])
    >>> config = model.to_json()
    >>> loaded_model = keras.models.model_from_json(config)

    Args:
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A Keras model instance (uncompiled).
    """
    from keras.src.saving import serialization_lib

    model_config = json.loads(json_string)
    return serialization_lib.deserialize_keras_object(
        model_config, custom_objects=custom_objects
    )


def functional_init_arguments(args, kwargs):
    return (
        (len(args) == 2)
        or (len(args) == 1 and "outputs" in kwargs)
        or ("inputs" in kwargs and "outputs" in kwargs)
    )


def inject_functional_model_class(cls):
    """Inject `Functional` into the hierarchy of this class if needed."""
    from keras.src.models import functional

    if cls is Model:
        return functional.Functional
    # In case there is any multiple inheritance, we stop injecting the
    # class if keras model is not in its class hierarchy.
    if cls is object:
        return object

    cls.__bases__ = tuple(
        inject_functional_model_class(base) for base in cls.__bases__
    )
    # Trigger any `__new__` class swapping that needed to happen on `Functional`
    # but did not because functional was not in the class hierarchy.
    cls.__new__(cls)

    return cls
