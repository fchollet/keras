"""Layer is an Operation with state.

Takes care of:

- Weights / variables (and tracking thereof)
- deferred build
- trainable argument value inference
- masking
- autocasting

And some more magic:

- add_loss
- metric tracking
- RNG seed tracking
- activity regularization
"""

import collections
import functools
import inspect
import warnings
from functools import wraps

from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import AutocastScope
from keras.src.backend.common import get_autocast_scope
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import global_state
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import remat
from keras.src.backend.common import set_keras_mask
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common import standardize_shape
from keras.src.backend.common.keras_tensor import any_symbolic_tensors
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.remat import get_current_remat_mode
from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.config import is_nnx_enabled
from keras.src.distribution import distribution_lib
from keras.src.dtype_policies import DTypePolicyMap
from keras.src.layers import input_spec
from keras.src.metrics.metric import Metric
from keras.src.ops.node import Node
from keras.src.ops.operation import Operation
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.quantization_config import validate_and_resolve_config
from keras.src.utils import python_utils
from keras.src.utils import summary_utils
from keras.src.utils import traceback_utils
from keras.src.utils import tracking
from keras.src.utils.module_utils import get_pluggable_backend_module

if backend.backend() == "tensorflow":
    from keras.src.backend.tensorflow.layer import TFLayer as BackendLayer
elif backend.backend() == "jax":
    from keras.src.backend.jax.layer import JaxLayer as BackendLayer
elif backend.backend() == "torch":
    from keras.src.backend.torch.layer import TorchLayer as BackendLayer
elif backend.backend() == "numpy":
    from keras.src.backend.numpy.layer import NumpyLayer as BackendLayer
else:
    backend_layer_module = get_pluggable_backend_module(
        "src.layer", allow_missing=True
    )
    if backend_layer_module is not None:
        BackendLayer = getattr(backend_layer_module, "BackendLayer")
    else:

        class BackendLayer:
            pass


@keras_export(["keras.Layer", "keras.layers.Layer"])
class Layer(BackendLayer, Operation):
    """This is the class from which all layers inherit.

    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_weight()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the layer, and supplies the shape(s) of the input(s),
      which may not have been known at initialization time.

    Layers are recursively composable: If you assign a Layer instance as an
    attribute of another Layer, the outer layer will start tracking the weights
    created by the inner layer. Nested layers should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a layer and then treat it as a callable.

    Args:
        trainable: Boolean, whether the layer's variables should be trainable.
        name: String name of the layer.
        dtype: The dtype of the layer's computations and weights. Can also be a
            `keras.DTypePolicy`, which allows the computation and weight dtype
            to differ. Defaults to `None`. `None` means to use
            `keras.config.dtype_policy()`, which is a `float32` policy unless
            set to different value (via `keras.config.set_dtype_policy()`).

    Attributes:
        name: The name of the layer (string).
        dtype: Dtype of the layer's weights. Alias of `layer.variable_dtype`.
        variable_dtype: Dtype of the layer's weights.
        compute_dtype: The dtype of the layer's computations.
            Layers automatically cast inputs to this dtype, which causes
            the computations and output to also be in this dtype.
            When mixed precision is used with a
            `keras.DTypePolicy`, this will be different
            than `variable_dtype`.
        trainable_weights: List of variables to be included in backprop.
        non_trainable_weights: List of variables that should not be
            included in backprop.
        weights: The concatenation of the lists trainable_weights and
            non_trainable_weights (in this order).
        trainable: Whether the layer should be trained (boolean), i.e.
            whether its potentially-trainable weights should be returned
            as part of `layer.trainable_weights`.
        input_spec: Optional (list of) `InputSpec` object(s) specifying the
            constraints on inputs that can be accepted by the layer.

    We recommend that descendants of `Layer` implement the following methods:

    * `__init__()`: Defines custom layer attributes, and creates layer weights
        that do not depend on input shapes, using `add_weight()`,
        or other state.
    * `build(self, input_shape)`: This method can be used to create weights that
        depend on the shape(s) of the input(s), using `add_weight()`, or other
        state. `__call__()` will automatically build the layer
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the layer to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
            2. `mask` (boolean tensor encoding masked timesteps in the input,
                used e.g. in RNN layers).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` and `mask` if the layer need them.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this layer. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the layer or a model that contains this layer.

    Examples:

    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).

    ```python
    class SimpleDense(Layer):
        def __init__(self, units=32):
            super().__init__()
            self.units = units

        # Create the state of the layer (weights)
        def build(self, input_shape):
            self.kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="kernel",
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )

        # Defines the computation
        def call(self, inputs):
            return ops.matmul(inputs, self.kernel) + self.bias

    # Instantiates the layer.
    linear_layer = SimpleDense(4)

    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(ops.ones((2, 2)))
    assert len(linear_layer.weights) == 2

    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```

    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:

    ```python
    class ComputeSum(Layer):

      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            name="total",
          )

      def call(self, inputs):
          self.total.assign(self.total + ops.sum(inputs))
          return self.total

    my_sum = ComputeSum(2)
    x = ops.ones((2, 2))
    y = my_sum(x)

    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```
    """

    # Whether this layer's own weights carry a LoRA update. A layer that
    # supports LoRA sets it in `enable_lora()`.
    lora_enabled = False

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        # Wrap the user-provided `build` method in the `build_wrapper`
        # to add name scope support and serialization support.
        original_build_method = obj.build

        @wraps(original_build_method)
        def build_wrapper(*args, **kwargs):
            with obj._open_name_scope():
                obj._path = current_path()
                original_build_method(*args, **kwargs)
            # Record build config.
            signature = inspect.signature(original_build_method)
            obj._build_shapes_dict = signature.bind(*args, **kwargs).arguments
            # Set built, post build actions, and lock state.
            obj.built = True
            obj._post_build()
            obj._lock_state()

        obj.build = build_wrapper

        return obj

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ):
        BackendLayer.__init__(self)
        self._lock = False
        Operation.__init__(self, name=name)
        self._dtype_policy = dtype_policies.get(dtype)
        # Quantization bookkeeping, always initialized so the quantization
        # paths read it directly: whether `quantized_build` has created the
        # mode's variables, and the `QuantizationConfig` recorded by
        # `quantize()` (subclasses with a `quantization_config` constructor
        # argument overwrite the latter).
        self._is_quantized = False
        self.quantization_config = None
        self.activity_regularizer = regularizers.get(activity_regularizer)
        input_dim_arg = kwargs.pop("input_dim", None)
        if input_dim_arg is not None:
            input_shape_arg = (input_dim_arg,)
        else:
            input_shape_arg = kwargs.pop("input_shape", None)
        if input_shape_arg is not None:
            warnings.warn(
                "Do not pass an `input_shape`/`input_dim` argument to "
                "a layer. When using Sequential models, "
                "prefer using an `Input(shape)` object as the "
                "first layer in the model instead.",
                stacklevel=2,
            )
            self._input_shape_arg = input_shape_arg
        if kwargs:
            raise ValueError(
                "Unrecognized keyword arguments "
                f"passed to {self.__class__.__name__}: {kwargs}"
            )

        self._path = None  # Will be determined in `build_wrapper`
        self.built = False
        self.autocast = autocast
        self._input_spec = None
        self._called = False
        self.supports_jit = True

        if not isinstance(trainable, bool):
            raise ValueError(
                "Expected `trainable` to be a boolean. "
                f"Received: trainable={trainable} (of type "
                f"{type(trainable)})"
            )
        self._trainable = trainable
        self._losses = []
        self._loss_ids = set()
        self._losses_override = []

        self._call_signature = inspect.signature(self.call)
        self.call_signature_parameters = [
            p.name for p in self._call_signature.parameters.values()
        ]
        self._call_has_training_arg = (
            "training" in self.call_signature_parameters
        )
        self._call_has_mask_arg = "mask" in self.call_signature_parameters

        # 1. collect names that should be auto‑propagated
        self._call_context_args = {"training"}

        # 2. remember which of them exist in *this* call signature
        self._call_has_context_arg = {
            arg: (arg in self.call_signature_parameters)
            for arg in self._call_context_args
        }

        self._supports_masking = not utils.is_default(self.compute_mask)
        # Whether to automatically convert (+ auto-cast) inputs to `call()`.
        self._convert_input_args = True
        # Whether to allow non-tensors as positional arguments in `call()`.
        self._allow_non_tensor_positional_args = False
        # Dict of shapes that were used to call `build()`.
        self._build_shapes_dict = None
        # Parent path
        self._parent_path = None
        self._initialize_tracker()

    @tracking.no_automatic_dependency_tracking
    def _initialize_tracker(self):
        if hasattr(self, "_tracker"):
            return

        trainable_variables = []
        non_trainable_variables = []
        layers = []
        metrics = []
        seed_generators = []
        self._tracker = tracking.Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: (
                        isinstance(x, backend.Variable) and not x.trainable
                    ),
                    non_trainable_variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), metrics),
                "layers": (
                    lambda x: (
                        isinstance(x, Layer) and not isinstance(x, Metric)
                    ),
                    layers,
                ),
                "seed_generators": (
                    lambda x: isinstance(x, backend.random.SeedGenerator),
                    seed_generators,
                ),
            },
            exclusions={"non_trainable_variables": ["trainable_variables"]},
        )
        if backend.backend() == "tensorflow":
            # Remove attribute tracking for lists (TF-specific attribute)
            _self_setattr_tracking = getattr(
                self, "_self_setattr_tracking", True
            )
            self._self_setattr_tracking = False

        self._trainable_variables = trainable_variables
        self._non_trainable_variables = non_trainable_variables
        self._layers = layers
        self._metrics = metrics
        self._seed_generators = seed_generators

        if backend.backend() == "tensorflow":
            # Reset attribute tracking (TF-specific)
            self._self_setattr_tracking = _self_setattr_tracking

    def _build_at_init(self):
        """Build the layer at `Layer.__init__`.

        We can only safely mark the layer as `built=True` in `Layer.__init__` if
        `build` is not overridden. Otherwise, it might cause the subclasses to
        ignore the user's `build`.
        """
        if utils.is_default(self.build):
            self.built = True
            self._post_build()
            self._lock_state()

    @property
    def path(self):
        """The path of the layer.

        If the layer has not been built yet, it will be `None`.
        """
        return self._path

    @property
    def input_spec(self):
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value):
        self._input_spec = value

    @utils.default
    def build(self, input_shape):
        self._check_super_called()
        if utils.is_default(self.build) and might_have_unbuilt_state(self):
            warnings.warn(
                f"`build()` was called on layer '{self.name}', however "
                "the layer does not have a `build()` method implemented "
                "and it looks like it has unbuilt state. This will cause "
                "the layer to be marked as built, despite not being "
                "actually built, which may cause failures down the line. "
                "Make sure to implement a proper `build()` method."
            )
        self.built = True

    def _lock_state(self):
        """Prevent further state updates, called automatically in `build()`."""
        if not self._tracker.locked:
            self._tracker.lock(
                msg=(
                    "You cannot add new elements of state "
                    "(variables or sub-layers) "
                    "to a layer that is already built. All state "
                    "must be created in the `__init__()` method or "
                    "in the `build()` method."
                )
            )

    def get_build_config(self):
        """Returns a dictionary with the layer's input shape.

        This method returns a config dict that can be used by
        `build_from_config(config)` to create all states (e.g. Variables and
        Lookup tables) needed by the layer.

        By default, the config only contains the input shape that the layer
        was built with. If you're writing a custom layer that creates state in
        an unusual way, you should override this method to make sure this state
        is already created when Keras attempts to load its value upon model
        loading.

        Returns:
            A dict containing the input shape associated with the layer.
        """
        if self._build_shapes_dict is not None:
            if len(self._build_shapes_dict) == 1:
                return {
                    "input_shape": tuple(self._build_shapes_dict.values())[0],
                }
            else:
                return {"shapes_dict": self._build_shapes_dict}

    def build_from_config(self, config):
        """Builds the layer's states with the supplied config dict.

        By default, this method calls the `build(config["input_shape"])` method,
        which creates weights based on the layer's input shape in the supplied
        config. If your config contains other information needed to load the
        layer's state, you should override this method.

        Args:
            config: Dict containing the input shape associated with this layer.
        """
        if config:
            if "input_shape" in config:
                self.build(config["input_shape"])
            elif "shapes_dict" in config:
                self.build(**config["shapes_dict"])

    def _obj_type(self):
        return "Layer"

    def add_variable(
        self,
        shape,
        initializer,
        dtype=None,
        trainable=True,
        autocast=True,
        regularizer=None,
        constraint=None,
        layout=None,
        name=None,
    ):
        """Add a weight variable to the layer.

        Alias of `add_weight()`.
        """
        return self.add_weight(
            shape=shape,
            initializer=initializer,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            regularizer=regularizer,
            constraint=constraint,
            layout=layout,
            name=name,
        )

    def add_weight(
        self,
        *args,
        shape=None,
        initializer=None,
        dtype=None,
        trainable=True,
        autocast=True,
        regularizer=None,
        constraint=None,
        aggregation="none",
        overwrite_with_gradient=False,
        layout=None,
        name=None,
    ):
        """Add a weight variable to the layer.

        Args:
            shape: Shape tuple for the variable. Must be fully-defined
                (no `None` entries). Defaults to `()` (scalar) if unspecified.
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"glorot_uniform"` for floating-point variables and to `"zeros"`
                for all other types (e.g. int, bool).
            dtype: Dtype of the variable to create, e.g. `"float32"`. If
                unspecified, defaults to the layer's variable dtype
                (which itself defaults to `"float32"` if unspecified).
            trainable: Boolean, whether the variable should be trainable via
                backprop or whether its updates are managed manually. Defaults
                to `True`.
            autocast: Boolean, whether to autocast layers variables when
                accessing them. Defaults to `True`.
            regularizer: Regularizer object to call to apply penalty on the
                weight. These penalties are summed into the loss function
                during optimization. Defaults to `None`.
            constraint: Contrainst object to call on the variable after any
                optimizer update, or string name of a built-in constraint.
                Defaults to `None`.
            aggregation: Optional string, one of `None`, `"none"`, `"mean"`,
                `"sum"` or `"only_first_replica"`. Annotates the variable with
                the type of multi-replica aggregation to be used for this
                variable when writing custom data parallel training loops.
                Defaults to `"none"`.
            overwrite_with_gradient: Boolean, whether to overwrite the variable
                with the computed gradient. This is useful for float8 training.
                Defaults to `False`.
            layout: Optional. Sharding layout for the variable, used instead of
                the one the active Distribution would assign. Either a
                `keras.distribution.TensorLayout` with its `device_mesh` set, or
                a backend-specific layout.
            name: String name of the variable. Useful for debugging purposes.
        """
        self._check_super_called()
        if args:
            # `args` is only kept to detect the legacy Keras 2 call style
            # (`add_weight(shape, initializer, dtype, ...)`) and raise a clear
            # error for positional `name`.
            if len(args) > 3:
                raise TypeError(
                    "add_weight() takes at most 3 positional arguments "
                    f"but {len(args)} were given."
                )
            shape_arg = args[0]
            if isinstance(shape_arg, str):
                raise ValueError(
                    "`name` must be passed as a keyword argument. "
                    f"Received: add_weight('{shape_arg}', ...). "
                    f"Use: add_weight(shape=..., name='{shape_arg}')."
                )
            if shape is not None:
                raise ValueError(
                    "`shape` was passed both positionally and as "
                    "a keyword argument."
                )
            shape = shape_arg
            if len(args) > 1:
                if initializer is not None:
                    raise ValueError(
                        "`initializer` was passed both positionally and "
                        "as a keyword argument."
                    )
                initializer = args[1]
            if len(args) > 2:
                if dtype is not None:
                    raise ValueError(
                        "`dtype` was passed both positionally and as a "
                        "keyword argument."
                    )
                dtype = args[2]
        if shape is None:
            shape = ()
        if dtype is not None:
            dtype = standardize_dtype(dtype)
        else:
            dtype = self.variable_dtype
        if initializer is None:
            if "float" in dtype:
                initializer = "glorot_uniform"
            else:
                initializer = "zeros"
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                layout=layout,
                name=name,
            )
        # Will be added to layer.losses
        variable.regularizer = regularizers.get(regularizer)
        variable.constraint = constraints.get(constraint)
        variable.overwrite_with_gradient = overwrite_with_gradient
        self._track_variable(variable)
        return variable

    @property
    def trainable(self):
        """Settable boolean, whether this layer should be trainable or not."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for the layer and its sublayers.

        When this value is changed during training (e.g. with a
        `Callback`) you need to call the parent
        `Model.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
            value: Boolean with the desired state for the layer's trainable
                attribute.
        """
        value = bool(value)
        self._trainable = value
        for v in self._trainable_variables:
            v.trainable = value
        for layer in self._layers:
            layer.trainable = value

    @property
    def variables(self):
        """List of all layer state, including random seeds.

        This extends `layer.weights` to include all state used by the layer
        including `SeedGenerator`s.

        Note that metrics variables are not included here, use
        `metrics_variables` to visit all the metric variables.
        """
        # Return all `Variables` associate with the layer including metrics
        # and random seeds. Also deduplicate them.
        variables = []
        seen_ids = set()
        for v in self._trainable_variables + self._non_trainable_variables:
            if id(v) not in seen_ids:
                variables.append(v)
                seen_ids.add(id(v))
        for sg in self._seed_generators:
            variables.append(sg.state)
        for layer in self._layers:
            for v in layer.variables:
                if id(v) not in seen_ids:
                    variables.append(v)
                    seen_ids.add(id(v))
        return variables

    @property
    def trainable_variables(self):
        """List of all trainable layer state.

        This is equivalent to `layer.trainable_weights`.
        """
        if not self.trainable:
            return []
        return [v for v in self.variables if v.trainable]

    @property
    def non_trainable_variables(self):
        """List of all non-trainable layer state.

        This extends `layer.non_trainable_weights` to include all state used by
        the layer including state for metrics and `SeedGenerator`s.
        """
        if not self.trainable:
            return self.variables
        return [v for v in self.variables if not v.trainable]

    @property
    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        # Return only `Variables` directly owned by layers and sub-layers.
        # Also deduplicate them.
        weights = []
        seen_ids = set()
        for w in self._trainable_variables + self._non_trainable_variables:
            if id(w) not in seen_ids:
                weights.append(w)
                seen_ids.add(id(w))
        for layer in self._layers:
            for w in layer.weights:
                if id(w) not in seen_ids:
                    weights.append(w)
                    seen_ids.add(id(w))
        return weights

    @property
    def trainable_weights(self):
        """List of all trainable weight variables of the layer.

        These are the weights that get updated by the optimizer during training.
        """
        if not self.trainable:
            return []
        return [v for v in self.weights if v.trainable]

    @property
    def non_trainable_weights(self):
        """List of all non-trainable weight variables of the layer.

        These are the weights that should not be updated by the optimizer during
        training. Unlike, `layer.non_trainable_variables` this excludes metric
        state and random seeds.
        """
        if not self.trainable:
            return self.weights
        return [v for v in self.weights if not v.trainable]

    @property
    def metrics(self):
        """List of all metrics."""
        metrics = list(self._metrics)
        for layer in self._layers:
            metrics.extend(layer.metrics)
        return metrics

    @property
    def metrics_variables(self):
        """List of all metric variables."""
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def get_weights(self):
        """Return the values of `layer.weights` as a list of NumPy arrays."""
        return [v.numpy() for v in self.weights]

    def set_weights(self, weights):
        """Sets the values of `layer.weights` from a list of NumPy arrays."""
        layer_weights = self.weights
        if len(layer_weights) != len(weights):
            raise ValueError(
                f"You called `set_weights(weights)` on layer '{self.name}' "
                f"with a weight list of length {len(weights)}, but the layer "
                f"was expecting {len(layer_weights)} weights."
            )
        for variable, value in zip(layer_weights, weights):
            if variable.shape != value.shape:
                raise ValueError(
                    f"Layer {self.name} weight shape {variable.shape} "
                    "is not compatible with provided weight "
                    f"shape {value.shape}."
                )
            variable.assign(value)

    @property
    def dtype_policy(self):
        return self._dtype_policy

    @dtype_policy.setter
    def dtype_policy(self, value):
        policy = dtype_policies.get(value)
        if isinstance(self._dtype_policy, DTypePolicyMap) and self.path:
            if self.path in self._dtype_policy:
                del self._dtype_policy[self.path]
            self._dtype_policy[self.path] = policy
        else:
            self._dtype_policy = policy
        policy = self._own_dtype_policy
        if policy.quantization_mode is not None:
            if self.built and not self._is_quantized:
                # Forward the policy's full parameters into `quantize` so the
                # built variables agree with the policy name: assigning
                # "int4/32_from_float32" quantizes with block_size=32 instead
                # of silently falling back to the default block size while
                # keeping a name that says 32. A mode may refuse
                # policy-triggered quantization by raising here (GPTQ needs
                # a calibration dataset that a bare policy cannot carry).
                strategy = strategy_registry.get_strategy(
                    policy.quantization_mode
                )
                config = (
                    strategy.config_from_policy(policy)
                    if strategy is not None
                    else None
                )
                self.quantize(policy.quantization_mode, config=config)

    @property
    def _own_dtype_policy(self):
        """The dtype policy that applies to this layer.

        A layer that holds a `DTypePolicyMap` reads the map's entry for its
        `path`; `dtype_policy` returns the map itself. A map answers
        `quantization_mode` and `name` for its default policy, so every
        reader of the layer's own policy reads this property.
        """
        policy = self._dtype_policy
        if isinstance(policy, DTypePolicyMap) and self.path:
            policy = policy[self.path]
        return policy

    @property
    def dtype(self):
        """Alias of `layer.variable_dtype`."""
        return self.variable_dtype

    @property
    def compute_dtype(self):
        """The dtype of the computations performed by the layer."""
        return self._own_dtype_policy.compute_dtype

    @property
    def variable_dtype(self):
        """The dtype of the state (weights) of the layer."""
        return self._own_dtype_policy.variable_dtype

    @property
    def quantization_mode(self):
        """The quantization mode of this layer, `None` if not quantized."""
        return self._own_dtype_policy.quantization_mode

    @property
    def variable_serialization_spec(self):
        """Maps each supported quantization mode to its variable names.

        A quantizable layer returns a dict from mode name (`None` for the
        float layer) to the ordered names its `save_own_variables` and
        `load_own_variables` serialize in that mode. The base implementation
        returns `None`: the layer has no quantization support.

        Returns:
            `None`, or a dict such as

            ```python
            {
                None: ["kernel", "bias"],
                "int8": ["kernel", "bias", "kernel_scale"],
            }
            ```
        """
        return None

    @property
    def input_dtype(self):
        """The dtype layer inputs should be converted to."""
        return self.compute_dtype

    @property
    def supports_masking(self):
        """Whether this layer supports computing a mask using `compute_mask`."""
        return self._supports_masking

    @supports_masking.setter
    def supports_masking(self, value):
        self._supports_masking = value

    @utils.default
    def compute_mask(self, inputs, previous_mask):
        return previous_mask

    def symbolic_call(self, *args, **kwargs):
        # Node is created at the end of `__call__` instead of `symbolic_call`.
        return self.compute_output_spec(*args, **kwargs)

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        self._check_super_called()
        if not self._called:
            self._called = True

        original_args = args
        original_kwargs = kwargs

        #############################################################
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_convert(x):
            # Prevent _keras_mask from disappearing
            mask = get_keras_mask(x)
            y = self.dtype_policy.convert_input(
                x, self.autocast, self.input_dtype
            )
            if mask is not None:
                set_keras_mask(y, mask)
            return y

        # `kwargs` is usually a dict with just the `training` argument, which
        # doesn't need conversion.
        kwargs_may_need_convert = bool(kwargs) and not (
            len(kwargs) == 1 and "training" in kwargs
        )

        # Used to avoid expensive `tree` operations in the most common case.
        if (
            kwargs_may_need_convert
            or len(args) != 1
            or not is_backend_tensor_or_symbolic(args[0], allow_none=False)
            or standardize_dtype(args[0].dtype) != self.input_dtype
        ) and self._convert_input_args:
            args = tree.map_structure(maybe_convert, args)
            kwargs = tree.map_structure(maybe_convert, kwargs)
        else:
            # `kwargs` get mutated later, so create a copy.
            kwargs = dict(kwargs)

        ##########################################################
        # 2. Enforce that only tensors can be passed positionally.
        if not self._allow_non_tensor_positional_args:
            for arg in tree.flatten(args):
                if not is_backend_tensor_or_symbolic(arg, allow_none=True):
                    raise ValueError(
                        "Only input tensors may be passed as "
                        "positional arguments. The following argument value "
                        f"should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `call()` signature, args, kwargs.
        call_spec = CallSpec(
            self._call_signature, self._call_context_args, args, kwargs
        )

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(call_spec.first_arg)

        ################
        # 4. Call build
        self._maybe_build(call_spec)

        ##########################
        # 5. Infer training value
        # Training phase for `Layer.call` is set via (in order of priority):
        # (1) The `training` argument passed to this `Layer.call`, if not None
        # (2) The training argument of an outer `Layer.call`.
        # (4) Any non-None default value for `training` in the call signature
        # (5) False (treating the layer as if it's in inference)

        # Maintains info about the `Layer.call` stack
        # across nested calls.
        call_context = self._get_call_context()

        for context_arg in self._call_context_args:
            self._resolve_and_populate_arg(
                context_arg, call_spec, call_context, kwargs
            )

        ##############################
        # 6. Populate mask argument(s)
        if len(call_spec.tensor_arguments_dict) == 1:
            if (
                "mask" in call_spec.argument_names
                and call_spec.arguments_dict["mask"] is None
            ):
                arg_name = list(call_spec.tensor_arguments_dict.keys())[0]
                only_tensor_arg = call_spec.tensor_arguments_dict[arg_name]
                mask = tree.map_structure(
                    get_keras_mask,
                    only_tensor_arg,
                )
                kwargs["mask"] = mask
        elif len(call_spec.tensor_arguments_dict) > 1:
            for k, v in call_spec.tensor_arguments_dict.items():
                expected_mask_arg_name = f"{k}_mask"
                if expected_mask_arg_name in call_spec.argument_names:
                    if call_spec.arguments_dict[expected_mask_arg_name] is None:
                        mask = tree.map_structure(get_keras_mask, v)
                        kwargs[expected_mask_arg_name] = mask

        # We need to cache the `previous_mask` before `__call__` because the
        # mask might be removed during the call, such as `MultiHeadAttention`.
        if "mask" in kwargs and kwargs["mask"] is not None:
            # Case 1: Mask was explicitly passed or auto-populated in step 6.
            previous_mask = kwargs["mask"]
        else:
            # Case 2: Fallback to the mask attached to the first input tensor.
            previous_mask = tree.map_structure(
                get_keras_mask, call_spec.first_arg
            )

        ####################
        # 7. Call the layer.
        try:
            with self._open_name_scope():
                current_scope = get_autocast_scope()
                new_scope = None
                if current_scope is not None:
                    # Clear or update the current scope if necessary.
                    if not self.autocast:
                        new_scope = AutocastScope(None)
                    elif not is_float_dtype(self.compute_dtype):
                        # Some preprocessing layers might have a non-float
                        # dtype, we should not autocast in this case.
                        new_scope = AutocastScope(None)
                    elif current_scope.dtype != self.compute_dtype:
                        new_scope = AutocastScope(self.compute_dtype)
                elif self.compute_dtype != self.variable_dtype:
                    # Enter a new scope if our dtypes are "mixed".
                    new_scope = AutocastScope(self.compute_dtype)
                if new_scope is not None:
                    with new_scope:
                        outputs = super().__call__(*args, **kwargs)
                else:
                    outputs = super().__call__(*args, **kwargs)
                # Change the layout for the layer output if needed.
                # This is useful for relayout intermediate tensor in the model
                # to achieve the optimal performance.
                distribution = distribution_lib.distribution()
                if distribution is not None:
                    current_layer_path = current_path()
                    current_layer_path += "/output"
                    layout = distribution.get_tensor_layout(current_layer_path)
                    if layout:
                        outputs = distribution_lib.distribute_tensor(
                            outputs, layout
                        )

                if not self.built:
                    self.built = True
                # Record activity regularizer loss.
                if self.activity_regularizer is not None:
                    for output in tree.flatten(outputs):
                        if backend.ops.is_tensor(output):
                            loss = self.activity_regularizer(output)
                            if output.ndim > 0:
                                # Normalize by batch size to ensure consistent
                                # regularization strength across batch sizes
                                batch_size = ops.cast(
                                    ops.shape(output)[0], dtype=loss.dtype
                                )
                                loss = ops.divide_no_nan(loss, batch_size)
                            self.add_loss(loss)

            # Set `previous_mask` on outputs if available. It is provided only
            # for the first positional input arg and its mask.
            # TODO: consider extending this to all args and kwargs.
            if self.supports_masking:
                self._set_mask_metadata(
                    call_spec.first_arg, outputs, previous_mask
                )
            elif any(m is not None for m in tree.flatten(previous_mask)):
                warnings.warn(
                    f"Layer '{self.name}' (of type {self.__class__.__name__}) "
                    "was passed an input with a mask attached to it. "
                    "However, this layer does not support masking and will "
                    "therefore destroy the mask information. Downstream "
                    "layers will not see the mask."
                )
        finally:
            # Destroy call context if we created it
            self._maybe_reset_call_context()

        ################################################
        # 8. Add a node in the graph for symbolic calls.
        if any_symbolic_tensors(original_args, original_kwargs):
            Node(
                operation=self,
                call_args=original_args,
                call_kwargs=original_kwargs,
                outputs=outputs,
            )

        return outputs

    def call(self, *args, **kwargs):
        raise self._not_implemented_error(self.call)

    def _resolve_and_populate_arg(
        self, arg_name, call_spec, call_context, kwargs
    ):
        # 1) user explicitly passed it?
        if arg_name in call_spec.user_arguments_dict:
            value = call_spec.user_arguments_dict[arg_name]
            call_context.set_value(arg_name, value)
        # 2) else: inherited from outer layer call?
        elif call_context.get_value(arg_name) is not None:
            value = call_context.get_value(arg_name)
        # 3) else: default from the call() signature. This stays local: the
        # call context is shared across the whole call tree, so propagating
        # it would let a non-None default (e.g. a preprocessing layer's
        # `training=True`) leak to sibling and downstream layers. The bound
        # `call()` still applies its own default, so there is nothing
        # further to do here.
        else:
            return

        if (
            self._call_has_context_arg.get(arg_name, False)
            and value is not None
        ):
            kwargs[arg_name] = value

    @traceback_utils.filter_traceback
    def stateless_call(
        self,
        trainable_variables,
        non_trainable_variables,
        *args,
        return_losses=False,
        **kwargs,
    ):
        """Call the layer without any side effects.

        Args:
            trainable_variables: List of trainable variables of the model.
            non_trainable_variables: List of non-trainable variables of the
                model.
            *args: Positional arguments to be passed to `call()`.
            return_losses: If `True`, `stateless_call()` will return the list of
                losses created during `call()` as part of its return values.
            **kwargs: Keyword arguments to be passed to `call()`.

        Returns:
            A tuple. By default, returns `(outputs, non_trainable_variables)`.
                If `return_losses = True`, then returns
                `(outputs, non_trainable_variables, losses)`.

        Note: `non_trainable_variables` include not only non-trainable weights
        such as `BatchNormalization` statistics, but also RNG seed state
        (if there are any random operations part of the layer, such as dropout),
        and `Metric` state (if there are any metrics attached to the layer).
        These are all elements of state of the layer.

        Example:

        ```python
        model = ...
        data = ...
        trainable_variables = model.trainable_variables
        non_trainable_variables = model.non_trainable_variables
        # Call the model with zero side effects
        outputs, non_trainable_variables = model.stateless_call(
            trainable_variables,
            non_trainable_variables,
            data,
        )
        # Attach the updated state to the model
        # (until you do this, the model is still in its pre-call state).
        for ref_var, value in zip(
            model.non_trainable_variables, non_trainable_variables
        ):
            ref_var.assign(value)
        ```
        """
        self._check_super_called()
        if not self.built:
            raise ValueError(
                f"To call stateless_call, {self.__class__.__name__} must be "
                "built (i.e. its variables must have been already created). "
                "You can build it by calling it on some data."
            )
        if len(trainable_variables) != len(self.trainable_variables):
            raise ValueError(
                "Argument `trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().trainable_variables. "
                f"Received list with length {len(trainable_variables)}, "
                f"but expected {len(self.trainable_variables)} variables."
            )
        if len(non_trainable_variables) != len(self.non_trainable_variables):
            raise ValueError(
                "Argument `non_trainable_variables` must be a list of tensors "
                "corresponding 1:1 to "
                f"{self.__class__.__name__}().non_trainable_variables. "
                f"Received list with length {len(non_trainable_variables)}, "
                f"but expected {len(self.non_trainable_variables)} variables."
            )

        # Gather variable mapping
        trainable_mapping = zip(self.trainable_variables, trainable_variables)
        non_trainable_mapping = zip(
            self.non_trainable_variables, non_trainable_variables
        )
        mapping = list(trainable_mapping) + list(non_trainable_mapping)

        # Caches info about `call()` signature, args, kwargs.
        call_spec = CallSpec(
            self._call_signature, self._call_context_args, args, kwargs
        )

        # Maintains info about the `Layer.call` stack across nested calls.
        call_context = self._get_call_context()

        for context_arg in self._call_context_args:
            self._resolve_and_populate_arg(
                context_arg, call_spec, call_context, kwargs
            )

        try:
            # Call in stateless scope
            losses = None
            with backend.StatelessScope(
                state_mapping=mapping, collect_losses=return_losses
            ) as scope:
                outputs = self._dispatch_call(*args, **kwargs)
                if return_losses:
                    losses = self.losses
        finally:
            # Destroy call context if we created it
            self._maybe_reset_call_context()

        # Gather updated non-trainable variables
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            non_trainable_variables.append(new_v)

        if return_losses:
            return outputs, non_trainable_variables, losses
        return outputs, non_trainable_variables

    def compute_output_spec(self, *args, **kwargs):
        if utils.is_default(self.compute_output_shape):
            return super().compute_output_spec(*args, **kwargs)
        else:
            # Use compute_output_shape() to return the right output spec
            call_spec = CallSpec(
                self._call_signature, self._call_context_args, args, kwargs
            )
            shapes_dict = get_shapes_dict(call_spec)
            shapes_dict = update_shapes_dict_for_target_fn(
                self.compute_output_shape,
                shapes_dict=shapes_dict,
                call_spec=call_spec,
                class_name=self.__class__.__name__,
            )
            output_shape = self.compute_output_shape(**shapes_dict)

            if (
                isinstance(output_shape, list)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                output_shape = tuple(output_shape)
            if not isinstance(output_shape, (list, tuple, dict)):
                try:
                    output_shape = tuple(output_shape)
                except:
                    raise ValueError(
                        "Method `compute_output_shape()` of layer "
                        f"{self.__class__.__name__} is returning "
                        "a type that cannot be interpreted as a shape. "
                        "It should return a shape tuple. "
                        f"Received: {output_shape}"
                    )
            if (
                isinstance(output_shape, tuple)
                and output_shape
                and isinstance(output_shape[0], (int, type(None)))
            ):
                return KerasTensor(output_shape, dtype=self.compute_dtype)
            # Case: nested. Could be a tuple/list of shapes, or a dict of
            # shapes. Could be deeply nested.
            return tree.map_shape_structure(
                lambda s: KerasTensor(s, dtype=self.compute_dtype), output_shape
            )

    @utils.default
    def compute_output_shape(self, *args, **kwargs):
        raise self._not_implemented_error(
            self.compute_output_shape,
            "Should implement `def compute_output_shape(self, input_shape)`.",
        )

    def add_loss(self, loss):
        """Can be called inside of the `call()` method to add a scalar loss.

        Example:

        ```python
        class MyLayer(Layer):
            ...
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x
        ```
        """
        # Eager only.
        losses = tree.flatten(loss)
        for x in losses:
            if not backend.ops.is_tensor(x):
                raise ValueError(
                    "`add_loss()` can only be called from inside `build()` or "
                    f"`call()`, on a tensor input. Received invalid value: {x}"
                )
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                for x in losses:
                    scope.add_loss(x)
                    self._loss_ids.add(id(x))
        else:
            self._losses.extend(losses)

    def _get_own_losses(self):
        if backend.in_stateless_scope():
            losses = []
            scope = backend.get_stateless_scope()
            for loss in scope.losses:
                if id(loss) in self._loss_ids:
                    losses.append(loss)
            return losses
        else:
            return self._losses[:]

    def _get_regularization_losses(self):
        weight_regularization_losses = []
        for variable in self.trainable_weights:
            if variable.regularizer is None:
                continue
            if backend.in_stateless_scope() and not in_symbolic_scope():
                # If in symbolic scope, we might get `None` from
                # `get_current_value` in `backend.compute_output_spec`. So we
                # assign `variable` instead.
                v = backend.get_stateless_scope().get_current_value(variable)
            else:
                v = variable
            weight_regularization_losses.append(variable.regularizer(v))
        return weight_regularization_losses

    @property
    def losses(self):
        """List of scalar losses from `add_loss`, regularizers and sublayers."""
        if self._losses_override:
            return self._losses_override
        losses = self._get_own_losses()
        for layer in self._flatten_layers(include_self=False):
            losses.extend(layer._get_own_losses())
        weight_regularization_losses = self._get_regularization_losses()
        losses.extend(weight_regularization_losses)
        return losses

    def _clear_losses(self):
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_losses:
                # Filter by identity (id) rather than using list.remove(),
                # which compares by value. Value comparison on JAX tracers
                # during JIT causes TracerBoolConversionError.
                scope.losses[:] = [
                    x for x in scope.losses if id(x) not in self._loss_ids
                ]
        self._losses.clear()
        self._loss_ids.clear()
        for layer in self._layers:
            layer._clear_losses()

    # Quantization-related methods.
    #
    # Mode-specific behavior (variables, forward pass, quantized values,
    # policy naming) is owned by the strategies in
    # `keras.src.quantizers.strategy_registry`; the methods below look the
    # strategy up and delegate. A layer participates by exposing its
    # quantizable structure through `_quantization_geometry()` and by
    # listing the modes it supports in its `variable_serialization_spec`.

    def _supports_quantization_mode(self, strategy):
        """Whether this layer declares support for `strategy`'s mode.

        A layer declares support by listing the mode name in its
        `variable_serialization_spec`.

        Args:
            strategy: The `QuantizationStrategy` registered for the mode.

        Returns:
            A boolean.
        """
        spec = self.variable_serialization_spec
        return spec is not None and strategy.name in spec

    def _strategy_owns_weight_storage(self):
        """Whether the quantization strategy creates the weight storage.

        A strategy that owns its weight storage (int8, int4) creates the
        quantized variables in `quantized_build`, so `build` must not add
        the floating-point weight. Float8 and the unquantized layer keep
        the floating-point weight.

        Returns:
            A boolean.
        """
        strategy = strategy_registry.get_strategy(self.quantization_mode)
        return strategy is not None and strategy.owns_weight_storage

    def _check_lora_supported(self, mode):
        """Raises if the quantization `mode` does not support LoRA."""
        strategy = strategy_registry.get_strategy(mode)
        if strategy is not None and not strategy.supports_lora:
            raise NotImplementedError(
                f"lora is not currently supported with {mode.upper()} "
                "quantization."
            )

    def _check_quantized_build_attributes(self, geometry):
        """Raises if `build()` did not set an attribute `geometry` reads."""
        missing = [a for a in geometry.build_attributes if not hasattr(self, a)]
        if missing:
            raise ValueError(
                f"Layer '{self.name}' (of type '{type(self).__name__}') "
                f"cannot be quantized: its `build()` did not set "
                f"{', '.join(missing)}. "
                "A subclass that overrides `build()` must call "
                "`super().build()`."
            )

    def quantized_build(self, input_shape, mode, config=None):
        strategy = strategy_registry.get_strategy(mode)
        if strategy is None or not self._supports_quantization_mode(strategy):
            if self.variable_serialization_spec is None:
                # The layer has no quantization support at all.
                raise self._not_implemented_error(self.quantized_build)
            raise self._quantization_mode_error(mode)
        if config is not None:
            # The config the variables are built from is the one the layer
            # reports afterwards, whichever path called this.
            self.quantization_config = config
        strategy.build(self, input_shape, config)
        self._is_quantized = True

    def quantize(self, mode=None, type_check=True, config=None):
        """Quantizes this layer in place.

        Resolves `mode`/`config` into a `QuantizationConfig`, validates that
        the layer supports the mode *before* mutating any state, then lets
        the mode's strategy compute and swap the variables and updates the
        dtype policy.

        A layer supports quantization by defining
        `_quantization_geometry()`; only instances of the exact class that
        defines it can be quantized (pass `type_check=False` to quantize an
        instance of a subclass that inherits the geometry).

        Args:
            mode: The quantization mode, e.g. `"int8"`. Optional if `config`
                is provided.
            type_check: Whether to reject subclasses of the class that owns
                the quantization support. Defaults to `True`.
            config: Optional `QuantizationConfig` carrying the mode and its
                parameters.
        """
        config = validate_and_resolve_config(mode, config)
        mode = config.mode
        self._check_quantize_args(mode, self.compute_dtype)
        geometry = self._quantization_geometry()
        if geometry is None or (
            type_check and type(self) is not self._quantization_type_owner()
        ):
            raise self._not_implemented_error(self.quantize)
        strategy = strategy_registry.get_strategy(mode)
        if strategy is None or not self._supports_quantization_mode(strategy):
            raise self._quantization_mode_error(mode)
        self._check_quantized_build_attributes(geometry)
        strategy.check_quantizable(self)
        if self.lora_enabled:
            self._check_lora_supported(mode)
        self._tracker.unlock()
        try:
            # Record the config only after the mode is validated, so a
            # rejected mode leaves the layer untouched.
            self.quantization_config = config
            strategy.quantize(self, config)
            self._finalize_quantization_policy(strategy, config)
        finally:
            self._tracker.lock()

    def _quantization_geometry(self):
        """Returns this layer's quantization geometry, or `None`.

        The geometry (`keras.src.quantizers.geometry`) describes the layer's
        quantizable structure; the strategies consume it to build
        variables, compute quantized values, and run quantized forward
        passes. The base implementation returns `None`, meaning the layer
        has no generic quantization support. Defining this method on a
        subclass also marks that subclass as the owner of its quantization
        support for `quantize()`'s type check.

        Returns:
            A `keras.src.quantizers.geometry.QuantizationGeometry` (for
            example a `ProjectionGeometry` for a 2D kernel), or `None`.
        """
        return None

    def _quantized_weight(self):
        """Returns the `QuantizedWeight` view of the layer's weight.

        `None` when the layer is not quantized, or when its mode holds no
        integer codes for it (see `QuantizationStrategy.quantized_weight`).
        """
        if not self.built:
            return None
        strategy = strategy_registry.get_strategy(self.quantization_mode)
        if strategy is None:
            return None
        return strategy.quantized_weight(self)

    def _get_weight_with_merged_lora(self, name):
        """Returns `(value, scale, zero_point)` to save for weight `name`.

        Without a `QuantizedWeight` view this is the float property `name`
        (which merges any LoRA update itself) with no scale or zero point.
        Otherwise it is the stored codes, scale and zero point, or, with
        LoRA enabled, the mode's `encode` of the dequantized weight plus
        the LoRA update `lora_{name}_a @ lora_{name}_b`.
        """
        quantized_weight = self._quantized_weight()
        if quantized_weight is None:
            return getattr(self, name), None, None
        if not self.lora_enabled:
            return (
                quantized_weight.codes,
                quantized_weight.scale,
                quantized_weight.zero_point,
            )
        lora_delta = (self.lora_alpha / self.lora_rank) * ops.matmul(
            getattr(self, f"lora_{name}_a"), getattr(self, f"lora_{name}_b")
        )
        merged = ops.add(
            quantized_weight.dequantize(self.variable_dtype), lora_delta
        )
        strategy = strategy_registry.get_strategy(self.quantization_mode)
        return strategy.encode(self, merged, self.quantization_config)

    def _serialized_attribute(self, entry, attribute):
        """Returns the attribute that holds an entry of the spec."""
        try:
            return getattr(self, attribute)
        except AttributeError as e:
            raise AttributeError(
                f"Layer '{self.name}' lists '{entry}' in its "
                "`variable_serialization_spec` for the quantization mode "
                f"{self.quantization_mode!r}, but reading `self.{attribute}` "
                f"failed: {e}. Every name in the spec must be an attribute "
                "of the layer. Set it to `None` when the configuration omits "
                "the variable, or leave the name out of the spec for a layer "
                "that does not hold it."
            ) from e

    def _save_serialized_variables(self, store, name):
        """Saves the variables `variable_serialization_spec` lists.

        `name` is the spec entry of the layer's weight (`"kernel"` or
        `"embeddings"`), held at `_<name>`. The weight and its scale and
        zero point come from `_get_weight_with_merged_lora`. Each variable
        is stored under its position in the spec ("0", "1", ...). Every
        entry of the spec is an attribute of the layer, and an entry whose
        variable is `None` for this configuration is skipped.
        """
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)
        strategy = strategy_registry.get_strategy(mode)
        if strategy is not None:
            strategy.check_saveable(self)
        value, scale, zero_point = self._get_weight_with_merged_lora(name)
        merged = {name: value}
        if scale is not None:
            merged[f"{name}_scale"] = scale
        if zero_point is not None:
            merged[f"{name}_zero"] = zero_point
        idx = 0
        for entry in self.variable_serialization_spec[mode]:
            if entry in merged:
                value = merged[entry]
            else:
                value = self._serialized_attribute(entry, entry)
            if value is None:
                continue
            store[str(idx)] = value
            idx += 1

    def _load_serialized_variables(
        self, store, name, check_variable_count=True
    ):
        """Loads the variables `_save_serialized_variables` saved."""
        if not self.lora_enabled and check_variable_count:
            self._check_load_own_variables(store)
        if not self.built:
            return
        mode = self.quantization_mode
        if mode not in self.variable_serialization_spec:
            raise self._quantization_mode_error(mode)
        idx = 0
        for entry in self.variable_serialization_spec[mode]:
            target = self._serialized_attribute(
                entry, f"_{name}" if entry == name else entry
            )
            if target is None:
                continue
            value = store[str(idx)]
            if entry == "g_idx":
                # `g_idx` is stored as `float32` (see build). Cast to the
                # variable dtype on assign so both legacy `float32`
                # checkpoints and any `int32`-saved ones load correctly.
                value = ops.cast(value, target.dtype)
            target.assign(value)
            idx += 1
        if self.lora_enabled:
            for factor in ("a", "b"):
                lora = getattr(self, f"lora_{name}_{factor}")
                lora.assign(ops.zeros(lora.shape))
        strategy = strategy_registry.get_strategy(mode)
        if strategy is not None:
            strategy.variables_loaded(self)

    def _quantization_type_owner(self):
        """The class whose `_quantization_geometry` definition applies."""
        for cls in type(self).__mro__:
            if "_quantization_geometry" in cls.__dict__:
                return cls

    def _finalize_quantization_policy(self, strategy, config):
        # Set new dtype policy only for modes that don't already have one.
        policy = self._own_dtype_policy
        if policy.quantization_mode is None:
            policy_name = strategy.policy_suffix(self, config)
            self.dtype_policy = dtype_policies.get(
                f"{policy_name}_from_{policy.name}"
            )

    def _check_quantize_args(self, mode, compute_dtype):
        if not self.built:
            raise ValueError(
                "Cannot quantize a layer that isn't yet built. "
                f"Layer '{self.name}' (of type '{self.__class__.__name__}') "
                "is not built yet."
            )
        if self._is_quantized:
            raise ValueError(
                f"Layer '{self.name}' is already quantized with "
                f"dtype_policy='{self._own_dtype_policy.name}'. "
                f"Received: mode={mode}"
            )
        if mode == "int8" and compute_dtype == "float16":
            raise ValueError(
                f"Quantization mode='{mode}' doesn't work well with "
                "compute_dtype='float16'. Consider loading model/layer with "
                "another dtype policy such as 'mixed_bfloat16' or "
                "'mixed_float16' before calling `quantize()`."
            )

    def quantized_call(self, *args, **kwargs):
        current_remat_mode = get_current_remat_mode()

        if (
            current_remat_mode != self._remat_mode
            and current_remat_mode is not None
        ):
            warnings.warn(
                f"The RematScope at call time ({current_remat_mode}) differs "
                f"the one set during layer initialization "
                f"({self._remat_mode}). "
                f"Restoring the correct rematerialization mode "
                f"{self._remat_mode} for this layer."
            )
        mode = self.quantization_mode
        strategy = strategy_registry.get_strategy(mode)
        if strategy is None:
            raise self._quantization_mode_error(mode)
        return strategy.call(self, *args, **kwargs)

    def _not_implemented_error(self, attr, msg=None):
        if callable(attr):
            attr_name = attr.__name__
            attr_type = "method"
        else:
            attr_name = str(attr)
            attr_type = "attribute"
        msg = f" {msg}" if msg is not None else ""
        return NotImplementedError(
            f"Layer {self.__class__.__name__} does not have a `{attr_name}` "
            f"{attr_type} implemented.{msg}"
        )

    def _quantization_mode_error(self, mode):
        return NotImplementedError(
            "Invalid quantization mode. Expected one of "
            f"{strategy_registry.registered_modes()}. "
            f"Received: quantization_mode={mode}"
        )

    def save_own_variables(self, store):
        """Saves the state of the layer.

        You can override this method to take full control of how the state of
        the layer is saved upon calling `model.save()`.

        Args:
            store: Dict where the state of the model will be saved.
        """
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            store[f"{i}"] = v

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )

    def load_own_variables(self, store):
        """Loads the state of the layer.

        You can override this method to take full control of how the state of
        the layer is loaded upon calling `keras.models.load_model()`.

        Args:
            store: Dict from which the state of the model will be loaded.
        """
        self._check_load_own_variables(store)
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            v.assign(store[f"{i}"])

    def _track_variable(self, variable):
        if variable.trainable:
            self._tracker.add_to_store("trainable_variables", variable)
        else:
            self._tracker.add_to_store("non_trainable_variables", variable)
        if not self.trainable:
            variable.trainable = False
        self._post_track_variable(variable)

    def _untrack_variable(self, variable):
        previous_lock_state = self._tracker.locked
        self._tracker.unlock()
        self._tracker.untrack(variable)
        if previous_lock_state is True:
            self._tracker.lock()
        self._post_untrack_variable(variable)

    def add_metric(self, *args, **kwargs):
        # Permanently disabled
        raise NotImplementedError(
            "Layer `add_metric()` method is deprecated. "
            "Add your metric in `Model.compile(metrics=[...])`, "
            "or create metric trackers in init() or build() "
            "when subclassing the layer or model, then call "
            "`metric.update_state()` whenever necessary."
        )

    def count_params(self):
        """Count the total number of scalars composing the weights.

        Returns:
            An integer count.
        """
        if not self.built:
            raise ValueError(
                "You tried to call `count_params` "
                f"on layer '{self.name}', "
                "but the layer isn't built. "
                "You can build it manually via: "
                f"`layer.build(input_shape)`."
            )
        return summary_utils.count_params(self.weights)

    def _maybe_build(self, call_spec):
        if self.built:
            return

        with self._open_name_scope():
            self._build_from_call_spec(call_spec)

    def _build_from_call_spec(self, call_spec):
        shapes_dict = get_shapes_dict(call_spec)
        first_shape = next(iter(shapes_dict.values()), None)

        # If the layer has a build method, call it with our input shapes.
        if not utils.is_default(self.build):
            shapes_dict = update_shapes_dict_for_target_fn(
                self.build,
                shapes_dict=shapes_dict,
                call_spec=call_spec,
                class_name=self.__class__.__name__,
            )
            self.build(**shapes_dict)
            # Check input spec again (after build, since self.input_spec
            # may have been updated
            self._assert_input_compatibility(call_spec.first_arg)
            return

        # Otherwise, attempt to build the layer by calling it on symbolic input.
        if might_have_unbuilt_state(self):
            try:
                backend.compute_output_spec(
                    self.call, **call_spec.arguments_dict
                )
            except Exception as e:
                if call_spec.eager:
                    # Will let the actual eager call do state-building
                    return
                warnings.warn(
                    f"Layer '{self.name}' looks like it has unbuilt state, but "
                    "Keras is not able to trace the layer `call()` in order to "
                    "build it automatically. Possible causes:\n"
                    "1. The `call()` method of your layer may be crashing. Try "
                    "to `__call__()` the layer eagerly on some test input "
                    "first to see if it works. "
                    "E.g. `x = np.random.random((3, 4)); y = layer(x)`\n"
                    "2. If the `call()` method is correct, then you may need "
                    "to implement the `def build(self, input_shape)` method on "
                    "your layer. It should create all variables used by the "
                    "layer (e.g. by calling `layer.build()` on all its "
                    "children layers).\n"
                    f"Exception encountered: ''{e}''"
                )
        self.build(first_shape)

    def _build_by_run_for_single_pos_arg(self, input_shape):
        # Case: all inputs are in the first arg (possibly nested).
        input_tensors = tree.map_shape_structure(
            lambda s: backend.KerasTensor(s), input_shape
        )
        try:
            # Because `build` is not defined, we trace `call` via
            # `compute_output_spec` to build the nested layers. This is
            # reached from `build_from_config`, not from `build` or
            # `__call__`, so the name scope is not open yet; open it here so
            # nested layers capture the same `path` as a from-code build.
            with backend.name_scope(self.name):
                backend.compute_output_spec(self.call, input_tensors)
            # Mark this layer as built and capture the `path` of this layer.
            # We do this after so that it doesn't happen in case of failure of
            # `compute_output_spec`.
            self.build(input_shape)
            return True
        except:
            return False

    def _build_by_run_for_kwargs(self, shapes_dict):
        # Case: inputs were recorded as multiple keyword arguments.
        if not all(is_shape_tuple(s) for s in shapes_dict.values()):
            # Not supported: nested input keyword arguments.
            return False

        # Case: all input keyword arguments were plain tensors.
        input_tensors = {
            # We strip the `_shape` suffix to recover kwarg names.
            utils.removesuffix(k, "_shape"): backend.KerasTensor(shape)
            for k, shape in shapes_dict.items()
        }
        try:
            # Because `build` is not defined, we trace `call` via
            # `compute_output_spec` to build the nested layers. This is
            # reached from `build_from_config`, not from `build` or
            # `__call__`, so the name scope is not open yet; open it here so
            # nested layers capture the same `path` as a from-code build.
            with backend.name_scope(self.name):
                backend.compute_output_spec(self.call, **input_tensors)
            # Mark this layer as built and capture the `path` of this layer.
            # We do this after so that it doesn't happen in case of failure of
            # `compute_output_spec`.
            self.build(shapes_dict)
            return True
        except:
            return False

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} name={self.name}, built={self.built}>"
        )

    def __str__(self):
        return self.__repr__()

    def __setattr__(self, name, value):
        # Track Variables, Layers, Metrics, SeedGenerators.
        name, value = self._setattr_hook(name, value)
        if name != "_tracker":
            if not hasattr(self, "_tracker"):
                self._initialize_tracker()
            value = self._tracker.track(value)

        # NNX-specific bypass for Keras-internal bookkeeping attributes:
        # some are set during a traced call (e.g. inside the jitted train
        # step), which nnx.Module.__setattr__ forbids, and
        # `_compiled_trainable_state` is a dict keyed by layer objects,
        # which flax's attribute scanning cannot process.
        if (
            backend.backend() == "jax"
            and is_nnx_enabled()
            and name
            in (
                "_called",
                "built",
                "_losses_override",
                "_compiled_trainable_state",
            )
        ):
            object.__setattr__(self, name, value)
            return

        super().__setattr__(name, value)

    def __delattr__(self, name):
        obj = getattr(self, name)
        if isinstance(obj, backend.Variable):
            import gc

            # It will take a short amount of time for the corresponding buffer
            # to be actually removed from the device.
            # https://stackoverflow.com/a/74631949
            self._untrack_variable(obj)
            super().__delattr__(name)
            gc.collect()
        else:
            super().__delattr__(name)

    def _check_super_called(self):
        if getattr(self, "_lock", True):
            raise RuntimeError(
                f"In layer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. Go add it!"
            )

    def _assert_input_compatibility(self, arg_0):
        if self.input_spec:
            try:
                input_spec.assert_input_compatibility(
                    self.input_spec, arg_0, layer_name=self.name
                )
            except SystemError:
                if backend.backend() == "torch":
                    # TODO: The torch backend failed the ONNX CI with the error:
                    # SystemError: <method '__int__' of 'torch._C.TensorBase'
                    # objects> returned a result with an exception set
                    # As a workaround, we are skipping this for now.
                    pass
                else:
                    raise

    def _get_call_context(self):
        """Returns currently active `CallContext`."""
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None:
            # Enter new call context.
            layer_call_ctx = CallContext(entry_layer=self)
            global_state.set_global_attribute(
                "current_call_ctx", layer_call_ctx
            )
            self._clear_losses()
        return layer_call_ctx

    def _maybe_reset_call_context(self):
        layer_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if layer_call_ctx is None or layer_call_ctx.entry_layer == self:
            global_state.set_global_attribute("current_call_ctx", None)

    def _flatten_layers(self, include_self=True, recursive=True):
        layers = []
        if include_self:
            layers.append(self)
        seen_object_ids = set()
        deque = collections.deque(self._layers)
        while deque:
            layer = deque.popleft()
            if id(layer) in seen_object_ids:
                continue
            seen_object_ids.add(id(layer))
            layers.append(layer)
            # Introspect recursively through sublayers.
            if recursive:
                deque.extendleft(layer._layers)
        return layers

    def _set_mask_metadata(self, inputs, outputs, previous_mask):
        flat_outputs = tree.flatten(outputs)

        mask_already_computed = all(
            get_keras_mask(x) is not None for x in flat_outputs
        )
        if mask_already_computed:
            return

        output_masks = self.compute_mask(inputs, previous_mask)
        if output_masks is None:
            return

        flat_masks = tree.flatten(output_masks)
        for tensor, mask in zip(flat_outputs, flat_masks):
            if get_keras_mask(tensor) is None and mask is not None:
                set_keras_mask(tensor, mask)

    @python_utils.default
    def get_config(self):
        self._check_super_called()
        base_config = super().get_config()
        config = {
            "trainable": self.trainable,
            "dtype": dtype_policies.serialize(self.dtype_policy),
        }
        if self.activity_regularizer is not None:
            config["activity_regularizer"] = regularizers.serialize(
                self.activity_regularizer
            )
        return {**base_config, **config}

    def _open_name_scope(self):
        from keras.src.utils import jax_utils  # avoid circular imports

        if self._parent_path is None:
            # Avoid mutating _parent_path during a JAX trace if it's part of
            # nnx.Object state and the object was created at a different trace
            # level. We check if we are in NNX mode and if we are in a JAX
            # trace.
            if not (is_nnx_enabled() and jax_utils.is_in_jax_tracing_scope()):
                self._parent_path = current_path()

        return backend.name_scope(self.name, caller=self)

    def rematerialized_call(self, layer_call, *args, **kwargs):
        """Enable rematerialization dynamically for layer's call method.

        Args:
            layer_call: The original `call` method of a layer.

        Returns:
            Rematerialized layer's `call` method.
        """
        if not self._remat_mode:
            return layer_call
        if isinstance(kwargs.get("training"), bool):
            # Keep the training mode static when rematerialization traces the
            # call arguments. Layers such as Dropout use Python control flow.
            training = kwargs["training"]
            original_call = layer_call

            @functools.wraps(original_call)
            def layer_call(*args, **kwargs):
                kwargs = {**kwargs, "training": training}
                return original_call(*args, **kwargs)

        if self._remat_mode.mode == "activations":
            has_activation = (
                hasattr(self, "activation") and self.activation is not None
            )
            if has_activation:

                @functools.wraps(layer_call)
                def rematerialized_activation_call_wrapper(*args, **kwargs):
                    original_activation = self.activation
                    self.activation = remat.remat(original_activation)
                    try:
                        return layer_call(*args, **kwargs)
                    finally:
                        self.activation = original_activation

                return rematerialized_activation_call_wrapper
        elif self._remat_mode.mode == "list_of_layers" and (
            self.name in self._remat_mode.layer_names
        ):
            return remat.remat(layer_call)
        return super().rematerialized_call(layer_call, *args, **kwargs)

    def _register_call_context_args(self, *names):
        """Registers call-context args for this layer.

        If this layer declares a `call()` method that accepts
        one or more of the given args, those args will be
        automatically injected into the call signature of this
        layer. This layer will also propagate the args to any
        nested sublayers that are called from within this layer.

        If this layer doesn't declare a `call()` method that
        accepts one or more of the given args, these args will
        simply be propagated to any nested sublayers without
        being injected into the call signature of this layer.
        This is useful for propagating custom arguments
        from top-level layers/models to sublayers.

        Example:
        ```
        class Inner(layers.Layer):

            def __init__(self):
                super().__init__()
                # Register `foo_mode` as a call-context arg
                self._register_call_context_args("foo_mode")

            def call(self, x, foo_mode=False):
                # If foo_mode=True add 1, otherwise add 0
                add_val = ops.where(foo_mode, 1.0, 0.0)
                return x + add_val

        class Outer(layers.Layer):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

            def call(self, x):
                # We don't explicitly pass foo_mode here—Base Layer.__call__
                # should inject it into `self.inner`
                return self.inner(x)

        sample_input = np.array([[1.0], [2.0]])

        # Sequential model
        seq = models.Sequential([Outer()])

        # Tell the Sequential model to propagate foo_mode down
        # the call-stack
        seq._register_call_context_args("foo_mode")

        # foo_mode=True -> input + 1
        out_true = seq(sample_input, foo_mode=True)
        """
        if self._called:
            raise RuntimeError(
                "Cannot add call-context args after the layer has been called."
            )
        self._call_context_args = self._call_context_args | set(names)

        self._call_has_context_arg.update(
            {arg: (arg in self.call_signature_parameters) for arg in names}
        )


def is_backend_tensor_or_symbolic(x, allow_none=False):
    if allow_none and x is None:
        return True
    return backend.ops.is_tensor(x) or isinstance(x, backend.KerasTensor)


class CallSpec:
    def __init__(self, signature, call_context_args, args, kwargs):
        # Strip out user-supplied call-context args that this layer’s `call()`
        # does not accept (otherwise `signature.bind` would raise).
        # This includes built-in args like `training`, and user-defined args.
        call_args = {
            context_arg: kwargs.pop(context_arg)
            for context_arg in call_context_args
            if context_arg in kwargs and context_arg not in signature.parameters
        }

        bound_args = signature.bind(*args, **kwargs)

        # Combine the two dicts.
        self.user_arguments_dict = {**call_args, **bound_args.arguments}

        bound_args.apply_defaults()
        arg_dict = {}
        arg_names = []
        tensor_arg_dict = {}
        tensor_args = []
        tensor_arg_names = []
        nested_tensor_arg_names = []
        for name, value in bound_args.arguments.items():
            arg_dict[name] = value
            arg_names.append(name)
            if is_backend_tensor_or_symbolic(value):
                tensor_args.append(value)
                tensor_arg_names.append(name)
                tensor_arg_dict[name] = value
            elif tree.is_nested(value) and len(value) > 0:
                flat_values = tree.flatten(value)
                if all(
                    is_backend_tensor_or_symbolic(x, allow_none=True)
                    for x in flat_values
                ):
                    tensor_args.append(value)
                    tensor_arg_names.append(name)
                    tensor_arg_dict[name] = value
                    nested_tensor_arg_names.append(name)
                elif any(is_backend_tensor_or_symbolic(x) for x in flat_values):
                    raise ValueError(
                        "In a nested call() argument, "
                        "you cannot mix tensors and non-tensors. "
                        "Received invalid mixed argument: "
                        f"{name}={value}"
                    )
        self.arguments_dict = arg_dict
        self.argument_names = arg_names
        self.tensor_arguments_dict = tensor_arg_dict
        self.tensor_arguments_names = tensor_arg_names
        self.nested_tensor_argument_names = nested_tensor_arg_names
        self.first_arg = arg_dict[arg_names[0]]
        if all(
            backend.ops.is_tensor(x)
            for x in self.tensor_arguments_dict.values()
        ):
            self.eager = True
        else:
            self.eager = False


def get_arguments_dict(fn, args, kwargs):
    """Return a dict mapping argument names to their values."""
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    arg_dict = {}
    for name, value in bound_args.arguments.items():
        arg_dict[name] = value
    return arg_dict


def get_shapes_dict(call_spec):
    """Convert the call() arguments dict into a dict of input shape arguments.

    Example:

    ```
    >>> get_shapes_dict(call_spec)
    {"input_a_shape": (2, 3)}
    ```
    """

    def standardize_shape_or_none(x):
        return None if x is None else standardize_shape(x.shape)

    shapes_dict = {}
    for k, v in call_spec.tensor_arguments_dict.items():
        if k == "mask":
            # Do not include mask tensors in shapes dict
            continue
        if k == "kwargs" or k == "args":
            # Do not include catch-alls in shapes dict
            continue
        if k in call_spec.nested_tensor_argument_names:
            shapes_dict[f"{k}_shape"] = tree.map_structure(
                standardize_shape_or_none, v
            )
        else:
            shapes_dict[f"{k}_shape"] = standardize_shape_or_none(v)
    return shapes_dict


def update_shapes_dict_for_target_fn(
    target_fn,
    shapes_dict,
    call_spec,
    class_name,
):
    """Updates a `shapes_dict` for `build()` or `compute_output_shape()`.

    This function will align a dictionary of the shapes of all tensor
    passed to `call`, with the signatures of `build()` or
    `compute_output_shape()`.

    The alignment is a follows:

    - If `build()` or `compute_output_shape()` accept only one argument,
        forward the shape of the first positional argument from call without
        checking any argument names.
    - If `build()` or `compute_output_shape()` accept multiple arguments,
        enforce that all argument names match a call argument name, e.g.
        `foo_shape` would match call argument `foo`.

    Returns:
        An updated `shapes_dict` that can be used to invoke
        `target_fn(**shapes_dict)`.
    """
    if utils.is_default(target_fn):
        return None
    sig = inspect.signature(target_fn)
    expected_names = []
    for name, param in sig.parameters.items():
        if param.kind in (
            param.POSITIONAL_OR_KEYWORD,
            param.POSITIONAL_ONLY,
            param.KEYWORD_ONLY,
        ):
            expected_names.append(name)

    # Single arg: don't check names, pass first shape.
    if len(expected_names) == 1:
        key = expected_names[0]
        values = tuple(shapes_dict.values())
        if values:
            input_shape = values[0]
        else:
            input_shape = None
        return {key: input_shape}

    # Multiple args: check that all names line up.
    kwargs = {}
    for name in expected_names:
        method_name = target_fn.__name__
        error_preamble = (
            f"For a `{method_name}()` method with more than one argument, all "
            "arguments should have a `_shape` suffix and match an argument "
            f"from `call()`. E.g. `{method_name}(self, foo_shape, bar_shape)` "
        )
        if not name.endswith("_shape"):
            raise ValueError(
                f"{error_preamble} For layer '{class_name}', "
                f"Received `{method_name}()` argument "
                f"`{name}`, which does not end in `_shape`."
            )
        expected_call_arg = utils.removesuffix(name, "_shape")
        if expected_call_arg not in call_spec.arguments_dict:
            raise ValueError(
                f"{error_preamble} For layer '{class_name}', "
                f"received `{method_name}()` argument "
                f"`{name}`, but `call()` does not have argument "
                f"`{expected_call_arg}`."
            )
        if name in shapes_dict:
            kwargs[name] = shapes_dict[name]

    return kwargs


class CallContext:
    def __init__(self, entry_layer):
        self.entry_layer = entry_layer

    def get_value(self, arg_name, default=None):
        """Get the context value for `arg_name`, or `default` if unset."""
        return getattr(self, arg_name, default)

    def set_value(self, arg_name, value):
        """Set `arg_name` = `value` on this context object."""
        setattr(self, arg_name, value)


def is_shape_tuple(s):
    return isinstance(s, (list, tuple)) and all(
        d is None or isinstance(d, int) for d in s
    )


def might_have_unbuilt_state(layer):
    return any(not lr.built for lr in layer._layers)
