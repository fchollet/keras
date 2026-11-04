import copy
import inspect
import typing

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_shape
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.functional import Functional
from keras.src.models.model import Model
from keras.src.saving import serialization_lib


@keras_export(["keras.Sequential", "keras.models.Sequential"])
class Sequential(Model):
    """`Sequential` groups a linear stack of layers into a `Model`.

    Examples:

    ```python
    model = keras.Sequential()
    model.add(keras.Input(shape=(16,)))
    model.add(keras.layers.Dense(8))

    # Note that you can also omit the initial `Input`.
    # In that case the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(4))
    # model.weights not created yet

    # Whereas if you specify an `Input`, the model gets built
    # continuously as you are adding layers:
    model = keras.Sequential()
    model.add(keras.Input(shape=(16,)))
    model.add(keras.layers.Dense(8))
    len(model.weights)  # Returns "2"

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(4))
    model.build((None, 16))
    len(model.weights)  # Returns "4"

    # Note that when using the delayed-build pattern (no input shape specified),
    # the model gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the model on some input data.
    model = keras.Sequential()
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)
    ```
    """

    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, layers=None, trainable=True, name=None):
        super().__init__(trainable=trainable, name=name)
        self._functional = None
        self._layers = []
        if layers:
            for layer in layers:
                self.add(layer, rebuild=False)
            self._maybe_rebuild()

    def add(self, layer, rebuild=True):
        """Adds a layer instance on top of the layer stack.

        Args:
            layer: layer instance.
        """
        # Legacy case: if the first layer has an input_shape arg,
        # use it to build an InputLayer.
        if not self._layers:
            if getattr(layer, "_input_shape_arg", None) is not None:
                self.add(InputLayer(shape=layer._input_shape_arg))

        # If we are passed a Keras tensor created by keras.Input(), we
        # extract the input layer from its keras history and use that.
        if hasattr(layer, "_keras_history"):
            origin_layer = layer._keras_history[0]
            if isinstance(origin_layer, InputLayer):
                layer = origin_layer
        if not isinstance(layer, Layer):
            raise ValueError(
                "Only instances of `keras.Layer` can be "
                f"added to a Sequential model. Received: {layer} "
                f"(of type {type(layer)})"
            )
        if not self._is_layer_name_unique(layer):
            raise ValueError(
                "All layers added to a Sequential model "
                f"should have unique names. Name '{layer.name}' is already "
                "the name of a layer in this model. Update the `name` argument "
                "to pass a unique name."
            )
        if (
            isinstance(layer, InputLayer)
            and self._layers
            and isinstance(self._layers[0], InputLayer)
        ):
            raise ValueError(
                f"Sequential model '{self.name}' has already been configured "
                f"to use input shape {self._layers[0].batch_shape}. You cannot "
                f"add a different Input layer to it."
            )

        self._layers.append(layer)
        if rebuild:
            self._maybe_rebuild()
        else:
            self.built = False
            self._functional = None

    def pop(self, rebuild=True):
        """Removes the last layer in the model.

        Args:
            rebuild: `bool`. Whether to rebuild the model after removing
            the layer. Defaults to `True`.

        Returns:
            layer: layer instance.
        """
        layer = self._layers.pop()
        self.built = False
        self._functional = None
        if rebuild:
            self._maybe_rebuild()
        return layer

    def _maybe_rebuild(self):
        self.built = False
        self._functional = None
        if not self._layers:
            return
        if isinstance(self._layers[0], InputLayer) and len(self._layers) > 1:
            input_shape = self._layers[0].batch_shape
            self.build(input_shape)
        elif hasattr(self._layers[0], "input_shape") and len(self._layers) > 1:
            # We can build the Sequential model if the first layer has the
            # `input_shape` property. This is most commonly found in Functional
            # model.
            input_shape = self._layers[0].input_shape
            self.build(input_shape)

    def _lock_state(self):
        # Unlike other layers, Sequential is mutable after build.
        pass

    def _obj_type(self):
        return "Sequential"

    def build(self, input_shape=None):
        try:
            input_shape = standardize_shape(input_shape)
        except:
            # Do not attempt to build if the model does not have a single
            # input tensor.
            return
        if not self._layers:
            raise ValueError(
                f"Sequential model {self.name} cannot be built because it has "
                "no layers. Call `model.add(layer)`."
            )
        if isinstance(self._layers[0], InputLayer):
            if self._layers[0].batch_shape != input_shape:
                raise ValueError(
                    f"Sequential model '{self.name}' has already been "
                    "configured to use input shape "
                    f"{self._layers[0].batch_shape}. You cannot build it "
                    f"with input_shape {input_shape}"
                )
        else:
            dtype = self._layers[0].compute_dtype
            self._layers = [
                InputLayer(batch_shape=input_shape, dtype=dtype)
            ] + self._layers

        # Build functional model
        inputs = self._layers[0].output
        x = inputs
        for layer in self._layers[1:]:
            try:
                x = layer(x)
            except NotImplementedError:
                # Can happen if shape inference is not implemented.
                # TODO: consider reverting inbound nodes on layers processed.
                return
            except TypeError as e:
                signature = inspect.signature(layer.call)
                positional_args = [
                    param
                    for param in signature.parameters.values()
                    if param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                required_positional_args = [
                    param
                    for param in positional_args
                    if param.default == inspect.Parameter.empty
                ]
                if not positional_args:
                    raise ValueError(
                        "Layers added to a Sequential model should "
                        "have a single positional argument, the "
                        "input tensor. Layer "
                        f"{layer.__class__.__name__} has no "
                        "positional arguments."
                    )
                if len(required_positional_args) > 1:
                    raise ValueError(
                        "Layers added to a Sequential model can "
                        "only have a single required positional "
                        "argument, the input tensor. Layer "
                        f"{layer.__class__.__name__} has multiple "
                        "required positional arguments: "
                        f"{required_positional_args}"
                    )
                raise e
        outputs = x
        self._functional = Functional(inputs=inputs, outputs=outputs)

    def call(self, inputs, training=None, mask=None):
        if self._functional:
            return self._functional.call(inputs, training=training, mask=mask)

        # Fallback: Just apply the layer sequence.
        # This typically happens if `inputs` is a nested struct.
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and
            # `outputs` are the outputs of `layer` applied to `inputs`. At the
            # end of each iteration `inputs` is set to `outputs` to prepare for
            # the next layer.
            if layer._call_has_mask_arg:
                outputs = layer(inputs, mask=mask)
            else:
                outputs = layer(inputs)
            inputs = outputs

            mask = tree.map_structure(get_keras_mask, outputs)
        return outputs

    @property
    def layers(self):
        # Historically, `sequential.layers` only returns layers that were added
        # via `add`, and omits the auto-generated `InputLayer` that comes at the
        # bottom of the stack.
        layers = self._layers
        if layers and isinstance(layers[0], InputLayer):
            return layers[1:]
        return layers[:]

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Sequential.layers` attribute is reserved and should not be used. "
            "Use `add()` and `pop()` to change the layers in this model."
        )

    def compute_output_spec(self, inputs, training=None, mask=None, **kwargs):
        if self._functional:
            return self._functional.compute_output_spec(
                inputs, training=training, mask=mask, **kwargs
            )
        # Direct application
        for layer in self.layers:
            outputs = layer.compute_output_spec(
                inputs,
                training=training,
                **kwargs,
            )  # Ignore mask
            inputs = outputs
        return outputs

    def compute_output_shape(self, input_shape):
        if self._functional:
            return self._functional.compute_output_shape(input_shape)
        # Direct application
        for layer in self.layers:
            output_shape = layer.compute_output_shape(input_shape)
            input_shape = output_shape
        return output_shape

    @property
    def input_shape(self):
        if self._functional:
            return self._functional.input_shape
        raise AttributeError(
            f"Sequential model '{self.name}' has no defined input shape yet."
        )

    @property
    def output_shape(self):
        if self._functional:
            return self._functional.output_shape
        raise AttributeError(
            f"Sequential model '{self.name}' has no defined output shape yet."
        )

    @property
    def inputs(self):
        if self._functional:
            return self._functional.inputs
        raise AttributeError(
            f"Sequential model '{self.name}' has no defined inputs yet."
        )

    @property
    def outputs(self):
        if self._functional:
            return self._functional.outputs
        raise AttributeError(
            f"Sequential model '{self.name}' has no defined outputs yet."
        )

    @property
    def input_dtype(self):
        # Sequential.__call__ will try to convert its inputs
        # to the dtype expected by its input layer, if any.
        layers = self._layers
        if layers and isinstance(layers[0], InputLayer):
            return layers[0].dtype
        return super().input_dtype

    def _is_layer_name_unique(self, layer):
        for ref_layer in self._layers:
            if layer.name == ref_layer.name and ref_layer is not layer:
                return False
        return True

    def get_config(self):
        serialize_fn = serialization_lib.serialize_keras_object
        use_legacy_config = global_state.get_global_attribute(
            "use_legacy_config", False
        )
        if use_legacy_config:
            # Legacy format serialization used for H5 and SavedModel formats
            serialize_fn = legacy_serialization.serialize_keras_object
        layer_configs = []
        for layer in super().layers:
            # `super().layers` include the InputLayer if available (it is
            # filtered out of `self.layers`).
            layer_configs.append(serialize_fn(layer))
        config = Model.get_config(self)
        config["name"] = self.name
        config["layers"] = copy.deepcopy(layer_configs)
        if use_legacy_config and self._functional is not None:
            config["build_input_shape"] = self._layers[0].batch_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "layers" not in config:
            raise ValueError(
                "A Sequential model configuration must be a dictionary "
                f"containing the 'layers' key. Received: config={config}"
            )

        kwargs = {k: v for k, v in config.items() if k in ("name", "trainable")}
        model = cls(**kwargs)

        layer_configs = config["layers"]
        is_legacy = False
        for layer_config in layer_configs:
            if "module" not in layer_config:
                # Legacy format deserialization (no "module" key)
                # used for H5 and SavedModel formats
                is_legacy = True
                layer = saving_utils.model_from_config(
                    layer_config,
                    custom_objects=custom_objects,
                )
            else:
                layer = serialization_lib.deserialize_keras_object(
                    layer_config,
                    custom_objects=custom_objects,
                )
            # Do not build now, this will be done by `build_from_config`.
            model.add(layer, rebuild=False)

        if is_legacy:
            build_input_shape = config.get("build_input_shape")
            if (
                not model._functional
                and build_input_shape
                and isinstance(build_input_shape, (tuple, list))
            ):
                model.build(build_input_shape)
            else:
                model._maybe_rebuild()
        return model
