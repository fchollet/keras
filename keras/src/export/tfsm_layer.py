from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.export.saved_model import _list_variables_used_by_fns
from keras.src.saving import serialization_lib
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.TFSMLayer")
class TFSMLayer(layers.Layer):
    """Reload a Keras model/layer that was saved via SavedModel / ExportArchive.

    Arguments:
        filepath: `str` or `pathlib.Path` object. The path to the SavedModel.
        call_endpoint: Name of the endpoint to use as the `call()` method
            of the reloaded layer. If the SavedModel was created
            via `model.export()`,
            then the default endpoint name is `'serve'`. In other cases
            it may be named `'serving_default'`.

    Example:

    ```python
    model.export("path/to/artifact")
    reloaded_layer = TFSMLayer("path/to/artifact")
    outputs = reloaded_layer(inputs)
    ```

    The reloaded object can be used like a regular Keras layer, and supports
    training/fine-tuning of its trainable weights. Note that the reloaded
    object retains none of the internal structure or custom methods of the
    original object -- it's a brand new layer created around the saved
    function.

    **Limitations:**

    * Only call endpoints with a single `inputs` tensor argument
    (which may optionally be a dict/tuple/list of tensors) are supported.
    For endpoints with multiple separate input tensor arguments, consider
    subclassing `TFSMLayer` and implementing a `call()` method with a
    custom signature.
    * If you need training-time behavior to differ from inference-time behavior
    (i.e. if you need the reloaded object to support a `training=True` argument
    in `__call__()`), make sure that the training-time call function is
    saved as a standalone endpoint in the artifact, and provide its name
    to the `TFSMLayer` via the `call_training_endpoint` argument.

    **Safe mode:**

    Instantiating a `TFSMLayer` loads an external SavedModel, which carries a
    potential risk of arbitrary code execution. The constructor therefore
    raises a `ValueError` when it runs while safe mode is active, that is,
    during a deserialization performed with the default `safe_mode=True`. This
    covers a `TFSMLayer` reconstructed from a config as well as one
    constructed by a custom layer, model or metric that is itself being
    deserialized -- including `keras.models.clone_model()` of such a
    subclassed model, which offers no `safe_mode` argument. To load these
    artifacts, pass `safe_mode=False` to the loading function or call
    `keras.config.enable_unsafe_deserialization()`. Constructing a `TFSMLayer`
    outside of deserialization is not affected.
    """

    def __init__(
        self,
        filepath,
        call_endpoint="serve",
        call_training_endpoint=None,
        trainable=True,
        name=None,
        dtype=None,
    ):
        if backend.backend() != "tensorflow":
            raise NotImplementedError(
                "The TFSMLayer is only currently supported with the "
                "TensorFlow backend."
            )

        # Guard the load itself: deserialization can invoke the constructor
        # without going through `from_config()`, e.g. via a serialized
        # callable or via a custom layer that forwards a config value to
        # `TFSMLayer(filepath)`. A construction by a custom layer, model or
        # metric that is itself being deserialized in safe mode is also
        # rejected and needs `safe_mode=False`. Direct construction outside
        # a deserialization scope is allowed.
        if serialization_lib.in_safe_mode():
            raise ValueError(
                "Requested the deserialization of a `TFSMLayer`, which "
                "loads an external SavedModel. This carries a potential risk "
                "of arbitrary code execution and thus it is disallowed by "
                "default. If you trust the source of the artifact, you can "
                "override this error by passing `safe_mode=False` to the "
                "loading function, or calling "
                "`keras.config.enable_unsafe_deserialization()`. "
                f"SavedModel filepath: '{filepath}'"
            )

        # Initialize an empty layer, then add_weight() etc. as needed.
        super().__init__(trainable=trainable, name=name, dtype=dtype)

        self._reloaded_obj = tf.saved_model.load(filepath)

        self.filepath = filepath
        self.call_endpoint = call_endpoint
        self.call_training_endpoint = call_training_endpoint

        # Resolve the call function.
        if hasattr(self._reloaded_obj, call_endpoint):
            # Case 1: it's set as an attribute.
            self.call_endpoint_fn = getattr(self._reloaded_obj, call_endpoint)
        elif call_endpoint in self._reloaded_obj.signatures:
            # Case 2: it's listed in the `signatures` field.
            self.call_endpoint_fn = self._reloaded_obj.signatures[call_endpoint]
        else:
            raise ValueError(
                f"The endpoint '{call_endpoint}' "
                "is neither an attribute of the reloaded SavedModel, "
                "nor an entry in the `signatures` field of "
                "the reloaded SavedModel. Select another endpoint via "
                "the `call_endpoint` argument. Available endpoints for "
                "this SavedModel: "
                f"{list(self._reloaded_obj.signatures.keys())}"
            )

        # Resolving the training function.
        if call_training_endpoint:
            if hasattr(self._reloaded_obj, call_training_endpoint):
                self.call_training_endpoint_fn = getattr(
                    self._reloaded_obj, call_training_endpoint
                )
            elif call_training_endpoint in self._reloaded_obj.signatures:
                self.call_training_endpoint_fn = self._reloaded_obj.signatures[
                    call_training_endpoint
                ]
            else:
                raise ValueError(
                    f"The endpoint '{call_training_endpoint}' "
                    "is neither an attribute of the reloaded SavedModel, "
                    "nor an entry in the `signatures` field of "
                    "the reloaded SavedModel. Available endpoints for "
                    "this SavedModel: "
                    f"{list(self._reloaded_obj.signatures.keys())}"
                )

        # Add trainable and non-trainable weights from the call_endpoint_fn.
        all_fns = [self.call_endpoint_fn]
        if call_training_endpoint:
            all_fns.append(self.call_training_endpoint_fn)
        tvs, ntvs = _list_variables_used_by_fns(all_fns)
        for v in tvs:
            self._add_existing_weight(v)
        for v in ntvs:
            self._add_existing_weight(v)

        self._build_at_init()

    def _add_existing_weight(self, weight):
        """Tracks an existing weight."""
        variable = backend.Variable(
            initializer=weight,
            trainable=weight.trainable,
            dtype=weight.dtype,
            shape=weight.shape,
            # Keras variable names cannot contain slashes.
            name=weight.name.replace("/", "_"),
        )
        self._track_variable(variable)

    def call(self, inputs, training=False, **kwargs):
        if training:
            if self.call_training_endpoint:
                return self.call_training_endpoint_fn(inputs, **kwargs)
        return self.call_endpoint_fn(inputs, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        config = {
            # Note: this is not intended to be portable.
            "filepath": self.filepath,
            "call_endpoint": self.call_endpoint,
            "call_training_endpoint": self.call_training_endpoint,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None, safe_mode=None):
        """Creates a TFSMLayer from its config.
        Args:
            config: A Python dictionary, typically the output of `get_config`.
            custom_objects: Optional dictionary mapping names to custom objects.
            safe_mode: Boolean, whether to disallow loading TFSMLayer.
                When `safe_mode=True`, loading is disallowed because TFSMLayer
                loads external SavedModels that may contain attacker-controlled
                executable graph code. Defaults to `True`.
        Returns:
            A TFSMLayer instance.
        """
        effective_safe_mode = (
            safe_mode
            if safe_mode is not None
            else serialization_lib.in_safe_mode()
        )

        # Default to safe mode for direct from_config() calls, and propagate
        # explicit overrides to the constructor's SavedModel load guard.
        with serialization_lib.SafeModeScope(effective_safe_mode is not False):
            return cls(**config)
