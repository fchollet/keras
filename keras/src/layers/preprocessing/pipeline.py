from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import get_keras_mask
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Pipeline")
class Pipeline(Layer):
    """Applies a series of layers to an input.

    This class is useful to build a preprocessing pipeline,
    in particular an image data augmentation pipeline.
    Compared to a `Sequential` model, `Pipeline` features
    a few important differences:

    - It's not a `Model`, just a plain layer.
    - When the layers in the pipeline are compatible
        with `tf.data`, the pipeline will also
        remain `tf.data` compatible. That is to say,
        the pipeline will not attempt to convert
        its inputs to backend-native tensors
        when in a tf.data context (unlike a `Sequential`
        model).

    Example:

    ```python
    from keras import layers
    preprocessing_pipeline = layers.Pipeline([
        layers.AutoContrast(),
        layers.RandomZoom(0.2),
        layers.RandomRotation(0.2),
    ])

    # `ds` is a tf.data.Dataset
    preprocessed_ds = ds.map(
        preprocessing_pipeline,
        num_parallel_calls=4,
    )
    ```
    """

    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self._pipeline_layers = layers
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    @property
    def layers(self):
        return self._pipeline_layers

    def call(self, inputs, training=True, mask=None):
        for layer in self._pipeline_layers:
            kwargs = {}
            if layer._call_has_mask_arg:
                kwargs["mask"] = mask
            if layer._call_has_training_arg and training is not None:
                kwargs["training"] = training
            outputs = layer(inputs, **kwargs)
            inputs = outputs

            mask = tree.map_structure(get_keras_mask, outputs)
        return outputs

    @classmethod
    def from_config(cls, config):
        layers = config.get("layers")
        if not isinstance(layers, (list, tuple)):
            raise ValueError(
                "`Pipeline` config must contain a `layers` key mapping to a "
                "list of serialized layers. Received config with "
                f"layers={layers}."
            )
        config = {**config}
        config["layers"] = [
            serialization_lib.deserialize_keras_object(x) for x in layers
        ]
        return cls(**config)

    def get_config(self):
        config = {
            "layers": serialization_lib.serialize_keras_object(
                self._pipeline_layers
            ),
            "name": self.name,
        }
        return config
