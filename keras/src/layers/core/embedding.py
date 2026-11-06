import warnings

from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.layers.layer import Layer
from keras.src.quantizers.geometry import LookupGeometry
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Embedding")
class Embedding(Layer):
    """Turns nonnegative integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used on nonnegative integer inputs of a fixed range.

    Example:

    >>> model = keras.Sequential()
    >>> model.add(keras.layers.Embedding(1000, 64))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
            This is useful when using recurrent layers which
            may take variable length input. If this is `True`,
            then all subsequent layers in the model need
            to support masking or an exception will be raised.
            If `mask_zero` is set to `True`, as a consequence,
            index 0 cannot be used in the vocabulary (`input_dim` should
            equal size of vocabulary + 1).
        weights: Optional floating-point matrix of size
            `(input_dim, output_dim)`. The initial embeddings values
            to use.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's embeddings
            matrix to non-trainable and replaces it with a delta over the
            original matrix, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large embedding layers.
            You can also enable LoRA on an existing
            `Embedding` layer by calling `layer.enable_lora(rank)`.
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.

    Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        weights=None,
        lora_rank=None,
        lora_alpha=None,
        quantization_config=None,
        **kwargs,
    ):
        if (
            not isinstance(input_dim, int)
            or isinstance(input_dim, bool)
            or input_dim <= 0
        ):
            raise ValueError(
                "`input_dim` must be a positive integer. "
                f"Received: input_dim={input_dim} "
                f"(of type {type(input_dim).__name__})."
            )
        if (
            not isinstance(output_dim, int)
            or isinstance(output_dim, bool)
            or output_dim <= 0
        ):
            raise ValueError(
                "`output_dim` must be a positive integer. "
                f"Received: output_dim={output_dim} "
                f"(of type {type(output_dim).__name__})."
            )
        input_length = kwargs.pop("input_length", None)
        if input_length is not None:
            warnings.warn(
                "Argument `input_length` is deprecated. Just remove it."
            )
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.autocast = False
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_enabled = False
        self.quantization_config = quantization_config

        if weights is not None:
            self.build()
            if not (isinstance(weights, list) and len(weights) == 1):
                weights = [weights]
            self.set_weights(weights)

    def build(self, input_shape=None):
        if self.built:
            return
        embeddings_shape = (self.input_dim, self.output_dim)
        if self.quantization_mode:
            self.quantized_build(
                embeddings_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        if not self._strategy_owns_weight_storage():
            self._embeddings = self.add_weight(
                shape=embeddings_shape,
                initializer=self.embeddings_initializer,
                name="embeddings",
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                trainable=True,
            )
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank, lora_alpha=self.lora_alpha)

    @property
    def embeddings(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `embeddings`."
            )
        # A quantized layer exposes its integer codes in the table's shape;
        # without a `QuantizedWeight` view, the float table.
        quantized_weight = self._quantized_weight()
        embeddings = (
            self._embeddings
            if quantized_weight is None
            else quantized_weight.unpack()
        )
        if self.lora_enabled:
            embeddings = ops.cast(
                ops.add(
                    embeddings,
                    (self.lora_alpha / self.lora_rank)
                    * ops.matmul(
                        self.lora_embeddings_a, self.lora_embeddings_b
                    ),
                ),
                dtype=self.compute_dtype,
            )
        return embeddings

    def call(self, inputs):
        if inputs.dtype != "int32" and inputs.dtype != "int64":
            inputs = ops.cast(inputs, "int32")
        outputs = ops.take(self.embeddings, inputs, axis=0)
        return ops.cast(outputs, dtype=self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        return (*input_shape, self.output_dim)

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        ragged = getattr(inputs, "ragged", False)
        return KerasTensor(
            output_shape, dtype=self.compute_dtype, ragged=ragged
        )

    def enable_lora(
        self,
        rank,
        lora_alpha=None,
        a_initializer="he_uniform",
        b_initializer="zeros",
    ):
        if self.embeddings_constraint:
            raise ValueError(
                "Lora is incompatible with embedding constraints. "
                "In order to enable lora on this layer, remove the "
                "`embeddings_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. This can only be done once per layer."
            )
        self._check_lora_supported(self.quantization_mode)
        self._tracker.unlock()

        # LoRA weights should be float32 to avoid the risk of underflow or
        # overflow during fine-tuning.
        # When deploying the model, these weights should be merged with the
        # original embedding while maintaining the original embedding's dtype.
        self.lora_embeddings_a = self.add_weight(
            name="lora_embeddings_a",
            shape=(self.input_dim, rank),
            initializer=initializers.get(a_initializer),
            dtype="float32",
            regularizer=self.embeddings_regularizer,
        )
        self.lora_embeddings_b = self.add_weight(
            name="lora_embeddings_b",
            shape=(rank, self.output_dim),
            initializer=initializers.get(b_initializer),
            dtype="float32",
            regularizer=self.embeddings_regularizer,
        )
        # Freeze the underlying variable, not the `embeddings` property, which
        # returns a throwaway unpacked tensor in int4 mode.
        self._embeddings.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "embeddings")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "embeddings")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": initializers.serialize(
                self.embeddings_initializer
            ),
            "embeddings_regularizer": regularizers.serialize(
                self.embeddings_regularizer
            ),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "embeddings_constraint": constraints.serialize(
                self.embeddings_constraint
            ),
            "mask_zero": self.mask_zero,
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config", None)
            )
        )
        return super().from_config(config)

    def _quantization_mode_error(self, mode):
        return NotImplementedError(
            "Invalid quantization mode. Expected one of ('int8', 'int4'). "
            f"Received: quantization_mode={mode}"
        )

    @property
    def variable_serialization_spec(self):
        """Returns a dict mapping quantization modes to variable names in order.

        This spec is used by `save_own_variables` and `load_own_variables` to
        determine the correct ordering of variables during serialization for
        each quantization mode. `None` means no quantization.
        """
        return {
            None: [
                "embeddings",
            ],
            "int8": [
                "embeddings",
                "embeddings_scale",
            ],
            "int4": [
                "embeddings",
                "embeddings_scale",
                "embeddings_zero",
                "g_idx",
            ],
        }

    def _quantization_geometry(self):
        return LookupGeometry(self)
