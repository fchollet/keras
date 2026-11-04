import copy

from keras.src import layers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import set_keras_mask
from keras.src.quantizers.geometry import ReversibleLookupGeometry


@keras_export("keras.layers.ReversibleEmbedding")
class ReversibleEmbedding(layers.Embedding):
    """An embedding layer which can project backwards to the input dim.

    This layer is an extension of `keras.layers.Embedding` for language models.
    This layer can be called "in reverse" with `reverse=True`, in which case the
    layer will linearly project from `output_dim` back to `input_dim`.

    By default, the reverse projection will use the transpose of the
    `embeddings` weights to project to `input_dim` (weights are "tied"). If
    `tie_weights=False`, the model will use a separate, trainable variable for
    reverse projection.

    This layer has no bias terms.

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
        reverse_dtype: The dtype for the reverse projection computation.
            Defaults to the `compute_dtype` of the layer.
        logit_soft_cap: If `logit_soft_cap` is set and `reverse=True`, the
            output logits will be scaled by
            `tanh(logits / logit_soft_cap) * logit_soft_cap`. This narrows the
            range of output logits and can improve training.
        **kwargs: other keyword arguments passed to `keras.layers.Embedding`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to the layer.
        reverse: Boolean. If `True` the layer will perform a linear projection
            from `output_dim` to `input_dim`, instead of a normal embedding
            call. Default to `False`.

    Example:
    ```python
    batch_size = 16
    vocab_size = 100
    hidden_dim = 32
    seq_length = 50

    # Generate random inputs.
    token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))

    embedding = keras.layers.ReversibleEmbedding(vocab_size, hidden_dim)
    # Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.
    hidden_states = embedding(token_ids)
    # Project hidden states to shape `(batch_size, seq_length, vocab_size)`.
    logits = embedding(hidden_states, reverse=True)
    ```

    References:
    - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        reverse_dtype=None,
        logit_soft_cap=None,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = tie_weights
        self.reverse_dtype = reverse_dtype
        self.logit_soft_cap = logit_soft_cap

    def build(self, input_shape=None):
        super().build(input_shape)
        if not self.tie_weights and not self._strategy_owns_weight_storage():
            self.reverse_embeddings = self.add_weight(
                shape=(self.output_dim, self.input_dim),
                initializer=self.embeddings_initializer,
                name="reverse_embeddings",
                trainable=True,
            )

    def call(self, inputs, reverse=False):
        if not reverse:
            result = super().call(inputs)
            mask = super().compute_mask(inputs)
            if mask is not None:
                set_keras_mask(result, mask)
            return result
        else:
            if self.tie_weights:
                kernel = ops.transpose(self.embeddings)
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            logits = ops.matmul(inputs, kernel)
            # Optionally soft-cap logits.
            if self.logit_soft_cap is not None:
                soft_cap = self.logit_soft_cap
                logits = ops.multiply(
                    ops.tanh(ops.divide(logits, soft_cap)), soft_cap
                )
            return logits

    def quantized_call(self, inputs, reverse=False):
        outputs = super().quantized_call(inputs, reverse=reverse)
        if not reverse:
            # The forward lookup carries the mask, as in float mode.
            mask = super().compute_mask(inputs)
            if mask is not None:
                set_keras_mask(outputs, mask)
        return outputs

    def compute_mask(self, inputs, mask=None):
        # Disable masking from super class, masking is done directly in call.
        return None

    def compute_output_shape(self, input_shape, reverse=False):
        output_shape = list(input_shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return output_shape

    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return KerasTensor(output_shape, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
                "logit_soft_cap": self.logit_soft_cap,
            }
        )
        return config

    @property
    def variable_serialization_spec(self):
        # Avoid modifying the parent's spec.
        _spec = copy.deepcopy(super().variable_serialization_spec)
        if not self.tie_weights:
            for mode, variable_spec in _spec.items():
                variable_spec.append("reverse_embeddings")
                if mode in ("int4", "int8"):
                    variable_spec.append("reverse_embeddings_scale")
                if mode == "int4":
                    # reverse_embeddings_zero only exists for sub-channel
                    variable_spec.append("reverse_embeddings_zero")
        return _spec

    def _quantization_geometry(self):
        return ReversibleLookupGeometry(self)
