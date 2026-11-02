from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import standardize_dtype
from keras.src.layers.layer import Layer


@keras_export("keras.layers.Attention")
class Attention(Layer):
    """Dot-product attention layer, a.k.a. Luong-style attention.

    Inputs are a list with 2 or 3 elements:
    1. A `query` tensor of shape `(batch_size, Tq, dim)`.
    2. A `value` tensor of shape `(batch_size, Tv, dim)`.
    3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
        supplied, `value` will be used as a `key`.

    The calculation follows the steps:
    1. Calculate attention scores using `query` and `key` with shape
        `(batch_size, Tq, Tv)`.
    2. Use scores to calculate a softmax distribution with shape
        `(batch_size, Tq, Tv)`.
    3. Use the softmax distribution to create a linear combination of `value`
        with shape `(batch_size, Tq, dim)`.

    Args:
        use_scale: If `True`, will create a scalar variable to scale the
            attention scores.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            attention scores. Defaults to `0.0`.
        seed: A Python integer to use as random seed in case of `dropout`.
        score_mode: Function to use to compute attention scores, one of
            `{"dot", "concat"}`. `"dot"` refers to the dot product between the
            query and key vectors. `"concat"` refers to the hyperbolic tangent
            of the concatenation of the `query` and `key` vectors.

    Call arguments:
        inputs: List of the following tensors:
            - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
            - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
            - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
                not given, will use `value` for both `key` and `value`, which is
                the most common case.
        mask: List of the following tensors:
            - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
                If given, the output will be zero at the positions where
                `mask==False`.
            - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
                If given, will apply the mask such that values at positions
                 where `mask==False` do not contribute to the result.
        return_attention_scores: bool, it `True`, returns the attention scores
            (after masking and softmax) as an additional output argument.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
        use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
            a mask such that position `i` cannot attend to positions `j > i`.
            This prevents the flow of information from the future towards the
            past. Defaults to `False`.

    Output:
        Attention outputs of shape `(batch_size, Tq, dim)`.
        (Optional) Attention scores after masking and softmax with shape
            `(batch_size, Tq, Tv)`.
    """

    def __init__(
        self,
        use_scale=False,
        score_mode="dot",
        dropout=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_scale = use_scale
        self.score_mode = score_mode
        self.dropout = dropout
        self.seed = seed
        if self.dropout > 0:
            self.seed_generator = backend.random.SeedGenerator(seed=seed)

        if self.score_mode not in ["dot", "concat"]:
            raise ValueError(
                "Invalid value for argument score_mode. "
                "Expected one of {'dot', 'concat'}. "
                f"Received: score_mode={score_mode}"
            )

    def build(self, input_shape):
        self._validate_inputs(input_shape)
        self.scale = None
        self.concat_score_weight = None
        if self.use_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )
        if self.score_mode == "concat":
            self.concat_score_weight = self.add_weight(
                name="concat_score_weight",
                shape=(),
                initializer="ones",
                dtype=self.dtype,
                trainable=True,
            )

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a query-key dot product.

        Args:
            query: Query tensor of shape `(batch_size, Tq, dim)`.
            key: Key tensor of shape `(batch_size, Tv, dim)`.

        Returns:
            Tensor of shape `(batch_size, Tq, Tv)`.
        """
        if self.score_mode == "dot":
            scores = backend.ops.numpy.matmul(
                query, backend.ops.numpy.swapaxes(key, -2, -1)
            )
            if self.scale is not None:
                scores = backend.ops.numpy.multiply(scores, self.scale)
        elif self.score_mode == "concat":
            # Reshape tensors to enable broadcasting.
            # Reshape into [batch_size, Tq, 1, dim].
            q_reshaped = backend.ops.numpy.expand_dims(query, axis=-2)
            # Reshape into [batch_size, 1, Tv, dim].
            k_reshaped = backend.ops.numpy.expand_dims(key, axis=-3)
            if self.scale is not None:
                scores = self.concat_score_weight * backend.ops.numpy.sum(
                    backend.ops.numpy.tanh(
                        self.scale * (q_reshaped + k_reshaped)
                    ),
                    axis=-1,
                )
            else:
                scores = self.concat_score_weight * backend.ops.numpy.sum(
                    backend.ops.numpy.tanh(q_reshaped + k_reshaped), axis=-1
                )
        else:
            raise ValueError("scores not computed")

        return scores

    def _apply_scores(self, scores, value, scores_mask=None, training=False):
        """Applies attention scores to the given value tensor.

        To use this method in your attention layer, follow the steps:

        * Use `query` tensor of shape `(batch_size, Tq)` and `key` tensor of
            shape `(batch_size, Tv)` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
            `scores_mask`, calculates
            `attention_distribution = softmax(scores)`, then returns
            `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.

        Args:
            scores: Scores float tensor of shape `(batch_size, Tq, Tv)`.
            value: Value tensor of shape `(batch_size, Tv, dim)`.
            scores_mask: A boolean mask tensor of shape `(batch_size, 1, Tv)`
                or `(batch_size, Tq, Tv)`. If given, scores at positions where
                `scores_mask==False` do not contribute to the result. A line
                with no `True` value attends to nothing, so its result and
                attention scores are 0.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode
                (no dropout).

        Returns:
            Tensor of shape `(batch_size, Tq, dim)`.
            Attention scores after masking and softmax with shape
                `(batch_size, Tq, Tv)`.
        """
        if scores_mask is not None:
            padding_mask = backend.ops.numpy.logical_not(scores_mask)
            # Replace masked scores with a large negative constant so they do
            # not contribute to the attention distribution. Note 65504. is the
            # max float16 value. A constant rather than a bias keeps the
            # scores finite: in float16, `scores - 65504.` overflows to -inf
            # for any score below -16, and a row of -inf gives NaN weights.
            max_value = (
                65504.0
                if standardize_dtype(scores.dtype) == "float16"
                else 1.0e9
            )
            if len(padding_mask.shape) == 2:
                padding_mask = backend.ops.numpy.expand_dims(
                    padding_mask, axis=-2
                )
            scores = backend.ops.numpy.where(padding_mask, -max_value, scores)

        weights = backend.ops.nn.softmax(scores, axis=-1)
        if scores_mask is not None:
            # Zero out masked positions in case a whole row is masked, where
            # softmax would otherwise spread the weights over masked values.
            weights = backend.ops.numpy.where(padding_mask, 0.0, weights)
        if training and self.dropout > 0:
            weights = backend.random.dropout(
                weights,
                self.dropout,
                seed=self.seed_generator,
            )
        return backend.ops.numpy.matmul(weights, value), weights

    def _calculate_score_mask(self, scores, v_mask, use_causal_mask):
        if use_causal_mask:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j > i. This prevents the flow of information from the
            # future into the past.
            score_shape = backend.ops.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            mask_shape = (1, score_shape[-2], score_shape[-1])
            ones_mask = backend.ops.numpy.ones(shape=mask_shape, dtype="int32")
            row_index = backend.ops.numpy.cumsum(ones_mask, axis=-2)
            col_index = backend.ops.numpy.cumsum(ones_mask, axis=-1)
            causal_mask = backend.ops.numpy.greater_equal(row_index, col_index)

            if v_mask is not None:
                # Mask of shape [batch_size, 1, Tv].
                v_mask = backend.ops.numpy.expand_dims(v_mask, axis=-2)
                return backend.ops.numpy.logical_and(v_mask, causal_mask)
            return causal_mask
        else:
            # If not using causal mask, return the value mask as is,
            # or None if the value mask is not provided.
            return v_mask

    def call(
        self,
        inputs,
        mask=None,
        training=False,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        self._validate_inputs(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        scores_mask = self._calculate_score_mask(
            scores, v_mask, use_causal_mask
        )
        attention_output, attention_scores = self._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask, training=training
        )
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = backend.ops.numpy.expand_dims(q_mask, axis=-1)
            attention_output = backend.ops.numpy.where(
                q_mask, attention_output, 0
            )
        if return_attention_scores:
            return (attention_output, attention_scores)
        else:
            return attention_output

    def compute_mask(self, inputs, mask=None):
        self._validate_inputs(inputs=inputs, mask=mask)
        if mask is None or mask[0] is None:
            return None
        return ops.convert_to_tensor(mask[0])

    def compute_output_shape(self, input_shape):
        query_shape = input_shape[0]
        value_shape = input_shape[1]
        return (*query_shape[:-1], value_shape[-1])

    def compute_output_spec(
        self,
        inputs,
        mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        self._validate_inputs(inputs, mask)
        query = inputs[0]
        value = inputs[1]
        key = inputs[2] if len(inputs) > 2 else value

        output_shape = self.compute_output_shape(
            [query.shape, value.shape, key.shape]
        )
        output_spec = KerasTensor(output_shape, dtype=self.compute_dtype)

        if return_attention_scores:
            scores_shape = (
                *query.shape[:-1],
                key.shape[-2],
            )  # (*batch_dims, Tq, Tv)
            return output_spec, KerasTensor(
                scores_shape, dtype=self.compute_dtype
            )

        return output_spec

    def _validate_inputs(self, inputs, mask=None):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError(
                f"{class_name} layer must be called on a list of inputs, "
                "namely [query, value] or [query, value, key]. "
                f"Received: inputs={inputs}."
            )
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError(
                f"{class_name} layer accepts inputs list of length 2 or 3, "
                "namely [query, value] or [query, value, key]. "
                f"Received length: {len(inputs)}."
            )
        if mask is not None:
            if not isinstance(mask, list):
                raise ValueError(
                    f"{class_name} layer mask must be a list, "
                    f"namely [query_mask, value_mask]. Received: mask={mask}."
                )
            if len(mask) < 2 or len(mask) > 3:
                raise ValueError(
                    f"{class_name} layer accepts mask list of length 2 or 3. "
                    f"Received: inputs={inputs}, mask={mask}."
                )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "use_scale": self.use_scale,
            "score_mode": self.score_mode,
            "dropout": self.dropout,
            "seed": self.seed,
        }
        return {**base_config, **config}
