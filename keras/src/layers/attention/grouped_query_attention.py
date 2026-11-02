import math

from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.backend.config import is_flash_attention_enabled
from keras.src.layers.activations.softmax import Softmax
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.layer import Layer
from keras.src.layers.regularization.dropout import Dropout


@keras_export(
    ["keras.layers.GroupedQueryAttention", "keras.layers.GroupQueryAttention"]
)
class GroupedQueryAttention(Layer):
    """Grouped Query Attention layer.

    This is an implementation of grouped-query attention introduced by
    [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here
    `num_key_value_heads` denotes number of groups, setting
    `num_key_value_heads` to 1 is equivalent to multi-query attention, and
    when `num_key_value_heads` is equal to `num_query_heads` it is equivalent
    to multi-head attention.

    This layer first projects `query`, `key`, and `value` tensors. Then, `key`
    and `value` are repeated to match the number of heads of `query`.

    Then, the `query` is scaled and dot-producted with `key` tensors. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities and concatenated back to a single
    tensor.

    Args:
        head_dim: Size of each attention head.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key and value attention heads.
        dropout: Dropout probability.
        use_bias: Boolean, whether the dense layers use bias vectors/matrices.
        sliding_window: Optional positive integer. If set, restricts each
            query position to attend only to key positions within
            `sliding_window - 1` of itself (a symmetric band). Composes
            naturally with `use_causal_mask=True` to give the causal
            sliding window used by Mistral, Llama-3 long-context, and
            Phi-3. Defaults to `None` (full attention).
        flash_attention: If `None`, the layer attempts to use flash
            attention for faster and more memory-efficient attention
            computations when possible. This behavior can be configured using
            `keras.config.enable_flash_attention()` or
            `keras.config.disable_flash_attention()`.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        use_gate: Boolean, whether to apply a gated attention mechanism.
            When True, an additional gating branch is added based on the
            (Gated Attention for Large Language Models)[https://arxiv.org/abs/2505.06708].
            It applies a sigmoid-activated linear projection to the query
            which then gates the attention output. This helps improve training
            stability and eliminates "attention sinks".
        seed: Optional integer to seed the dropout layer.

    Call arguments:
        query: Query tensor of shape `(batch_dim, target_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `target_seq_len` is the length of
            target sequence, and `feature_dim` is dimension of feature.
        value: Value tensor of shape `(batch_dim, source_seq_len, feature_dim)`,
            where `batch_dim` is batch size, `source_seq_len` is the length of
            source sequence, and `feature_dim` is dimension of feature.
        key: Optional key tensor of shape
            `(batch_dim, source_seq_len, feature_dim)`. If not given, will use
            `value` for both `key` and `value`, which is most common case.
        attention_mask: A boolean mask of shape
            `(batch_dim, target_seq_len, source_seq_len)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, where 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        return_attention_scores: A boolean to indicate whether the output
            should be `(attention_output, attention_scores)` if `True`, or
            `attention_output` if `False`. Defaults to `False`.
        training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (no dropout).
            Will go with either using the training mode of the parent
            layer/model or `False` (inference) if there is no parent layer.
        use_causal_mask: A boolean to indicate whether to apply a causal mask to
            prevent tokens from attending to future tokens (e.g., used in a
            decoder Transformer).

    Returns:
        attention_output: Result of the computation, of shape
            `(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len`
            is for target sequence length and `feature_dim` is the query input
            last dim.
        attention_scores: (Optional) attention coefficients of shape
            `(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.
    """

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        dropout=0.0,
        use_bias=True,
        sliding_window=None,
        flash_attention=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_gate=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_query_heads` must be divisible by `num_key_value_heads`."
            )
        self.num_repeats = num_query_heads // num_key_value_heads
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_gate = use_gate
        if sliding_window is not None and (
            not isinstance(sliding_window, int) or sliding_window <= 0
        ):
            raise ValueError(
                "`sliding_window` must be `None` or a positive integer. "
                f"Received: sliding_window={sliding_window}"
            )
        self.sliding_window = sliding_window
        self._flash_attention = flash_attention or is_flash_attention_enabled()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.seed = seed

        self._inverse_sqrt_head_dim = 1.0 / math.sqrt(float(self.head_dim))
        self._return_attention_scores = False

        # Check for flash attention constraints
        if self._flash_attention and self.dropout > 0.0:
            raise ValueError(
                "Dropout is not supported when flash attention is enabled. "
                "Please set dropout to 0.0 to use flash attention."
            )

    def build(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        key_shape = value_shape if key_shape is None else key_shape
        self.feature_dim = query_shape[-1]
        self._query_dense = EinsumDense(
            "bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            bias_axes="uh" if self.use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)

        self._key_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)
        if self.use_gate:
            self._gate_dense = EinsumDense(
                "bqm,muh->bquh",
                output_shape=(None, self.num_query_heads, self.head_dim),
                bias_axes="uh" if self.use_bias else None,
                activation="sigmoid",
                name="gate",
                **self._get_common_kwargs_for_sublayer(),
            )
            self._gate_dense.build(query_shape)
        self._value_dense = EinsumDense(
            "bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh" if self.use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        self._softmax = Softmax(axis=-1, dtype=self.dtype_policy)
        self._dropout_layer = Dropout(
            rate=self.dropout, dtype=self.dtype_policy, seed=self.seed
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self._output_dense = EinsumDense(
            "bquh,uhm->bqm",
            output_shape=(None, self.feature_dim),
            bias_axes="m" if self.use_bias else None,
            name="attention_output",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            dtype=self.dtype_policy,
        )
        # Create new clone of kernel/bias initializer, so that we don't reuse
        # the initializer instance, which could lead to same init value since
        # initializer is stateless.
        kernel_initializer = self.kernel_initializer.__class__.from_config(
            self.kernel_initializer.get_config()
        )
        bias_initializer = self.bias_initializer.__class__.from_config(
            self.bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        self._return_attention_scores = return_attention_scores
        if key is None:
            key = value

        # When causal masking is the only mask source, route through the
        # backend's native causal kernel by passing `is_causal=True` rather
        # than materializing a [T, S] mask tensor. Skipped when a subclass
        # overrides `_compute_attention`, since such a subclass would
        # receive `attention_mask=None` and silently do unmasked attention.
        causal_only = (
            use_causal_mask
            and query_mask is None
            and value_mask is None
            and key_mask is None
            and attention_mask is None
            and self.sliding_window is None
            and type(self)._compute_attention
            is GroupedQueryAttention._compute_attention
        )
        if causal_only:
            attention_mask = None
        else:
            attention_mask = self._compute_attention_mask(
                query,
                value,
                query_mask=query_mask,
                value_mask=value_mask,
                key_mask=key_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )
        if self.use_gate:
            gate = self._gate_dense(query)
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)

        key = backend.ops.numpy.repeat(
            key, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)
        value = backend.ops.numpy.repeat(
            value, self.num_repeats, axis=2
        )  # (batch_dim, source_seq_len, query_heads, head_dim)

        extra_attention_kwargs = (
            {"use_causal_mask": True} if causal_only else {}
        )
        output, scores = self._compute_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            training=training,
            **extra_attention_kwargs,
        )
        # (batch_dim, target_seq_len, feature_dim)
        if self.use_gate:
            output = self._output_dense(
                backend.ops.numpy.multiply(output, gate)
            )
        else:
            output = self._output_dense(output)

        if return_attention_scores:
            return output, scores
        return output

    def _compute_attention_mask(
        self,
        query,
        value,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        use_causal_mask=False,
    ):
        """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query tensor of shape `(B, T, N, key_dim)`.
            key: Projected key tensor of shape `(B, T, N, key_dim)`.
            value: Projected value tensor of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
        auto_mask = None
        if query_mask is not None:
            # defensive casting
            query_mask = backend.ops.cast(query_mask, "bool")
            # B = batch size, T = max query length
            # shape is [B, T, 1]
            auto_mask = backend.ops.numpy.expand_dims(query_mask, -1)
        if value_mask is not None:
            # defensive casting
            value_mask = backend.ops.cast(value_mask, "bool")
            # B = batch size, S == max value length
            # shape is [B, 1, S]
            mask = backend.ops.numpy.expand_dims(value_mask, -2)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            # defensive casting
            key_mask = backend.ops.cast(key_mask, "bool")
            # B == batch size, S == max key length == max value length
            # shape is [B, 1, S]
            mask = backend.ops.numpy.expand_dims(key_mask, -2)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if self.sliding_window is not None:
            # the shape of the sliding window mask is [1, T, S]
            mask = self._compute_sliding_window_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else backend.ops.cast(attention_mask, "bool") & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean tensor equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query tensor of shape `(B, T, ...)`.
            value: value tensor of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean tensor of shape `(1, T, S)` containing a lower
                triangular matrix of shape `(T, S)`.
        """
        q_seq_length = backend.ops.shape(query)[1]
        v_seq_length = (
            q_seq_length if value is None else backend.ops.shape(value)[1]
        )
        ones_mask = backend.ops.numpy.ones(
            (1, q_seq_length, v_seq_length), dtype="int32"
        )
        row_index = backend.ops.numpy.cumsum(ones_mask, axis=-2)
        col_index = backend.ops.numpy.cumsum(ones_mask, axis=-1)
        return backend.ops.numpy.greater_equal(row_index, col_index)

    def _compute_sliding_window_mask(self, query, value=None):
        """Computes a banded sliding-window mask of shape `(1, T, S)`.

        Returns a symmetric band where each query position attends to keys
        within `self.sliding_window - 1` positions on either side. When
        ANDed with a causal mask, this yields the canonical causal
        sliding-window pattern used by Mistral, Llama-3 long-context, and
        Phi-3.
        """
        q_seq_length = backend.ops.shape(query)[1]
        v_seq_length = (
            q_seq_length if value is None else backend.ops.shape(value)[1]
        )
        row_index = backend.ops.numpy.reshape(
            backend.ops.numpy.arange(q_seq_length, dtype="int32"),
            (1, q_seq_length, 1),
        )
        col_index = backend.ops.numpy.reshape(
            backend.ops.numpy.arange(v_seq_length, dtype="int32"),
            (1, 1, v_seq_length),
        )
        return backend.ops.numpy.less(
            backend.ops.numpy.abs(row_index - col_index), self.sliding_window
        )

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        use_causal_mask=False,
    ):
        # Check for flash attention constraints
        if self._flash_attention and self._return_attention_scores:
            raise ValueError(
                "Returning attention scores is not supported when flash "
                "attention is enabled. Please disable flash attention to access"
                " attention scores."
            )

        # Determine whether to use dot-product attention
        use_dot_product_attention = not (
            self.dropout > 0.0
            or self._return_attention_scores
            or (len(query.shape) != 4)
        )

        if use_dot_product_attention:
            if use_causal_mask and attention_mask is None:
                # Skip materializing the [T, S] mask and let the backend
                # use its native causal kernel.
                attention_output = backend.ops.nn.dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    bias=None,
                    mask=None,
                    scale=self._inverse_sqrt_head_dim,
                    is_causal=True,
                    flash_attention=self._flash_attention,
                )
                return attention_output, None
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape for broadcasting
                # Expected shape: [batch_size, num_heads, query_seq_len,
                # key_seq_len].
                mask_expansion_axis = -1 * 2 - 1
                len_attention_scores_shape = 4  # Only accepts 4D inputs
                for _ in range(
                    len_attention_scores_shape - len(attention_mask.shape)
                ):
                    attention_mask = backend.ops.numpy.expand_dims(
                        attention_mask, axis=mask_expansion_axis
                    )
                attention_mask = backend.ops.cast(attention_mask, dtype="bool")
            # Directly compute the attention output using dot-product attention
            attention_output = backend.ops.nn.dot_product_attention(
                query=query,
                key=key,
                value=value,
                bias=None,
                mask=attention_mask,
                scale=self._inverse_sqrt_head_dim,
                is_causal=use_causal_mask,
                flash_attention=self._flash_attention,
            )
            return attention_output, None

        # Default behavior without flash attention, with explicit attention
        # scores. We skipped the fused is_causal kernel above, so build the
        # causal mask here. `call` drops attention_mask when causal is the
        # only mask source, expecting that kernel to run, and this path never
        # sees the is_causal flag otherwise. When an explicit mask is also
        # present, combine the two so neither constraint is lost.
        if use_causal_mask:
            causal_mask = self._compute_causal_mask(query, value)
            attention_mask = (
                causal_mask
                if attention_mask is None
                else backend.ops.numpy.logical_and(
                    backend.ops.cast(attention_mask, "bool"), causal_mask
                )
            )
        query = backend.ops.numpy.multiply(
            query, backend.ops.cast(self._inverse_sqrt_head_dim, query.dtype)
        )
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        scores = backend.ops.numpy.einsum(
            self._dot_product_equation, query, key
        )  # (batch_dim, query_heads, target_seq_len, source_seq_len)
        scores = self._masked_softmax(scores, attention_mask=attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout > 0.0:
            scores_dropout = self._dropout_layer(scores, training=training)
        else:
            scores_dropout = scores
        output = backend.ops.numpy.einsum(
            self._combine_equation, scores_dropout, value
        )
        return output, scores

    def _masked_softmax(self, scores, attention_mask=None):
        # Normalize the attention scores to probabilities.
        # scores = [B, N, T, S]
        if attention_mask is not None:
            # The expand dim happens starting from the `num_heads` dimension,
            # (<batch_dims>, num_heads, <query_attention_dims,
            # key_attention_dims>)
            mask_expansion_axis = -1 * 2 - 1
            for _ in range(len(scores.shape) - len(attention_mask.shape)):
                attention_mask = backend.ops.numpy.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(scores, mask=attention_mask)

    def compute_output_shape(
        self,
        query_shape,
        value_shape,
        key_shape=None,
    ):
        if key_shape is None:
            key_shape = value_shape

        if query_shape[-1] != value_shape[-1]:
            raise ValueError(
                "The last dimension of `query_shape` and `value_shape` "
                f"must be equal, but are {query_shape[-1]}, {value_shape[-1]}. "
                f"Received: query_shape={query_shape}, "
                f"value_shape={value_shape}"
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                "All dimensions of `value` and `key`, except the last one, "
                f"must be equal. Received: value_shape={value_shape} and "
                f"key_shape={key_shape}"
            )

        return query_shape

    def compute_output_spec(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        output_shape = self.compute_output_shape(
            query.shape,
            value.shape,
            key.shape if key is not None else None,
        )
        output_spec = backend.KerasTensor(
            output_shape, dtype=self.compute_dtype
        )
        if return_attention_scores:
            query_len = query.shape[1]
            kv_len = value.shape[1] if key is None else key.shape[1]
            scores_shape = (
                query.shape[0],
                self.num_query_heads,
                query_len,
                kv_len,
            )
            return output_spec, backend.KerasTensor(
                scores_shape, dtype=self.compute_dtype
            )
        return output_spec

    def get_config(self):
        config = {
            "head_dim": self.head_dim,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "use_bias": self.use_bias,
            "use_gate": self.use_gate,
            "sliding_window": self.sliding_window,
            "dropout": self.dropout,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(
                self.activity_regularizer
            ),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
