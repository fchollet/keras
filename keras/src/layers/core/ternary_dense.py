from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.quantizers.geometry import ProjectionGeometry
from keras.src.quantizers.quantizers import ternarize


@keras_export("keras.layers.TernaryDense")
class TernaryDense(Layer):
    """Densely-connected layer with Straight-Through Estimator ternary training.

    Trains a dense layer whose weights are constrained to `{-1, 0, +1}` via a
    Straight-Through Estimator (STE). On every forward pass the float kernel is
    ternarized before the matrix multiply; gradients flow back through the
    quantization step as if it were the identity, keeping the underlying float
    weights trainable.

    Quantization rule (BitNet b1.58, https://arxiv.org/abs/2402.17764 §3.1):

    ```
    kernel_ternary[i,j] = sign(kernel[i,j])  if |kernel[i,j]| > threshold
                          0                   otherwise
    output = beta * matmul(inputs, kernel_ternary) + bias
    ```

    where `threshold` defaults to `0.5 * mean(|kernel|)` and `beta =
    mean(|kernel|)` per forward pass (both per the BitNet b1.58 §3.1 spec).

    For post-training quantization of a regular `Dense` layer (no STE), use
    `keras.layers.Dense` with `layer.quantize("ternary")` instead. That path
    packs the trained float kernel to `~1.58 bits/weight` for checkpoint
    storage.

    Call `layer.quantize("ternary")` on a `TernaryDense` after training to
    freeze the ternarized kernel into a packed representation. The packed
    format stores `{-1, 0, +1}` weights at the information-theoretic floor of
    `log2(3) ~= 1.58` bits/value: five trits per byte
    (`3 ** 5 == 243 <= 256`), denser than int4 or int8.

    **Checkpoint size.** `quantize("ternary")` stores weights at ~1.58
    bits/value (~5× smaller than float32, ~2.5× smaller than int4).

    **Inference note.** `quantize("ternary")` is a storage format, not a
    compute win: the packed kernel is unpacked to a full `{-1, 0, +1}` matrix
    on every forward pass and fed to a standard matmul, so inference is
    slightly slower than a plain `Dense` call. A native ternary kernel that
    reads the packed format directly would be needed for a speedup.

    Args:
        units: Positive integer, dimensionality of the output space.
        threshold: Non-negative float or `None`. Quantization boundary applied
            element-wise to `|kernel|`. `None` (default) uses
            `0.5 * mean(|kernel|)` per forward pass, matching the BitNet b1.58
            §3.1 rule. `0.0` maps every weight to ±1 (no zeros). Large values
            (e.g. `1e9`) yield an all-zero effective kernel.
        activation: Activation function to use. Defaults to no activation
            (linear: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
            Defaults to `True`.
        kernel_initializer: Initializer for the `kernel` weights matrix.
            Defaults to `"glorot_uniform"`.
        bias_initializer: Initializer for the bias vector.
            Defaults to `"zeros"`.
        kernel_regularizer: Regularizer function applied to the `kernel`
            weights matrix. Defaults to `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Defaults to `None`.
        activity_regularizer: Regularizer function applied to the output of
            the layer. Defaults to `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Defaults to `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Defaults to `None`.

    Input shape:
        N-D tensor with shape `(batch_size, ..., input_dim)`.

    Output shape:
        N-D tensor with shape `(batch_size, ..., units)`.

    Example:

    >>> layer = keras.layers.TernaryDense(64)
    >>> inputs = np.random.rand(4, 32).astype("float32")
    >>> outputs = layer(inputs)
    >>> outputs.shape
    (4, 64)

    Quantize for export and inference (kernel stored at ~1.6 bits/value):

    >>> layer = keras.layers.TernaryDense(64)
    >>> _ = layer(np.random.rand(4, 32).astype("float32"))
    >>> layer.quantize("ternary")
    >>> outputs = layer(np.random.rand(4, 32).astype("float32"))
    >>> outputs.shape
    (4, 64)
    """

    def __init__(
        self,
        units,
        threshold=None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        if not isinstance(units, int) or units <= 0:
            raise ValueError(
                "Received an invalid value for `units`, expected a positive "
                f"integer. Received: units={units}"
            )
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError(
                "Received an invalid value for `threshold`, expected a float "
                f"or None. Received: threshold={threshold}"
            )
        if threshold is not None and threshold < 0:
            raise ValueError(
                "Received an invalid value for `threshold`, expected a "
                "non-negative float or None. "
                f"Received: threshold={threshold}"
            )

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.threshold = threshold
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = (input_dim, self.units)
        # The float kernel's shape, whatever a quantization mode stores.
        self.kernel_shape = kernel_shape
        if self.quantization_mode:
            # Packed kernel + scale are created here; the float `kernel` is not.
            self.quantized_build(kernel_shape, mode=self.quantization_mode)
        else:
            self._kernel = self.add_weight(
                name="kernel",
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                autocast=False,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        # Frozen: the codes `{-1, 0, +1}`; otherwise the float kernel.
        quantized_weight = self._quantized_weight()
        return (
            self._kernel
            if quantized_weight is None
            else quantized_weight.unpack()
        )

    def _ternary_kernel(self):
        """The straight-through kernel and `beta = mean(|kernel|)`.

        The kernel's forward value is in {-1, 0, +1} and gradients flow via
        STE. The trits and `beta` are decided on the stored kernel in
        float32, whatever the autocast scope, like `quantizers.ternarize`
        in `quantize()`.
        """
        # Not `self._kernel._value`: the JAX trainer detaches it during `fit`,
        # `evaluate` and `predict` and feeds the weights through a stateless
        # scope, which only `.value` reads. The kernel is created with
        # `autocast=False`, so `.value` is its stored value in every scope.
        stored = self._kernel.value
        abs_k = ops.abs(ops.cast(stored, "float32"))
        beta = ops.mean(abs_k)
        t = 0.5 * beta if self.threshold is None else self.threshold
        k_ternary = ops.sign(stored) * ops.cast(
            ops.greater(abs_k, t), dtype=stored.dtype
        )
        # The straight-through sum, and the matmul after it, run in the
        # compute dtype, as they do for an autocast kernel.
        kernel = ops.cast(stored, self.compute_dtype)
        k_ternary = ops.cast(k_ternary, self.compute_dtype)
        return (
            ops.add(kernel, ops.stop_gradient(ops.subtract(k_ternary, kernel))),
            beta,
        )

    def call(self, inputs):
        k, beta = self._ternary_kernel()
        x = ops.matmul(inputs, k)
        if self.threshold is None:
            x = ops.multiply(x, ops.cast(beta, x.dtype))
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    # Quantization: real ternary export + inference.
    #
    # During training the float `kernel` is kept and ternarized on the fly
    # (the `call` path above). `quantize("ternary")` freezes that result into a
    # packed representation: the `{-1, 0, +1}` kernel is stored at the
    # information-theoretic floor of ~1.6 bits/value (five trits per byte,
    # `3 ** 5 == 243 <= 256`), denser than int4 or int8 can express. Inference
    # then runs from the packed kernel via the ternary mode's forward pass.
    # The layer contributes only its geometry; `Layer.quantize` does the
    # rest.

    def _quantization_geometry(self):
        return _TernaryDenseGeometry(self)

    @property
    def variable_serialization_spec(self):
        """Maps each quantization mode to its ordered variable names."""
        return {
            None: ["kernel", "bias"],
            "ternary": ["kernel", "bias", "kernel_scale"],
        }

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(store, "kernel")

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
            "threshold": self.threshold,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
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
        }
        return {**base_config, **config}


class _TernaryDenseGeometry(ProjectionGeometry):
    """TernaryDense's quantization geometry.

    The ternarization rule is the layer's own: the straight-through-estimator
    kernel with its optional fixed threshold, so freezing it does not change
    the layer's outputs.
    """

    def ternary_values(self):
        layer = self.layer
        # The rule of the straight-through kernel on the stored kernel, as
        # in `_ternary_kernel`, so the frozen codes are exactly its forward
        # value; the scale is `beta`, or 1.0 with a fixed threshold.
        # `quantize()` runs eagerly, outside any stateless scope, so `_value`
        # is the live kernel in its own dtype.
        return ternarize(layer._kernel._value, layer.threshold)
