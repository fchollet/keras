import dataclasses
import re
import string

import numpy as np

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.initializers.random_initializers import VarianceScaling
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.quantizers.geometry import EinsumProjectionGeometry
from keras.src.saving import serialization_lib


@keras_export("keras.layers.EinsumDense")
class EinsumDense(Layer):
    """A layer that uses `einsum` as the backing computation.

    This layer can perform einsum calculations of arbitrary dimensionality.

    Args:
        equation: An equation describing the einsum to perform.
            This equation must be a valid einsum string of the form
            `ab,bc->ac`, `...ab,bc->...ac`, or
            `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum
            axis expression sequence.
        output_shape: The expected shape of the output tensor
            (excluding the batch dimension and any dimensions
            represented by ellipses). You can specify `None` for any dimension
            that is unknown or can be inferred from the input shape.
        activation: Activation function to use. If you don't specify anything,
            no activation is applied
            (that is, a "linear" activation: `a(x) = x`).
        bias_axes: A string containing the output dimension(s)
            to apply a bias to. Each character in the `bias_axes` string
            should correspond to a character in the output portion
            of the `equation` string.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix.
        bias_constraint: Constraint function applied to the bias vector.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices
            (the factorization happens on the last dimension).
            This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing
            `EinsumDense` layer by calling `layer.enable_lora(rank)`.
         lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.
        dora_rank: Optional integer. If set, the layer's forward pass
            will implement DoRA (Weight-Decomposed Low-Rank Adaptation)
            with the provided rank. DoRA decomposes the layer's kernel
            into magnitude and direction components and applies LoRA
            to the direction component. This can improve the learning
            capacity of LoRA. You can also enable DoRA on an existing
            `EinsumDense` layer by calling `layer.enable_dora(rank)`.
        dora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta during the forward pass. The delta
            is scaled by `dora_alpha / dora_rank`, allowing you to fine-tune
            the strength of the DoRA adjustment independently of `dora_rank`.
        quantization_config: Optional configuration object for quantization.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Examples:

    **Biased dense layer with einsums**

    This example shows how to instantiate a standard Keras dense layer using
    einsum operations. This example is equivalent to
    `keras.layers.Dense(64, use_bias=True)`.

    >>> layer = keras.layers.EinsumDense("ab,bc->ac",
    ...                                       output_shape=64,
    ...                                       bias_axes="c")
    >>> input_tensor = keras.Input(shape=[32])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 64)

    **Applying a dense layer to a sequence**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence. Here, the `output_shape` has two
    values (since there are two non-batch dimensions in the output); the first
    dimension in the `output_shape` is `None`, because the sequence dimension
    `b` has an unknown shape.

    >>> layer = keras.layers.EinsumDense("abc,cd->abd",
    ...                                       output_shape=(None, 64),
    ...                                       bias_axes="d")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)

    **Applying a dense layer to a sequence using ellipses**

    This example shows how to instantiate a layer that applies the same dense
    operation to every element in a sequence, but uses the ellipsis notation
    instead of specifying the batch and sequence dimensions.

    Because we are using ellipsis notation and have specified only one axis, the
    `output_shape` arg is a single value. When instantiated in this way, the
    layer can handle any number of sequence dimensions - including the case
    where no sequence dimension exists.

    >>> layer = keras.layers.EinsumDense("...x,xy->...y",
    ...                                       output_shape=64,
    ...                                       bias_axes="y")
    >>> input_tensor = keras.Input(shape=[32, 128])
    >>> output_tensor = layer(input_tensor)
    >>> output_tensor.shape
    (None, 32, 64)
    """

    def __init__(
        self,
        equation,
        output_shape,
        activation=None,
        bias_axes=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        lora_alpha=None,
        dora_rank=None,
        dora_alpha=None,
        quantization_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.equation = equation
        if isinstance(output_shape, int):
            self.partial_output_shape = (output_shape,)
        else:
            self.partial_output_shape = tuple(output_shape)
        self.bias_axes = bias_axes
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.dora_rank = dora_rank
        self.dora_alpha = dora_alpha if dora_alpha is not None else dora_rank
        if self.lora_rank and self.dora_rank:
            raise ValueError(
                "Only one of `lora_rank` or `dora_rank` can be set."
            )
        self.lora_enabled = False
        self.dora_enabled = False
        self.quantization_config = quantization_config

    def _update_kernel_initializer(self, input_axes, output_axes):
        """
        Updates the kernel initializer with the correct input and output axes.
        """
        config = self.kernel_initializer.get_config()
        # RandomInitializer.get_config() returns the seed passed to the
        # constructor, which is often None. However, to maintain backwards
        # compatibility with the old behavior of having one initializer called
        # multiple times, we pass the concrete seed when cloning the
        # initializer.
        config["seed"] = self.kernel_initializer.seed
        config["input_axes"] = input_axes
        config["output_axes"] = output_axes
        return self.kernel_initializer.__class__.from_config(config)

    def build(self, input_shape):
        shape_data = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        kernel_shape, bias_shape, _, input_axes, output_axes = shape_data
        self.input_spec = InputSpec(ndim=len(input_shape))
        # The float kernel's N-D shape, whatever a quantization mode stores,
        # and the equation's axis bookkeeping the quantization modes read.
        self.kernel_shape = tuple(kernel_shape)
        self.einsum_axes = EinsumAxes.from_equation(
            self.equation, len(input_shape)
        )

        kernel_initializer = self.kernel_initializer
        if isinstance(self.kernel_initializer, VarianceScaling) and (
            self.kernel_initializer.input_axes is None
            or self.kernel_initializer.output_axes is None
        ):
            kernel_initializer = self._update_kernel_initializer(
                input_axes, output_axes
            )

        if self.quantization_mode is not None:
            # A strategy that owns the weight storage creates the kernel. A
            # strategy that keeps the float kernel (float8) adds its
            # variables after the float weights, as `quantize` does.
            if self._strategy_owns_weight_storage():
                self.quantized_build(
                    kernel_shape,
                    mode=self.quantization_mode,
                    config=self.quantization_config,
                )
        if not self._strategy_owns_weight_storage():
            self._kernel = self.add_weight(
                name="kernel",
                shape=tuple(kernel_shape),
                initializer=kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        if bias_shape is not None:
            self.bias = self.add_weight(
                name="bias",
                shape=tuple(bias_shape),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        if self.quantization_mode and not self._strategy_owns_weight_storage():
            self.quantized_build(
                kernel_shape,
                mode=self.quantization_mode,
                config=self.quantization_config,
            )
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank, lora_alpha=self.lora_alpha)
        if self.dora_rank:
            self.enable_dora(self.dora_rank, dora_alpha=self.dora_alpha)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        # A quantized layer exposes its integer codes in the kernel's shape;
        # without a `QuantizedWeight` view, the float kernel.
        quantized_weight = self._quantized_weight()
        kernel = (
            self._kernel
            if quantized_weight is None
            else quantized_weight.unpack()
        )

        # Apply LoRA if enabled
        if self.lora_enabled:
            kernel = ops.cast(
                ops.add(
                    kernel,
                    (self.lora_alpha / self.lora_rank)
                    * ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                ),
                dtype=self.compute_dtype,
            )
        elif self.dora_enabled:
            lora_delta = (self.dora_alpha / self.dora_rank) * ops.matmul(
                self.dora_kernel_a, self.dora_kernel_b
            )
            combined_kernel = ops.add(kernel, lora_delta)
            axis = tuple(range(len(kernel.shape) - 1))
            norm = ops.sqrt(ops.sum(ops.square(combined_kernel), axis=axis))
            kernel = self.dora_magnitude * ops.divide_no_nan(
                combined_kernel, norm
            )
            kernel = ops.cast(kernel, dtype=self.compute_dtype)

        return kernel

    def compute_output_shape(self, input_shape):
        _, _, full_output_shape, _, _ = _analyze_einsum_string(
            self.equation,
            self.bias_axes,
            input_shape,
            self.partial_output_shape,
        )
        return tuple(full_output_shape)

    def call(self, inputs, training=None):
        x = ops.einsum(self.equation, inputs, self.kernel)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def enable_lora(
        self,
        rank,
        lora_alpha=None,
        a_initializer="he_uniform",
        b_initializer="zeros",
    ):
        if self.kernel_constraint:
            raise ValueError(
                "Lora is incompatible with kernel constraints. "
                "In order to enable lora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable lora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError(
                "lora is already enabled. This can only be done once per layer."
            )
        if self.dora_enabled:
            raise ValueError("dora is already enabled. Cannot enable LoRA.")
        if self.quantization_mode == "gptq":
            raise NotImplementedError(
                "lora is not currently supported with GPTQ quantization."
            )
        self._check_lora_supported(self.quantization_mode)
        self._tracker.unlock()

        # LoRA weights should be float32 to avoid the risk of underflow or
        # overflow during fine-tuning.
        # When deploying the model, these weights should be merged with the
        # original kernel while maintaining the original kernel's dtype.
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=(self.kernel_shape[:-1] + (rank,)),
            initializer=initializers.get(a_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.kernel_shape[-1]),
            initializer=initializers.get(b_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank

    def enable_dora(
        self,
        rank,
        dora_alpha=None,
        a_initializer="he_uniform",
        b_initializer="zeros",
    ):
        if self.kernel_constraint:
            raise ValueError(
                "DoRA is incompatible with kernel constraints. "
                "In order to enable dora on this layer, remove the "
                "`kernel_constraint` argument."
            )
        if not self.built:
            raise ValueError(
                "Cannot enable dora on a layer that isn't yet built."
            )
        if self.lora_enabled:
            raise ValueError("LoRA is already enabled. Cannot enable DoRA.")
        if self.dora_enabled:
            raise ValueError(
                "dora is already enabled. This can only be done once per layer."
            )
        if self.quantization_mode is not None:
            raise NotImplementedError(
                "DoRA is not currently supported with quantized layers."
            )
        self._tracker.unlock()
        if self.quantization_mode == "int4":
            kernel_shape_for_lora = tuple(self.original_kernel_shape)
        else:
            kernel_shape_for_lora = self.kernel.shape

        self.dora_kernel_a = self.add_weight(
            name="dora_kernel_a",
            shape=(kernel_shape_for_lora[:-1] + (rank,)),
            initializer=initializers.get(a_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self.dora_kernel_b = self.add_weight(
            name="dora_kernel_b",
            shape=(rank, kernel_shape_for_lora[-1]),
            initializer=initializers.get(b_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )

        # Initialize Magnitude Vector
        # We reduce all axes except the last one
        axis = tuple(range(len(self.kernel.shape) - 1))
        initial_magnitude = ops.sqrt(
            ops.sum(ops.square(self.kernel), axis=axis)
        )
        self.dora_magnitude = self.add_weight(
            name="dora_magnitude",
            shape=(kernel_shape_for_lora[-1],),
            initializer=initializers.Constant(initial_magnitude),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )

        self._kernel.trainable = False
        self._tracker.lock()
        self.dora_enabled = True
        self.dora_rank = rank
        self.dora_alpha = dora_alpha if dora_alpha is not None else rank

    def save_own_variables(self, store):
        self._save_serialized_variables(store, "kernel")

    def load_own_variables(self, store):
        self._load_serialized_variables(
            store, "kernel", check_variable_count=not self.dora_enabled
        )
        if self.dora_enabled:
            self.dora_kernel_a.assign(ops.zeros(self.dora_kernel_a.shape))
            self.dora_kernel_b.assign(ops.zeros(self.dora_kernel_b.shape))
            axis = tuple(range(len(self._kernel.shape) - 1))
            self.dora_magnitude.assign(
                ops.sqrt(ops.sum(ops.square(self._kernel), axis=axis))
            )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "output_shape": self.partial_output_shape,
            "equation": self.equation,
            "activation": activations.serialize(self.activation),
            "bias_axes": self.bias_axes,
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
            "quantization_config": serialization_lib.serialize_keras_object(
                self.quantization_config
            ),
        }
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        if self.dora_rank:
            config["dora_rank"] = self.dora_rank
            config["dora_alpha"] = self.dora_alpha
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        # Written by earlier releases; the unpacked column count is read
        # from the stored group parameters now.
        config.pop("gptq_unpacked_column_size", None)
        config["quantization_config"] = (
            serialization_lib.deserialize_keras_object(
                config.get("quantization_config", None)
            )
        )
        return super().from_config(config)

    @property
    def variable_serialization_spec(self):
        """Returns a dict mapping quantization modes to variable names in order.

        This spec is used by `save_own_variables` and `load_own_variables` to
        determine the correct ordering of variables during serialization for
        each quantization mode. `None` means no quantization.
        """
        return {
            None: [
                "kernel",
                "bias",
            ],
            "int8": [
                "kernel",
                "bias",
                "kernel_scale",
            ],
            "int4": [
                "kernel",
                "bias",
                "kernel_scale",
                "kernel_zero",
                "g_idx",
            ],
            "float8": [
                "kernel",
                "bias",
                "inputs_scale",
                "inputs_amax_history",
                "kernel_scale",
                "kernel_amax_history",
                "outputs_grad_scale",
                "outputs_grad_amax_history",
            ],
            "gptq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "g_idx",
            ],
            "awq": [
                "bias",
                "quantized_kernel",
                "kernel_scale",
                "kernel_zero",
                "awq_scales",
                "g_idx",
            ],
        }

    def _quantization_geometry(self):
        return EinsumProjectionGeometry(self)

    def _get_kernel_scale_shape(self, kernel_shape):
        """Get the shape of the kernel scale tensor."""
        kernel_scale_shape = np.array(kernel_shape)
        kernel_scale_shape[self._kernel_reduced_axes] = 1
        kernel_scale_shape = kernel_scale_shape[self._kernel_transpose_axes]
        kernel_scale_shape = kernel_scale_shape.tolist()
        for axis in sorted(self._kernel_expand_axes):
            kernel_scale_shape.insert(axis, 1)
        for axis in sorted(self._kernel_squeeze_axes, reverse=True):
            kernel_scale_shape.pop(axis)
        return kernel_scale_shape

    def _adjust_scale_for_dequant(self, scale):
        """Adjusts scale tensor layout for dequantization.

        Helper method to handle scale adjustments before dequantization.
        This is the reverse order of operations used when building the layer.

        Args:
            scale: The scale tensor to adjust.

        Returns:
            The adjusted scale tensor.
        """
        if self._kernel_squeeze_axes:
            scale = ops.expand_dims(scale, axis=self._kernel_squeeze_axes)
        if self._kernel_expand_axes:
            scale = ops.squeeze(scale, axis=self._kernel_expand_axes)
        if self._kernel_transpose_axes:
            # We need to reverse the transpose operation.
            reverse_transpose = sorted(
                range(len(self._kernel_transpose_axes)),
                key=self._kernel_transpose_axes.__getitem__,
            )
            scale = ops.transpose(scale, axes=reverse_transpose)
        return scale

    def _adjust_scale_for_quant(self, scale, tensor_type="kernel"):
        """Adjusts scale tensor layout after quantization.

        Helper method to handle scale adjustments after re-quantization.
        This is the forward order of operations used when building the layer.

        Args:
            scale: The scale tensor to adjust.
            tensor_type: The type of tensor to adjust the scale for.
                "kernel" or "input".
        Returns:
            The adjusted scale tensor.
        """
        if tensor_type == "kernel":
            transpose_axes = self._kernel_transpose_axes
            expand_axes = self._kernel_expand_axes
            squeeze_axes = self._kernel_squeeze_axes
        elif tensor_type == "input":
            transpose_axes = self._input_transpose_axes
            expand_axes = self._input_expand_axes
            squeeze_axes = self._input_squeeze_axes
        else:
            raise ValueError(f"Invalid tensor type: {tensor_type}")

        if transpose_axes:
            scale = ops.transpose(scale, transpose_axes)
        if expand_axes:
            scale = ops.expand_dims(scale, axis=expand_axes)
        if squeeze_axes:
            scale = ops.squeeze(scale, axis=squeeze_axes)
        return scale

    def _set_quantization_info(self):
        if hasattr(self, "_input_reduced_axes"):
            # Already set.
            return
        (
            self._input_reduced_axes,
            self._kernel_reduced_axes,
            self._input_transpose_axes,
            self._kernel_transpose_axes,
            self._input_expand_axes,
            self._kernel_expand_axes,
            self._input_squeeze_axes,
            self._kernel_squeeze_axes,
            self._custom_gradient_equation,
            self._kernel_reverse_transpose_axes,
        ) = _analyze_quantization_info(
            self.equation, [None] * self.input_spec.ndim
        )


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
    """Parses an einsum string to determine the shapes of the weights.

    This function is the main entry point for analyzing the einsum equation.
    It handles equations with and without ellipses (`...`) by converting them
    to a standard format and then delegating to `_analyze_split_string` for
    the core logic.

    Args:
        equation: The einsum equation string, e.g., "ab,bc->ac" or
            "...ab,bc->...ac".
        bias_axes: A string indicating which output axes to apply a bias to.
        input_shape: The shape of the input tensor.
        output_shape: The user-specified shape of the output tensor (may be
            partial).

    Returns:
        A tuple `(
            kernel_shape, bias_shape, full_output_shape, input_axes, output_axes
        )` where:
            `kernel_shape`: The calculated shape of the einsum kernel.
            `bias_shape`: The calculated shape of the bias, or `None`.
            `full_output_shape`: The fully-resolved shape of the output tensor.
            `input_axes`: The axes representing input units.
            `output_axes`: The axes representing output units.

    Raises:
        ValueError: If the einsum `equation` is not in a supported format, or
            if the `ndim` of `input_shape` does not fit the input labels of
            the `equation`.
    """

    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

    # This is the case where no ellipses are present in the string.
    split_string = re.match(
        "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        _check_input_ndim(
            equation, input_shape, split_string.group(1), exact=True
        )
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    # This is the case where ellipses are present on the left.
    split_string = re.match(
        "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", dot_replaced_string
    )
    if split_string:
        _check_input_ndim(equation, input_shape, split_string.group(1))
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape, left_elided=True
        )

    # This is the case where ellipses are present on the right.
    split_string = re.match(
        "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", dot_replaced_string
    )
    if split_string:
        _check_input_ndim(equation, input_shape, split_string.group(1))
        return _analyze_split_string(
            split_string, bias_axes, input_shape, output_shape
        )

    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


# The accepted forms, with "..." replaced by "0", and the side each elides.
_EQUATION_FORMS = (
    ("([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", None),
    ("0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", "left"),
    ("([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", "right"),
)


def _split_equation(equation):
    """Splits an equation into its inputs, kernel and output labels.

    Returns:
        `(input_spec, weight_spec, output_spec, ellipsis)`, where
        `ellipsis` is the side the equation elides: `"left"`, `"right"`
        or `None`.
    """
    dot_replaced_string = re.sub(r"\.\.\.", "0", equation)
    for pattern, ellipsis in _EQUATION_FORMS:
        split_string = re.match(pattern, dot_replaced_string)
        if split_string:
            return (*split_string.groups(), ellipsis)
    raise ValueError(
        f"Invalid einsum equation '{equation}'. Equations must be in the form "
        "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
    )


def _check_input_ndim(equation, input_shape, input_spec, exact=False):
    """Checks that the input `ndim` can hold the labels of the input spec.

    An equation without an ellipsis labels every input axis, so `ndim` must
    match the number of labels. An equation with an ellipsis labels only some
    of the axes, so `ndim` must be at least the number of labels.

    Args:
        equation: The einsum equation string, used in the error message.
        input_shape: The shape of the input tensor.
        input_spec: The input labels of the equation, without the ellipsis.
        exact: A boolean indicating if `ndim` must equal the number of
            labels instead of being at least that number.

    Raises:
        ValueError: If the `ndim` of `input_shape` does not fit `input_spec`.
    """
    ndim = len(input_shape)
    num_labels = len(input_spec)
    if ndim == num_labels or (ndim > num_labels and not exact):
        return
    expected = f"ndim={num_labels}" if exact else f"min_ndim={num_labels}"
    raise ValueError(
        f"The einsum equation '{equation}' expects an input with {expected}, "
        f"but received input shape {tuple(input_shape)} with ndim={ndim}."
    )


def _analyze_split_string(
    split_string, bias_axes, input_shape, output_shape, left_elided=False
):
    """Computes kernel and bias shapes from a parsed einsum equation.

    This function takes the components of an einsum equation, validates them,
    and calculates the required shapes for the kernel and bias weights.

    Args:
        split_string: A regex match object containing the input, weight, and
            output specifications.
        bias_axes: A string indicating which output axes to apply a bias to.
        input_shape: The shape of the input tensor.
        output_shape: The user-specified partial shape of the output tensor.
        left_elided: A boolean indicating if the ellipsis "..." was on the
            left side of the equation.

    Returns:
        A tuple `(kernel_shape, bias_shape, full_output_shape)` where:
            `kernel_shape`: The calculated shape of the einsum kernel.
            `bias_shape`: The calculated shape of the bias, or `None`.
            `full_output_shape`: The fully-resolved shape of the output tensor.

    Raises:
        ValueError: If there are inconsistencies between the input and output
            shapes or if the equation specifications are invalid.
    """
    input_spec = split_string.group(1)
    weight_spec = split_string.group(2)
    output_spec = split_string.group(3)
    elided = len(input_shape) - len(input_spec)

    if isinstance(output_shape, int):
        output_shape = [output_shape]
    else:
        output_shape = list(output_shape)

    output_shape.insert(0, input_shape[0])

    if elided > 0 and left_elided:
        for i in range(1, elided):
            # We already inserted the 0th input dimension at dim 0, so we need
            # to start at location 1 here.
            output_shape.insert(1, input_shape[i])
    elif elided > 0 and not left_elided:
        for i in range(len(input_shape) - elided, len(input_shape)):
            output_shape.append(input_shape[i])

    if left_elided:
        # If we have beginning dimensions elided, we need to use negative
        # indexing to determine where in the input dimension our values are.
        input_dim_map = {
            dim: (i + elided) - len(input_shape)
            for i, dim in enumerate(input_spec)
        }
        # Because we've constructed the full output shape already, we don't need
        # to do negative indexing.
        output_dim_map = {
            dim: (i + elided) for i, dim in enumerate(output_spec)
        }
    else:
        input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
        output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

    for dim in input_spec:
        input_shape_at_dim = input_shape[input_dim_map[dim]]
        if dim in output_dim_map:
            output_shape_at_dim = output_shape[output_dim_map[dim]]
            if (
                output_shape_at_dim is not None
                and output_shape_at_dim != input_shape_at_dim
            ):
                raise ValueError(
                    "Input shape and output shape do not match at shared "
                    f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
                    "and output shape "
                    f"is {output_shape[output_dim_map[dim]]}."
                )

    for dim in output_spec:
        if dim not in input_spec and dim not in weight_spec:
            raise ValueError(
                f"Dimension '{dim}' was specified in the output "
                f"'{output_spec}' but has no corresponding dim in the input "
                f"spec '{input_spec}' or weight spec '{output_spec}'"
            )

    weight_shape = []
    input_axes, output_axes = [], []
    for i, dim in enumerate(weight_spec):
        if dim in output_dim_map:
            weight_shape.append(output_shape[output_dim_map[dim]])
            output_axes.append(i)
        elif dim in input_dim_map:
            weight_shape.append(input_shape[input_dim_map[dim]])
            input_axes.append(i)
        else:
            raise ValueError(
                f"Weight dimension '{dim}' did not have a match in either "
                f"the input spec '{input_spec}' or the output "
                f"spec '{output_spec}'. For this layer, the weight must "
                "be fully specified."
            )

    if bias_axes is not None:
        num_left_elided = elided if left_elided else 0
        idx_map = {
            char: output_shape[i + num_left_elided]
            for i, char in enumerate(output_spec)
        }

        for char in bias_axes:
            if char not in output_spec:
                raise ValueError(
                    f"Bias dimension '{char}' was requested, but is not part "
                    f"of the output spec '{output_spec}'"
                )

        first_bias_location = min(
            [output_spec.find(char) for char in bias_axes]
        )
        bias_output_spec = output_spec[first_bias_location:]

        bias_shape = [
            idx_map[char] if char in bias_axes else 1
            for char in bias_output_spec
        ]

        if not left_elided:
            for _ in range(elided):
                bias_shape.append(1)
    else:
        bias_shape = None

    return weight_shape, bias_shape, output_shape, input_axes, output_axes


@dataclasses.dataclass(frozen=True)
class EinsumAxes:
    """The labels of an `EinsumDense` equation, any ellipsis spelled out.

    Every axis fact the quantization modes read derives from the three
    label strings. An input or kernel axis is reduced (contracted) when
    its label is not in the output.
    """

    inputs: str
    kernel: str
    output: str

    @classmethod
    def from_equation(cls, equation, input_rank):
        inputs, kernel, output, ellipsis = _split_equation(equation)
        if ellipsis is not None:
            # Give the elided axes labels of their own.
            unused = sorted(
                set(string.ascii_letters) - set(inputs + kernel + output)
            )
            elided = "".join(unused[: input_rank - len(inputs)])
            if ellipsis == "left":
                inputs, output = elided + inputs, elided + output
            else:
                inputs, output = inputs + elided, output + elided
        return cls(inputs, kernel, output)

    @property
    def input_reduced_axes(self):
        """The input axes the equation contracts."""
        return tuple(
            i for i, label in enumerate(self.inputs) if label not in self.output
        )

    @property
    def gradient_equation(self):
        """The einsum that produces the inputs gradient."""
        return f"{self.output},{self.kernel}->{self.inputs}"


def _analyze_quantization_info(equation, input_shape):
    """Analyzes an einsum equation to derive information for quantization.

    This function canonicalizes the einsum equation (handling ellipses) and
    determines the necessary tensor manipulations (reduction, transposition,
    expansion, squeezing) required to correctly apply per-axis quantization
    to the inputs and kernel. It also derives the einsum equation needed for
    the custom gradient.

    Args:
        equation: The einsum equation string.
        input_shape: The shape of the input tensor.

    Returns:
        A tuple containing metadata for quantization operations:
        `input_reduced_axes`: Axes to reduce for input quantization.
        `kernel_reduced_axes`: Axes to reduce for kernel quantization.
        `input_transpose_axes`: Permutation for transposing the input scale.
        `kernel_transpose_axes`: Permutation for transposing the kernel scale.
        `input_expand_axes`: Axes to expand for the input scale.
        `kernel_expand_axes`: Axes to expand for the kernel scale.
        `input_squeeze_axes`: Axes to squeeze from the input scale.
        `kernel_squeeze_axes`: Axes to squeeze from the kernel scale.
        `custom_gradient_equation`: Einsum equation for the backward pass.
        `kernel_reverse_transpose_axes`: Permutation to reverse the kernel
            scale transpose.
    """

    axes = EinsumAxes.from_equation(equation, len(input_shape))
    input_spec, weight_spec, output_spec = axes.inputs, axes.kernel, axes.output

    # Determine the axes that should be reduced by the quantizer
    input_reduced_axes = []
    weight_reduced_axes = []
    for i, label in enumerate(input_spec):
        index = output_spec.find(label)
        if index == -1:
            input_reduced_axes.append(i)
    for i, label in enumerate(weight_spec):
        index = output_spec.find(label)
        if index == -1:
            weight_reduced_axes.append(i)

    # Determine the axes of `ops.expand_dims`
    input_expand_axes = []
    weight_expand_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input == -1:
            input_expand_axes.append(i)
        if index_weight == -1:
            weight_expand_axes.append(i)

    # Determine the axes of `ops.transpose`
    input_transpose_axes = []
    weight_transpose_axes = []
    for i, label in enumerate(output_spec):
        index_input = input_spec.find(label)
        index_weight = weight_spec.find(label)
        if index_input != -1:
            input_transpose_axes.append(index_input)
        if index_weight != -1:
            weight_transpose_axes.append(index_weight)
    # Postprocess the information:
    # 1. Add dummy axes (1) to transpose_axes
    # 2. Add axis to squeeze_axes if 1. failed. The axis then stays at its
    #    own position, so the squeeze removes that size-1 axis.
    input_squeeze_axes = []
    weight_squeeze_axes = []
    for ori_index in input_reduced_axes:
        try:
            index = input_expand_axes.pop(0)
        except IndexError:
            input_squeeze_axes.append(ori_index)
            index = ori_index
        input_transpose_axes.insert(index, ori_index)
    for ori_index in weight_reduced_axes:
        try:
            index = weight_expand_axes.pop(0)
        except IndexError:
            weight_squeeze_axes.append(ori_index)
            index = ori_index
        weight_transpose_axes.insert(index, ori_index)
    # Prepare equation for `einsum_with_inputs_gradient`
    custom_gradient_equation = f"{output_spec},{weight_spec}->{input_spec}"
    weight_reverse_transpose_axes = [
        i
        for (_, i) in sorted(
            (v, i) for (i, v) in enumerate(weight_transpose_axes)
        )
    ]
    return (
        input_reduced_axes,
        weight_reduced_axes,
        input_transpose_axes,
        weight_transpose_axes,
        input_expand_axes,
        weight_expand_axes,
        input_squeeze_axes,
        weight_squeeze_axes,
        custom_gradient_equation,
        weight_reverse_transpose_axes,
    )
