from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


@keras_export("keras.layers.GroupNormalization")
class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes nearly
    identical to Layer Normalization (see Layer Normalization docs for details).

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is
    equal to number of channels), then this operation becomes identical to
    Instance Normalization. You can achieve this via `groups=-1`.

    Args:
        groups: Integer, the number of groups for Group Normalization. Can be in
            the range `[1, N]` where N is the input dimension. The input
            dimension must be divisible by the number of groups.
            Defaults to 32.
        axis: Integer or List/Tuple. The axis or axes to normalize across.
            Typically, this is the features axis/axes. The left-out axes are
            typically the batch axis/axes. -1 is the last dimension in the
            input. Defaults to `-1`.
        epsilon: Small float added to variance to avoid dividing by zero.
            Defaults to 1e-3.
        center: If `True`, add offset of `beta` to normalized tensor.
            If `False`, `beta` is ignored. Defaults to `True`.
        scale: If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
            When the next layer is linear (also e.g. `relu`), this can be
            disabled since the scaling will be done by the next layer.
            Defaults to `True`.
        beta_initializer: Initializer for the beta weight. Defaults to zeros.
        gamma_initializer: Initializer for the gamma weight. Defaults to ones.
        beta_regularizer: Optional regularizer for the beta weight. None by
            default.
        gamma_regularizer: Optional regularizer for the gamma weight. None by
            default.
        beta_constraint: Optional constraint for the beta weight.
            None by default.
        gamma_constraint: Optional constraint for the gamma weight. None by
            default.  Input shape: Arbitrary. Use the keyword argument
            `input_shape` (tuple of integers, does not include the samples
            axis) when using this layer as the first layer in a model.
            Output shape: Same shape as input.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Reference:

    - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups=32,
        axis=-1,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        if not isinstance(groups, int) or (groups <= 0 and groups != -1):
            raise ValueError(
                "Received an invalid value for argument `groups`, expected "
                "a positive integer (or `-1` for instance normalization). "
                f"Received: groups={groups}"
            )
        if epsilon is None or epsilon <= 0:
            raise ValueError(
                "Argument `epsilon` must be positive. "
                f"Received: epsilon={epsilon}"
            )
        super().__init__(**kwargs)
        self.supports_masking = True
        self.autocast = False
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError(
                f"Axis {self.axis} of input tensor should have a defined "
                "dimension but the layer received an input with shape "
                f"{input_shape}."
            )

        if self.groups == -1:
            self.groups = dim

        if dim < self.groups:
            raise ValueError(
                f"Number of groups ({self.groups}) cannot be more than the "
                f"number of channels ({dim})."
            )

        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must be a multiple "
                f"of the number of channels ({dim})."
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

        if self.scale:
            self.gamma = self.add_weight(
                shape=(dim,),
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                shape=(dim,),
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                autocast=False,
            )
        else:
            self.beta = None

        super().build(input_shape)

    def call(self, inputs):
        reshaped_inputs = self._reshape_into_groups(inputs)
        normalized_inputs = self._apply_normalization(
            reshaped_inputs, inputs.shape
        )
        outputs = ops.reshape(normalized_inputs, ops.shape(inputs))
        return ops.cast(outputs, self.compute_dtype)

    def _reshape_into_groups(self, inputs):
        input_shape = ops.shape(inputs)
        group_shape = list(inputs.shape)
        group_shape[0] = -1
        for i, e in enumerate(group_shape[1:]):
            if e is None:
                group_shape[i + 1] = input_shape[i + 1]

        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        reshaped_inputs = ops.reshape(inputs, group_shape)
        return reshaped_inputs

    def _apply_normalization(self, reshaped_inputs, input_shape):
        inputs_dtype = reshaped_inputs.dtype
        compute_dtype = backend.result_type(inputs_dtype, "float32")
        # GN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        reshaped_inputs = ops.cast(reshaped_inputs, compute_dtype)

        group_reduction_axes = list(range(1, len(reshaped_inputs.shape)))

        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        broadcast_shape = self._create_broadcast_shape(input_shape)
        mean, variance = ops.moments(
            reshaped_inputs, axes=group_reduction_axes, keepdims=True
        )

        # Compute the batch normalization.
        inv = ops.rsqrt(variance + self.epsilon)
        if self.scale:
            gamma = ops.reshape(self.gamma, broadcast_shape)
            gamma = ops.cast(gamma, reshaped_inputs.dtype)
            inv = inv * gamma

        normalized_inputs = (reshaped_inputs - mean) * inv
        if self.center:
            beta = ops.reshape(self.beta, broadcast_shape)
            beta = ops.cast(beta, reshaped_inputs.dtype)
            normalized_inputs = normalized_inputs + beta

        normalized_inputs = ops.cast(normalized_inputs, inputs_dtype)

        return normalized_inputs

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

    def compute_output_shape(self, input_shape):
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = self.axis

        for axis in axes:
            if axis >= len(input_shape) or axis < -len(input_shape):
                raise ValueError(
                    f"Axis {axis} is out of bounds for "
                    f"input shape {input_shape}. "
                    f"Received: axis={self.axis}"
                )
        return input_shape

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
