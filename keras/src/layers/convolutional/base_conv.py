"""Keras base class for convolution layers."""

from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.backend import standardize_data_format
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.ops.operation_utils import compute_conv_output_shape
from keras.src.utils.argument_validation import standardize_padding
from keras.src.utils.argument_validation import standardize_tuple


class BaseConv(Layer):
    """Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved (actually
    cross-correlated) with the layer input to produce a tensor of outputs. If
    `use_bias` is True (and a `bias_initializer` is provided), a bias vector is
    created and added to the outputs. Finally, if `activation` is not `None`, it
    is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Args:
        rank: int, the rank of the convolution, e.g. 2 for 2D convolution.
        filters: int, the dimension of the output space (the number of filters
            in the convolution).
        kernel_size: int or tuple/list of `rank` integers, specifying the size
            of the convolution window.
        strides: int or tuple/list of `rank` integers, specifying the stride
            length of the convolution. If only one int is specified, the same
            stride size will be used for all dimensions. `strides > 1` is
            incompatible with `dilation_rate > 1`.
        padding: string, either `"valid"` or `"same"` (case-insensitive).
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input. When `padding="same"` and
            `strides=1`, the output has the same size as the input.
        data_format: string, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, steps, features)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, features, steps)`. It defaults to the `image_data_format`
            value found in your Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
        dilation_rate: int or tuple/list of `rank` integers, specifying the
            dilation rate to use for dilated convolution. If only one int is
            specified, the same dilation rate will be used for all dimensions.
        groups: A positive int specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters // groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function. If `None`, no activation is applied.
        use_bias: bool, if `True`, bias will be added to the output.
        kernel_initializer: Initializer for the convolution kernel. If `None`,
            the default initializer (`"glorot_uniform"`) will be used.
        bias_initializer: Initializer for the bias vector. If `None`, the
            default initializer (`"zeros"`) will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The
            function must take as input the unprojected variable and must return
            the projected variable (which must have the same shape). Constraints
            are not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        lora_rank: Optional integer. If set, the layer's forward pass
            will implement LoRA (Low-Rank Adaptation)
            with the provided rank. LoRA sets the layer's kernel
            to non-trainable and replaces it with a delta over the
            original kernel, obtained via multiplying two lower-rank
            trainable matrices. This can be useful to reduce the
            computation cost of fine-tuning large dense layers.
            You can also enable LoRA on an existing layer by calling
            `layer.enable_lora(rank)`.
        lora_alpha: Optional integer. If set, this parameter scales the
            low-rank adaptation delta (computed as the product of two lower-rank
            trainable matrices) during the forward pass. The delta is scaled by
            `lora_alpha / lora_rank`, allowing you to fine-tune the strength of
            the LoRA adjustment independently of `lora_rank`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank=None,
        lora_alpha=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.rank = rank
        self.filters = filters
        self.groups = groups
        self.kernel_size = standardize_tuple(kernel_size, rank, "kernel_size")
        self.strides = standardize_tuple(strides, rank, "strides")
        self.dilation_rate = standardize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.padding = standardize_padding(padding, allow_causal=rank == 1)
        self.data_format = standardize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.lora_enabled = False
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if self.groups <= 0:
            raise ValueError(
                "The number of groups must be a positive integer. "
                f"Received: groups={self.groups}."
            )

        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                f"number of groups. Received: groups={self.groups}, "
                f"filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0. Received "
                f"kernel_size={self.kernel_size}."
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0. Received "
                f"strides={self.strides}"
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel_axis = -1
            input_channel = input_shape[-1]
        else:
            channel_axis = 1
            input_channel = input_shape[1]
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        if input_channel % self.groups != 0:
            raise ValueError(
                "The number of input channels must be evenly divisible by "
                f"the number of groups. Received groups={self.groups}, but the "
                f"input has {input_channel} channels (full input shape is "
                f"{input_shape})."
            )
        kernel_shape = self.kernel_size + (
            input_channel // self.groups,
            self.filters,
        )

        # compute_output_shape contains some validation logic for the input
        # shape, and make sure the output shape has all positive dimensions.
        self.compute_output_shape(input_shape)

        self._kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None
        self.built = True
        if self.lora_rank:
            self.enable_lora(self.lora_rank, lora_alpha=self.lora_alpha)

    @property
    def kernel(self):
        if not self.built:
            raise AttributeError(
                "You must build the layer before accessing `kernel`."
            )
        kernel = self._kernel
        if self.lora_enabled:
            kernel = ops.cast(
                ops.add(
                    kernel,
                    (self.lora_alpha / self.lora_rank)
                    * ops.matmul(self.lora_kernel_a, self.lora_kernel_b),
                ),
                dtype=self.compute_dtype,
            )
        return kernel

    def convolution_op(self, inputs, kernel):
        return ops.conv(
            inputs,
            kernel,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

    def call(self, inputs):
        outputs = self.convolution_op(inputs, self.kernel)
        if self.use_bias:
            if self.data_format == "channels_last":
                bias_shape = (1,) * (self.rank + 1) + (self.filters,)
            else:
                bias_shape = (1, self.filters) + (1,) * self.rank
            bias = ops.reshape(self.bias, bias_shape)
            outputs = ops.add(outputs, bias)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return compute_conv_output_shape(
            input_shape,
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

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
        self._tracker.unlock()

        # LoRA weights should be float32 to avoid the risk of underflow or
        # overflow during fine-tuning.
        # When deploying the model, these weights should be merged with the
        # original kernel while maintaining the original kernel's dtype.
        self.lora_kernel_a = self.add_weight(
            name="lora_kernel_a",
            shape=self._kernel.shape[:-1] + (rank,),
            initializer=initializers.get(a_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self.lora_kernel_b = self.add_weight(
            name="lora_kernel_b",
            shape=(rank, self.filters),
            initializer=initializers.get(b_initializer),
            dtype="float32",
            regularizer=self.kernel_regularizer,
        )
        self._kernel.trainable = False
        self._tracker.lock()
        self.lora_enabled = True
        self.lora_rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else rank

    def save_own_variables(self, store):
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self.kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            store[str(i)] = variable

    def load_own_variables(self, store):
        self._check_load_own_variables(store)
        # Do nothing if the layer isn't yet built
        if not self.built:
            return
        target_variables = [self._kernel]
        if self.use_bias:
            target_variables.append(self.bias)
        for i, variable in enumerate(target_variables):
            variable.assign(store[str(i)])
        if self.lora_enabled:
            self.lora_kernel_a.assign(ops.zeros(self.lora_kernel_a.shape))
            self.lora_kernel_b.assign(ops.zeros(self.lora_kernel_b.shape))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        if self.lora_rank:
            config["lora_rank"] = self.lora_rank
            config["lora_alpha"] = self.lora_alpha
        return config

    def _check_load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        if self.lora_enabled:
            # A save merges the LoRA update into the kernel, so the store
            # holds no LoRA factors.
            factors = {id(self.lora_kernel_a), id(self.lora_kernel_b)}
            all_vars = [v for v in all_vars if id(v) not in factors]
        if len(store.keys()) != len(all_vars):
            if len(all_vars) == 0 and not self.built:
                raise ValueError(
                    f"Layer '{self.name}' was never built "
                    "and thus it doesn't have any variables. "
                    f"However the weights file lists {len(store.keys())} "
                    "variables for this layer.\n"
                    "In most cases, this error indicates that either:\n\n"
                    "1. The layer is owned by a parent layer that "
                    "implements a `build()` method, but calling the "
                    "parent's `build()` method did NOT create the state of "
                    f"the child layer '{self.name}'. A `build()` method "
                    "must create ALL state for the layer, including "
                    "the state of any children layers.\n\n"
                    "2. You need to implement "
                    "the `def build_from_config(self, config)` method "
                    f"on layer '{self.name}', to specify how to rebuild "
                    "it during loading. "
                    "In this case, you might also want to implement the "
                    "method that generates the build config at saving time, "
                    "`def get_build_config(self)`. "
                    "The method `build_from_config()` is meant "
                    "to create the state "
                    "of the layer (i.e. its variables) upon deserialization.",
                )
            raise ValueError(
                f"Layer '{self.name}' expected {len(all_vars)} variables, "
                "but received "
                f"{len(store.keys())} variables during loading. "
                f"Expected: {[v.name for v in all_vars]}"
            )
