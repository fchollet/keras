from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer


# TODO(abheesht17): Move this to utils?
def _clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    Args:
        initializer: The initializer to clone.

    Returns:
        A cloned initializer if it is clonable, otherwise the original one.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    if isinstance(initializer, initializers.Initializer):
        config = initializer.get_config()
        return initializer.__class__.from_config(config)
    # If we get a string or dict, just return as we cannot and should not clone.
    return initializer


@keras_export("keras.layers.BatchNormalization")
class BatchNormalization(Layer):
    """Layer that normalizes its inputs.

    Batch normalization applies a transformation that maintains the mean output
    close to 0 and the output standard deviation close to 1.

    Importantly, batch normalization works differently during training and
    during inference.

    **During training** (i.e. when using `fit()` or when calling the layer/model
    with the argument `training=True`), the layer normalizes its output using
    the mean and standard deviation of the current batch of inputs. That is to
    say, for each channel being normalized, the layer returns
    `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

    - `epsilon` is small constant (configurable as part of the constructor
    arguments)
    - `gamma` is a learned scaling factor (initialized as 1), which
    can be disabled by passing `scale=False` to the constructor.
    - `beta` is a learned offset factor (initialized as 0), which
    can be disabled by passing `center=False` to the constructor.

    **During inference** (i.e. when using `evaluate()` or `predict()` or when
    calling the layer/model with the argument `training=False` (which is the
    default), the layer normalizes its output using a moving average of the
    mean and standard deviation of the batches it has seen during training. That
    is to say, it returns
    `gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta`.

    `self.moving_mean` and `self.moving_var` are non-trainable variables that
    are updated each time the layer in called in training mode, as such:

    - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
    - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

    As such, the layer will only normalize its inputs during inference
    *after having been trained on data that has similar statistics as the
    inference data*.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis). For instance, after a `Conv2D` layer
            with `data_format="channels_first"`, use `axis=1`.
        momentum: Momentum for the moving average.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If `True`, add offset of `beta` to normalized tensor.
            If `False`, `beta` is ignored.
        scale: If `True`, multiply by `gamma`. If `False`, `gamma` is not used.
            When the next layer is linear this can be disabled
            since the scaling will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        synchronized: Only applicable with the TensorFlow backend.
            If `True`, synchronizes the global batch statistics (mean and
            variance) for the layer across all devices at each training step
            in a distributed training strategy.
            If `False`, each replica uses its own local batch statistics.
        renorm: Whether to use
            [Batch Renormalization](https://arxiv.org/abs/1702.03275). This
            adds extra variables during training. The inference is the same
            for either value of this parameter.
        renorm_clipping: Dictionary, valid only if `renorm = True`.
            Maps optional keys `"rmax"`, `"rmin"`, `"dmax"` to floats used to
            clip the renorm correction. The correction `(r, d)` is used as
            `corrected_value = normalized_value * r + d`, with `r` clipped to
            `[rmin, rmax]`, and `d` to `[-dmax, dmax]`. Missing `rmax`, `rmin`,
            `dmax` are set to `inf`, `0`, `inf`, respectively.
        renorm_momentum: Momentum used to update the moving means and standard
            deviations with renorm. Valid only if `renorm= True`. Unlike
            `momentum`, this affects training and should be neither too small
            (which would add noise) nor too large (which would give stale
            estimates). Note that `momentum` is still applied to get the means
            and variances for inference.
        **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).

    Call arguments:
        inputs: Input tensor (of any rank).
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode.
            - `training=True`: The layer will normalize its inputs using
            the mean and variance of the current batch of inputs.
            - `training=False`: The layer will normalize its inputs using
            the mean and variance of its moving statistics, learned during
            training.
        mask: Binary tensor of shape broadcastable to `inputs` tensor, with
            `True` values indicating the positions for which mean and variance
            should be computed. Masked elements of the current inputs are not
            taken into account for mean and variance computation during
            training. Any prior unmasked element values will be taken into
            account until their momentum expires.

    Reference:

    - [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).

    **About setting `layer.trainable = False` on a `BatchNormalization` layer:**

    The meaning of setting `layer.trainable = False` is to freeze the layer,
    i.e. its internal state will not change during training:
    its trainable weights will not be updated
    during `fit()` or `train_on_batch()`, and its state updates will not be run.

    Usually, this does not necessarily mean that the layer is run in inference
    mode (which is normally controlled by the `training` argument that can
    be passed when calling a layer). "Frozen state" and "inference mode"
    are two separate concepts.

    However, in the case of the `BatchNormalization` layer, **setting
    `trainable = False` on the layer means that the layer will be
    subsequently run in inference mode** (meaning that it will use
    the moving mean and the moving variance to normalize the current batch,
    rather than using the mean and variance of the current batch).

    Note that:

    - Setting `trainable` on an model containing other layers will recursively
        set the `trainable` value of all inner layers.
    - If the value of the `trainable` attribute is changed after calling
        `compile()` on a model, the new value doesn't take effect for this model
        until `compile()` is called again.
    """

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        synchronized=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = int(axis)

        if synchronized and backend.backend() != "tensorflow":
            raise ValueError(
                "Argument synchronized=True is only supported "
                "with the TensorFlow backend."
            )
        self.synchronized = synchronized

        self.momentum = float(momentum)
        if not 0 <= self.momentum <= 1:
            raise ValueError(
                "Argument `momentum` must be a float in the range "
                f"[0, 1]. Received: momentum={momentum}"
            )
        self.epsilon = float(epsilon)
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = True

        self.renorm = renorm
        if renorm:
            renorm_clipping = renorm_clipping or {}
            keys = ["rmax", "rmin", "dmax"]
            if set(renorm_clipping) - set(keys):
                raise ValueError(
                    "Received invalid keys for `renorm_clipping` argument: "
                    f"{renorm_clipping}. Supported values: {keys}."
                )
            rmax = renorm_clipping.get("rmax")
            rmin = renorm_clipping.get("rmin")
            dmax = renorm_clipping.get("dmax")

            if rmax is not None and rmin is not None and rmax < rmin:
                raise ValueError(
                    "rmax should be greater than rmin in the `renorm_clipping` "
                    "argument. Received: rmax={rmax}, rmin={rmin}."
                )
            if dmax is not None and dmax < 0:
                raise ValueError(
                    "dmax should be non-negative in the `renorm_clipping` "
                    """argument. Received: dmax={dmax}."""
                )

        self.renorm_clipping = renorm_clipping
        self.renorm_momentum = renorm_momentum

        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None
        self._reduction_axes = None

    def build(self, input_shape):
        shape = (input_shape[self.axis],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                autocast=False,
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                autocast=False,
            )
        self.moving_mean = self.add_weight(
            shape=shape,
            name="moving_mean",
            initializer=self.moving_mean_initializer,
            trainable=False,
            autocast=False,
        )
        self.moving_variance = self.add_weight(
            shape=shape,
            name="moving_variance",
            initializer=self.moving_variance_initializer,
            trainable=False,
            autocast=False,
        )

        if self.renorm:
            # In batch renormalization we track the inference moving stddev
            # instead of the moving variance to more closely align with the
            # paper. The stddev is initialized as sqrt of the variance
            # initializer.
            def moving_stddev_initializer(shape, dtype=None):
                cloned = _clone_initializer(self.moving_variance_initializer)
                return ops.sqrt(cloned(shape, dtype=dtype))

            self.moving_stddev = self.add_weight(
                shape=shape,
                name="moving_stddev",
                initializer=moving_stddev_initializer,
                trainable=False,
                autocast=False,
            )
            # Create variables to maintain the moving mean and standard
            # deviation. These are used in training and thus are different
            # from the moving averages above.
            self.renorm_mean = self.add_weight(
                shape=shape,
                name="renorm_mean",
                initializer=_clone_initializer(self.moving_mean_initializer),
                trainable=False,
                autocast=False,
            )
            self.renorm_stddev = self.add_weight(
                shape=shape,
                name="renorm_stddev",
                initializer=moving_stddev_initializer,
                trainable=False,
                autocast=False,
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={self.axis: input_shape[self.axis]}
        )

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        # torch compilation will cast list to TrackedList and
        # force unnecessary graph break if used for reduction directly
        self._reduction_axes = tuple(reduction_axes)

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

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)

        moving_mean = ops.cast(self.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.moving_variance, inputs.dtype)

        if self.scale:
            gamma = ops.cast(self.gamma, inputs.dtype)
        else:
            gamma = None

        if self.center:
            beta = ops.cast(self.beta, inputs.dtype)
        else:
            beta = None

        if training and self.trainable:
            mean, variance = self._moments(inputs, mask)

            if self.renorm:
                # Compute renorm corrections (r and d).
                (
                    r,
                    d,
                    mean,
                    variance,
                ) = self._renorm_correction_and_moments(mean, variance)

                # x = x * gamma + beta without renorm, and
                # (x * r + d) * gamma + beta = x * (r * gamma) + (d * gamma +
                # beta) with renorm.
                gamma, beta = self._compose_transforms(
                    r, d, gamma, beta, inputs.dtype
                )

                # Update moving statistics.
                self._update_renorm_statistics(mean, variance)
            else:
                self.moving_mean.assign(
                    moving_mean * self.momentum + mean * (1.0 - self.momentum)
                )
                self.moving_variance.assign(
                    moving_variance * self.momentum
                    + variance * (1.0 - self.momentum)
                )
        else:
            mean = moving_mean
            variance = moving_variance

        outputs = ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        return ops.cast(outputs, self.compute_dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "moving_mean_initializer": initializers.serialize(
                self.moving_mean_initializer
            ),
            "moving_variance_initializer": initializers.serialize(
                self.moving_variance_initializer
            ),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "synchronized": self.synchronized,
            "renorm": self.renorm,
            "renorm_clipping": self.renorm_clipping,
            "renorm_momentum": self.renorm_momentum,
        }
        return {**base_config, **config}

    def _moments(self, inputs, mask):
        if mask is None:
            return ops.moments(
                inputs,
                axes=self._reduction_axes,
                synchronized=self.synchronized,
            )

        mask_broadcasted = ops.expand_dims(mask, axis=-1)
        mask_broadcasted = ops.broadcast_to(mask_broadcasted, ops.shape(inputs))
        mask_broadcasted = ops.cast(mask_broadcasted, "bool")
        masked_inputs = ops.where(mask_broadcasted, inputs, 0)

        masked_input_sum = ops.sum(
            masked_inputs,
            self._reduction_axes,
            keepdims=True,
        )
        sum_of_weights = ops.sum(
            ops.cast(mask_broadcasted, inputs.dtype),
            self._reduction_axes,
            keepdims=True,
        )
        mean = masked_input_sum / (sum_of_weights + backend.epsilon())
        # Explicit del to free large intermediates in eager mode (e.g. Torch).
        # No effect under jit/tf.function, but harmless.
        del masked_inputs, masked_input_sum

        difference = ops.where(mask_broadcasted, inputs - mean, 0)
        squared_difference = ops.square(difference)
        del difference
        weighted_distsq = ops.sum(
            squared_difference,
            self._reduction_axes,
            keepdims=True,
        )
        del squared_difference
        variance = weighted_distsq / (sum_of_weights + backend.epsilon())
        del weighted_distsq

        return ops.squeeze(mean), ops.squeeze(variance)

    def _renorm_correction_and_moments(self, mean, variance):
        """Computes the correction for batch renormalization.

        This method computes the r and d correction factors.

        Args:
            mean: The mean of the current batch.
            variance: The variance of the current batch.

        Returns:
            A tuple (r, s, mean, variance) where r and d are the correction
            factors, and mean/variance are passed through unchanged.
        """
        stddev = ops.sqrt(variance + self.epsilon)

        # Get the renorm moving statistics.
        renorm_mean = ops.cast(self.renorm_mean, mean.dtype)
        # Avoid divide by zero early on in training.
        renorm_stddev = ops.maximum(
            ops.cast(self.renorm_stddev, mean.dtype),
            ops.sqrt(ops.cast(self.epsilon, mean.dtype)),
        )

        # Compute the corrections for batch renorm.
        r = ops.divide(stddev, renorm_stddev)
        d = ops.divide(ops.subtract(mean, renorm_mean), renorm_stddev)

        # Apply clipping.
        rmin = self.renorm_clipping.get("rmin")
        rmax = self.renorm_clipping.get("rmax")
        dmax = self.renorm_clipping.get("dmax")

        if rmin is not None:
            r = ops.maximum(r, rmin)
        if rmax is not None:
            r = ops.minimum(r, rmax)
        if dmax is not None:
            d = ops.clip(d, -dmax, dmax)

        return r, d, mean, variance

    def _compose_transforms(self, r, d, gamma, beta, dtype):
        """Composes the renorm correction with gamma and beta.

        When training with renorm, the normalized values (x) are transformed
        as: (x * r + d) * gamma + beta = x * (r * gamma) + (d * gamma + beta).
        This method computes the effective scale and offset.

        Args:
            r: The r correction factor.
            d: The d correction factor.
            gamma: The gamma (scale) parameter, or None.
            beta: The beta (offset) parameter, or None.
            dtype: The dtype for the output.

        Returns:
            A tuple (effective_gamma, effective_beta).
        """
        r = ops.stop_gradient(r)
        d = ops.stop_gradient(d)

        if gamma is not None:
            effective_gamma = ops.multiply(r, gamma)
            effective_beta = ops.multiply(d, gamma)
        else:
            effective_gamma = ops.cast(r, dtype)
            effective_beta = ops.cast(d, dtype)

        if beta is not None:
            effective_beta = ops.add(effective_beta, beta)

        return effective_gamma, effective_beta

    def _update_renorm_statistics(self, mean, variance):
        """Updates the renorm and moving statistics.
        Args:
            mean: The mean of the current batch.
            variance: The variance of the current batch.
        """
        stddev = ops.sqrt(variance + self.epsilon)

        # Update renorm moving mean and stddev.
        renorm_mean = ops.cast(self.renorm_mean, mean.dtype)
        renorm_stddev = ops.cast(self.renorm_stddev, mean.dtype)

        self.renorm_mean.assign(
            renorm_mean * self.renorm_momentum
            + mean * (1.0 - self.renorm_momentum)
        )
        self.renorm_stddev.assign(
            renorm_stddev * self.renorm_momentum
            + stddev * (1.0 - self.renorm_momentum)
        )

        moving_mean = ops.cast(self.moving_mean, mean.dtype)
        moving_stddev = ops.cast(self.moving_stddev, mean.dtype)

        self.moving_mean.assign(
            moving_mean * self.momentum + mean * (1.0 - self.momentum)
        )

        new_moving_stddev = moving_stddev * self.momentum + stddev * (
            1.0 - self.momentum
        )
        self.moving_stddev.assign(new_moving_stddev)

        # Derive `moving_variance` from `moving_stddev`, applying ReLU in case
        # floating point rounding causes it to go negative.
        self.moving_variance.assign(
            ops.relu(new_moving_stddev * new_moving_stddev - self.epsilon)
        )
