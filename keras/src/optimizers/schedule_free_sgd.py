from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer


@keras_export(["keras.optimizers.ScheduleFreeSGD"])
class ScheduleFreeSGD(optimizer.Optimizer):
    """Optimizer that implements the Schedule-Free SGD algorithm.

    Schedule-Free learning is a method that avoids the need for a learning rate
    schedule by maintaining a combination of interpolation and averaging.
    This approach eliminates the requirement to specify stopping time in advance
    and typically matches or outperforms cosine and linear decay schedules.

    The optimizer maintains two sets of auxiliary variables internally:
    - `momentum`: The sequence (z) where gradient updates are applied
    - `averaged`: The averaged sequence (x) used for evaluation

    During training, the model parameters (y) are set to an interpolation
    between `momentum` (z) and `averaged` (x).

    Args:
        learning_rate: A float, a
            `keras.optimizers.schedules.LearningRateSchedule` instance, or
            a callable that takes no arguments and returns the actual value to
            use. The learning rate. Defaults to `0.0025`.
        beta_1: A float value or a constant float tensor, or a callable
            that takes no arguments and returns the actual value to use. The
            exponential decay rate for the 1st moment estimates and controls
            the interpolation between `momentum` and `averaged`.
            Defaults to `0.9`.
        warmup_steps: Number of warmup steps for learning rate warmup.
            During warmup, the learning rate linearly increases from 0 to the
            specified learning rate. Defaults to `0`.
        {{base_optimizer_keyword_args}}

    References:

    - [Defazio et al., 2024](https://arxiv.org/abs/2405.15682)
    - [Schedule-Free repository](
        https://github.com/facebookresearch/schedule_free)

    Example:

    >>> optimizer = keras.optimizers.ScheduleFreeSGD(learning_rate=0.0025)
    >>> model.compile(optimizer=optimizer, loss="mse")
    >>> model.fit(x_train, y_train)

    """

    def __init__(
        self,
        learning_rate=0.0025,
        beta_1=0.9,
        warmup_steps=0,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name=None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.warmup_steps = warmup_steps

    def build(self, var_list):
        """Initialize optimizer variables.

        ScheduleFreeSGD optimizer has the following variables:
        - `momentum`: Auxiliary variable where gradient updates are applied
        - `averaged`: Auxiliary variable storing the averaged sequence (x)

        Args:
            var_list: list of model variables to build optimizer variables on.
        """
        if self.built:
            return
        super().build(var_list)
        self._momentums = self.add_optimizer_variables(var_list, "momentum")
        self._averageds = self.add_optimizer_variables(var_list, "averaged")

        # Track sum of squared learning rates for weighted averaging
        self._sum_sq_lrs = self.add_variable(
            shape=(),
            initializer="zeros",
            dtype=backend.floatx(),
            name="sum_sq_lrs",
        )
        self._last_iteration = self.add_variable(
            shape=(),
            initializer="zeros",
            dtype="int",
            name="last_iteration",
        )
        self.assign(self._last_iteration, -1)

        # Initialize variables to match the initial parameter values
        for momentum, averaged, var in zip(
            self._momentums, self._averageds, var_list
        ):
            if momentum is not None:
                self.assign(momentum, ops.copy(var))
            if averaged is not None:
                self.assign(averaged, ops.copy(var))

    def _apply_weight_decay(self, variables):
        # We apply weight decay in update_step
        pass

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)

        # Apply warmup
        if self.warmup_steps > 0:
            warmup_steps = ops.cast(self.warmup_steps, variable.dtype)
            warmup_factor = ops.minimum(local_step / warmup_steps, 1.0)
            lr = lr * warmup_factor

        var_index = self._get_variable_index(variable)
        momentum = self._momentums[var_index]
        averaged = self._averageds[var_index]

        # Update sum of squared learning rates (once per iteration)
        last_iteration = ops.cast(self._last_iteration, "int")
        current_iteration = ops.cast(self.iterations, "int")

        new_sum_sq_lrs = ops.cond(
            ops.greater(current_iteration, last_iteration),
            lambda: ops.add(self._sum_sq_lrs, ops.square(lr)),
            lambda: ops.cast(self._sum_sq_lrs, lr.dtype),
        )
        self.assign(self._sum_sq_lrs, new_sum_sq_lrs)
        self.assign(self._last_iteration, current_iteration)
        sum_sq_lrs = new_sum_sq_lrs

        # Compute weight for averaging: weight = lr^2 / sum_sq_lrs
        weight = ops.divide_no_nan(ops.square(lr), sum_sq_lrs)

        # Apply weight decay
        if self.weight_decay is not None:

            def apply_wd():
                wd = ops.cast(self.weight_decay, variable.dtype)
                return ops.add(gradient, ops.multiply(wd, variable))

            gradient = ops.cond(
                self._use_weight_decay(variable),
                apply_wd,
                lambda: gradient,
            )

        # Update momentum: momentum = momentum - lr * gradient
        # Use updated values after possible initialization
        new_momentum = ops.subtract(momentum, ops.multiply(lr, gradient))
        self.assign(momentum, new_momentum)

        # Update averaged sequence:
        # x_new = (1 - weight) * x_old + weight * momentum_new
        new_averaged = ops.add(
            ops.multiply(1 - weight, averaged),
            ops.multiply(weight, new_momentum),
        )
        self.assign(averaged, new_averaged)

        # Update model variable:
        # y_new = (1 - beta_1) * momentum_new + beta_1 * x_new
        y_new = ops.add(
            ops.multiply(1 - beta_1, new_momentum),
            ops.multiply(beta_1, new_averaged),
        )

        self.assign(variable, y_new)

    def finalize_variable_values(self, var_list):
        """Overwrite model variables with the averaged sequence."""
        super().finalize_variable_values(var_list)
        for var, averaged in zip(var_list, self._averageds):
            if averaged is not None:
                self.assign(var, averaged)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "warmup_steps": self.warmup_steps,
            }
        )
        return config


if ScheduleFreeSGD.__doc__ is not None:
    ScheduleFreeSGD.__doc__ = ScheduleFreeSGD.__doc__.replace(
        "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
    )
