from keras.src import backend
from keras.src import initializers
from keras.src.api_export import keras_export
from keras.src.optimizers import optimizer
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


@keras_export(
    [
        "keras.optimizers.LossScaleOptimizer",
        "keras.mixed_precision.LossScaleOptimizer",
    ]
)
class LossScaleOptimizer(optimizer.Optimizer):
    """An optimizer that dynamically scales the loss to prevent underflow.

    Loss scaling is a technique to prevent numeric underflow in intermediate
    gradients when float16 is used. To prevent underflow, the loss is multiplied
    (or "scaled") by a certain factor called the "loss scale", which causes
    intermediate gradients to be scaled by the loss scale as well. The final
    gradients are divided (or "unscaled") by the loss scale to bring them back
    to their original value.

    `LossScaleOptimizer` wraps another optimizer and applies dynamic loss
    scaling to it. This loss scale is dynamically updated over time as follows:
    - On any train step, if a nonfinite gradient is encountered, the loss scale
      is halved, and the train step is skipped.
    - If `dynamic_growth_steps` have occurred since the last time the loss scale
      was updated, and no nonfinite gradients have occurred, the loss scale
      is doubled.

    Args:
        inner_optimizer: The `keras.optimizers.Optimizer` instance to wrap.
        initial_scale: Float. The initial loss scale. This scale will be updated
            during training. It is recommended for this to be a very high
            number, because a loss scale that is too high gets lowered far more
            quickly than a loss scale that is too low gets raised.
        dynamic_growth_steps: Int. How often to update the scale upwards. After
            every `dynamic_growth_steps` steps with finite gradients, the
            loss scale is doubled.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        inner_optimizer,
        initial_scale=2.0**15,
        dynamic_growth_steps=2000,
        name=None,
        **kwargs,
    ):
        if not kwargs.pop("dynamic", True):
            raise ValueError(
                "LossScaleOptimizer no longer supports `dynamic=False`. "
                "Instead, simply set `loss_scale_factor` directly on the "
                "`inner_optimizer`."
            )

        # Backwards compatibility code for deserialization.
        # LossScaleOptimizer used to return all these parameters in `get_config`
        # from `super.get_config` even though they are all non-functional. We
        # no longer let user set them, but we have to allow the default values
        # to be passed during deserialization to support older models.
        base_optimizer_defaults = {
            "weight_decay": None,
            "clipnorm": None,
            "global_clipnorm": None,
            "clipvalue": None,
            "use_ema": False,
            "ema_momentum": 0.99,
            "ema_overwrite_frequency": None,
            "loss_scale_factor": None,
            "gradient_accumulation_steps": None,
        }
        for arg_name, default_value in base_optimizer_defaults.items():
            if arg_name not in kwargs:
                continue
            arg_value = kwargs.pop(arg_name)
            if (
                default_value is None and arg_value is not None
            ) or arg_value != default_value:
                raise ValueError(
                    f"LossScaleOptimizer does not support `{arg_name}`. "
                    f"Instead, set `{arg_name}` on the `inner_optimizer`."
                )

        if kwargs:
            raise ValueError(
                "LossScaleOptimizer does not support arguments: "
                f"`{'`, `'.join(kwargs.keys())}`."
            )

        super().__init__(learning_rate=0.0, name=name)
        self.inner_optimizer = inner_optimizer
        self.initial_scale = initial_scale
        self.dynamic_growth_steps = dynamic_growth_steps
        # Disable the inner optimizer's loss scaling, otherwise
        # gradients will be scaled twice.
        self.inner_optimizer.loss_scale_factor = None

    @tracking.no_automatic_dependency_tracking
    def build(self, var_list):
        self.step_counter = self.add_variable(
            shape=(),
            dtype="int",
            initializer=initializers.Zeros(),
            aggregation="none",
            name="step_counter",
        )
        self.dynamic_scale = self.add_variable(
            shape=(),
            dtype="float32",
            initializer=initializers.Constant(self.initial_scale),
            aggregation="none",
            name="dynamic_scale",
        )
        self.inner_optimizer.build(var_list)
        super().build(var_list)

    @property
    def variables(self):
        return self._variables + self.inner_optimizer.variables

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        if not self.built:
            raise ValueError(
                f"To call `stateless_apply`, {self.__class__.__name__} "
                "must be built (i.e. its variables must have been created). "
                "You can build it via `optimizer.build(trainable_variables)`."
            )
        finite = self.check_finite(grads)
        return backend.ops.cond(
            finite,
            lambda: self._stateless_handle_finite_grads(
                optimizer_variables, grads, trainable_variables
            ),
            lambda: self._stateless_handle_non_finite_grads(
                optimizer_variables, trainable_variables
            ),
        )

    def _unscale_grads(self, grads, variables):
        # Divide gradients by the loss scale, keeping any gradient that
        # overwrites its variable untouched.
        return [
            g
            if g is None or self._overwrite_variable_with_gradient(v)
            else backend.ops.numpy.divide(g, self.dynamic_scale)
            for g, v in zip(grads, variables)
        ]

    def _stateless_handle_finite_grads(
        self, optimizer_variables, grads, trainable_variables
    ):
        def upscale():
            mapping = list(zip(self.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping) as scope:
                self.step_counter.assign(0)
                self.dynamic_scale.assign(
                    backend.ops.numpy.multiply(self.dynamic_scale, 2.0)
                )
            return [scope.get_current_value(v) for v in self._variables]

        def increment():
            mapping = list(zip(self.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping) as scope:
                self.step_counter.assign_add(1)
            return [scope.get_current_value(v) for v in self._variables]

        mapping = list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping):
            # Potentially upscale loss and reset counter.
            own_variables = backend.ops.cond(
                backend.ops.numpy.equal(
                    self.step_counter, self.dynamic_growth_steps - 1
                ),
                upscale,
                increment,
            )

            unscaled_grads = self._unscale_grads(
                grads, self._trainable_variables
            )
            (
                new_trainable_variables,
                new_inner_variables,
            ) = self.inner_optimizer.stateless_apply(
                self.inner_optimizer.variables,
                unscaled_grads,
                trainable_variables,
            )

        new_optimizer_variables = own_variables + new_inner_variables
        return new_trainable_variables, new_optimizer_variables

    def _stateless_handle_non_finite_grads(
        self, optimizer_variables, trainable_variables
    ):
        mapping = list(zip(self.variables, optimizer_variables))
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.step_counter.assign(0)
            self.dynamic_scale.assign(
                backend.ops.numpy.multiply(self.dynamic_scale, 0.5)
            )
        new_optimizer_variables = []
        for v in self.variables:
            new_optimizer_variables.append(scope.get_current_value(v))
        return trainable_variables, new_optimizer_variables

    def apply(self, grads, trainable_variables=None):
        # Optionally build optimizer.
        if not self.built:
            with backend.name_scope(self.name, caller=self):
                self.build(trainable_variables)
            self.built = True

        if backend.backend() == "tensorflow":
            self._tf_apply(grads, trainable_variables)
        else:
            self._common_apply(grads, trainable_variables)

    def _stateful_handle_finite_grads(self, grads, trainable_variables):
        tvs = trainable_variables or self._trainable_variables
        unscaled_grads = self._unscale_grads(grads, tvs)
        self.inner_optimizer.apply(
            unscaled_grads, trainable_variables=trainable_variables
        )

        def upscale():
            self.step_counter.assign(0)
            self.dynamic_scale.assign(
                backend.ops.numpy.multiply(self.dynamic_scale, 2.0)
            )

        def increment():
            self.step_counter.assign_add(1)

        # Potentially upscale loss and reset counter.
        backend.ops.cond(
            backend.ops.numpy.equal(
                self.step_counter, self.dynamic_growth_steps - 1
            ),
            upscale,
            increment,
        )

    def _stateful_handle_non_finite_grads(self):
        # If any inf or nan in grads, downscale loss and reset counter.
        self.step_counter.assign(0)
        self.dynamic_scale.assign(
            backend.ops.numpy.multiply(self.dynamic_scale, 0.5)
        )

    def _common_apply(self, grads, trainable_variables=None):
        finite = self.check_finite(grads)
        backend.ops.cond(
            finite,
            lambda: self._stateful_handle_finite_grads(
                grads, trainable_variables
            ),
            self._stateful_handle_non_finite_grads,
        )

    def _tf_apply(self, grads, trainable_variables=None):
        """Tensorflow specific logic for apply, which handles distribution."""
        from keras.src.utils.module_utils import tensorflow as tf

        if tf.distribute.in_cross_replica_context():
            raise ValueError("apply() must be called in a replica context.")

        if tf.__internal__.distribute.strategy_supports_no_merge_call():
            self._common_apply(grads, trainable_variables=trainable_variables)
        else:

            def _handle_cross_replica(distribution, grads, trainable_variables):
                finite_per_replica = (
                    distribution.extended.call_for_each_replica(
                        lambda grads: tf.cast(
                            self.check_finite(grads), tf.int32
                        ),
                        args=(grads,),
                    )
                )
                # Gradients are still local to each replica, so all replicas
                # must have finite gradients before any update is applied.
                finite = distribution.reduce(
                    tf.distribute.ReduceOp.SUM, finite_per_replica, axis=None
                )
                finite = tf.equal(finite, distribution.num_replicas_in_sync)

                def apply_fn():
                    distribution.extended.call_for_each_replica(
                        self._stateful_handle_finite_grads,
                        args=(grads, trainable_variables),
                    )

                # Note: We must call this cond() in a cross-replica context.
                # DistributionStrategy does not support having a cond in a
                # replica context with a branch that calls `merge_call`, and
                # self._optimizer.apply_gradients calls `merge_call`.
                backend.ops.cond(
                    finite, apply_fn, self._stateful_handle_non_finite_grads
                )

            tf.distribute.get_replica_context().merge_call(
                _handle_cross_replica, args=(grads, trainable_variables)
            )

    def check_finite(self, grads):
        tensor_grads = [g for g in grads if g is not None]
        finite_grads = [
            backend.ops.numpy.all(backend.ops.numpy.isfinite(g))
            for g in tensor_grads
        ]
        return backend.ops.numpy.all(
            backend.ops.convert_to_tensor(finite_grads)
        )

    @property
    def learning_rate(self):
        return self.inner_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.inner_optimizer.learning_rate = learning_rate

    @property
    def iterations(self):
        return self.inner_optimizer.iterations

    def scale_loss(self, loss):
        scale = self.dynamic_scale if self.built else self.initial_scale
        return backend.ops.numpy.multiply(loss, scale)

    def finalize_variable_values(self, var_list):
        self.inner_optimizer.finalize_variable_values(var_list)

    def get_config(self):
        # Do not use super().get_config() as only "name" is supported.
        inner_optimizer_config = serialization_lib.serialize_keras_object(
            self.inner_optimizer
        )
        return {
            "name": self.name,
            "inner_optimizer": inner_optimizer_config,
            "initial_scale": self.initial_scale,
            "dynamic_growth_steps": self.dynamic_growth_steps,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        inner_optimizer = serialization_lib.deserialize_keras_object(
            config.pop("inner_optimizer"),
            custom_objects=custom_objects,
        )
        return cls(inner_optimizer, **config)


if LossScaleOptimizer.__doc__ is not None:
    LossScaleOptimizer.__doc__ = LossScaleOptimizer.__doc__.replace(
        "{{base_optimizer_keyword_args}}", optimizer.base_optimizer_keyword_args
    )
