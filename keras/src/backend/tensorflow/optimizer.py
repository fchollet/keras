"""A class for Tensorflow specific optimizer logic.

The major behavior change for this class is for tf.distribute.

It will override methods from base Keras core Optimizer,
which provide distribute specific functionality, e.g. variable
creation, loss reduction, etc.
"""

import warnings

import tensorflow as tf

from keras.src import backend
from keras.src import ops
from keras.src.backend.tensorflow.trackable import KerasAutoTrackable
from keras.src.optimizers import base_optimizer


class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution_strategy = tf.distribute.get_strategy()

    def add_variable(
        self,
        shape,
        initializer="zeros",
        dtype=None,
        aggregation="none",
        layout=None,
        name=None,
    ):
        with tf.init_scope():
            return super().add_variable(
                shape,
                initializer=initializer,
                dtype=dtype,
                aggregation=aggregation,
                layout=layout,
                name=name,
            )

    def add_variable_from_reference(
        self, reference_variable, name=None, initializer="zeros"
    ):
        if isinstance(reference_variable, backend.Variable):
            colocate_var = reference_variable.value
        else:
            colocate_var = reference_variable

        with self._distribution_strategy.extended.colocate_vars_with(
            colocate_var
        ):
            return super().add_variable_from_reference(
                reference_variable, name=name, initializer=initializer
            )

    def stateless_apply(self, optimizer_variables, grads, trainable_variables):
        # This is mainly due to the interaction with tf.distribute.Strategy,
        # which requires tf.Variable as the inputs for most of its APIs.
        raise ValueError(
            "stateless_apply is not supported with the TensorFlow backend "
            "(as it is incompatible with tf.distribute)."
        )

    def assign(self, variable, value):
        if isinstance(variable, backend.Variable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)
        if isinstance(value, tf.IndexedSlices):
            variable.scatter_update(value)
        else:
            variable.assign(value)

    def assign_add(self, variable, value):
        if isinstance(variable, backend.Variable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)
        if isinstance(value, tf.IndexedSlices):
            variable.scatter_add(value)
        else:
            variable.assign_add(value)

    def assign_sub(self, variable, value):
        if isinstance(variable, backend.Variable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)
        if isinstance(value, tf.IndexedSlices):
            variable.scatter_sub(value)
        else:
            variable.assign_sub(value)

    def _var_key(self, variable):
        if isinstance(variable, backend.Variable):
            variable = variable.value  # Convert to tf.Variable
        if hasattr(variable, "_distributed_container"):
            variable = variable._distributed_container()
        elif (
            isinstance(variable, tf.__internal__.CompositeTensor)
            and hasattr(variable, "handle")
            and hasattr(variable.handle, "_distributed_container")
        ):
            # For ResourceVariables, the _distributed_container attribute
            # is added to their handle tensors.
            variable = variable.handle._distributed_container()
        return variable._unique_id

    def _apply_weight_decay(self, variables):
        if self.weight_decay is None:
            return

        def distributed_apply_weight_decay(distribution, variables, **kwargs):
            def weight_decay_fn(variable):
                if self._use_weight_decay(variable):
                    lr = tf.cast(self.learning_rate, variable.dtype)
                    wd = tf.cast(self.weight_decay, variable.dtype)
                    variable.assign_sub(variable * wd * lr)

            for variable in variables:
                if isinstance(variable, backend.Variable):
                    variable = variable.value  # Convert to tf.Variable
                distribution.extended.update(
                    variable, weight_decay_fn, group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            distributed_apply_weight_decay,
            self._distribution_strategy,
            variables,
        )

    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        trainable_variables = [
            v.value if isinstance(v, backend.Variable) else v
            for v in trainable_variables
        ]
        grads_and_vars = list(zip(grads, trainable_variables))
        grads_and_vars = self._all_reduce_sum_gradients(grads_and_vars)
        tf.__internal__.distribute.interim.maybe_merge_call(
            self._distributed_tf_update_step,
            self._distribution_strategy,
            grads_and_vars,
            learning_rate,
        )

    def _backend_apply_gradients(self, grads, trainable_variables):
        if (
            not self.gradient_accumulation_steps
            or tf.distribute.in_cross_replica_context()
            or tf.__internal__.distribute.strategy_supports_no_merge_call()
        ):
            return super()._backend_apply_gradients(grads, trainable_variables)

        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            return super()._backend_apply_gradients(grads, trainable_variables)

        def _distributed_apply_gradient_accumulation(
            distribution, grads, trainable_variables
        ):
            def _replica_is_update_step():
                return (
                    self._iterations + 1
                ) % self.gradient_accumulation_steps == 0

            is_update_step_per_replica = (
                distribution.extended.call_for_each_replica(
                    _replica_is_update_step
                )
            )
            is_update_step = distribution.experimental_local_results(
                is_update_step_per_replica
            )[0]

            def update_fn():
                def _replica_update(grads, trainable_variables):
                    acc_grads = self._get_accumulated_gradients(
                        trainable_variables
                    )
                    self._backend_apply_accumulated_gradients(
                        grads, trainable_variables, acc_grads
                    )

                return distribution.extended.call_for_each_replica(
                    _replica_update, args=(grads, trainable_variables)
                )

            def increment_fn():
                def _replica_increment(grads, trainable_variables):
                    acc_grads = self._get_accumulated_gradients(
                        trainable_variables
                    )
                    self._backend_increment_gradient_accumulators(
                        grads, acc_grads
                    )

                return distribution.extended.call_for_each_replica(
                    _replica_increment, args=(grads, trainable_variables)
                )

            # The branch bodies may call `merge_call` via all-reduce/update
            # hooks, so the conditional itself must live in cross-replica
            # context.
            return ops.cond(is_update_step, update_fn, increment_fn)

        return replica_context.merge_call(
            _distributed_apply_gradient_accumulation,
            args=(grads, trainable_variables),
        )

    def _distributed_tf_update_step(
        self, distribution, grads_and_vars, learning_rate
    ):
        def apply_grad_to_update_var(var, grad, learning_rate):
            return self.update_step(grad, var, learning_rate)

        for grad, var in grads_and_vars:
            distribution.extended.update(
                var,
                apply_grad_to_update_var,
                args=(grad, learning_rate),
                group=False,
            )

    def _all_reduce_sum_gradients(self, grads_and_vars):
        """Returns all-reduced gradients aggregated via summation.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.

        Returns:
            List of (gradient, variable) pairs
            where gradients have been all-reduced.
        """
        replica_context = tf.distribute.get_replica_context()
        if not replica_context:
            return grads_and_vars

        grads_and_vars = list(grads_and_vars)
        filtered_grads_and_vars = filter_empty_gradients(grads_and_vars)
        if filtered_grads_and_vars:
            grads = [pair[0] for pair in filtered_grads_and_vars]
            reduced = tf.distribute.get_replica_context().all_reduce(
                tf.distribute.ReduceOp.SUM, grads
            )
        else:
            reduced = []
        # Copy 'reduced' but add None gradients back in
        reduced_with_nones = []
        reduced_pos = 0
        for g, v in grads_and_vars:
            if g is None:
                reduced_with_nones.append((None, v))
            else:
                reduced_with_nones.append((reduced[reduced_pos], v))
                reduced_pos += 1
        if reduced_pos != len(reduced):
            raise ValueError(
                "Internal error: Failed to add all gradients. Expected to "
                f"process {len(reduced)} gradients, but processed "
                f"{reduced_pos}."
            )
        return reduced_with_nones

    def _backend_maybe_overwrite_model_variables(self, trainable_variables):
        maybe_overwrite = super()._backend_maybe_overwrite_model_variables
        if (
            tf.distribute.in_cross_replica_context()
            or tf.__internal__.distribute.strategy_supports_no_merge_call()
        ):
            return maybe_overwrite(trainable_variables)

        # The overwrite calls `distribution.extended.update`, which calls
        # `merge_call` in replica context. DistributionStrategy does not
        # support a cond in replica context with a branch that calls
        # `merge_call`, so we call the cond in cross-replica context.
        def _distributed_maybe_overwrite(distribution):
            maybe_overwrite(trainable_variables)

        tf.distribute.get_replica_context().merge_call(
            _distributed_maybe_overwrite
        )

    def _overwrite_model_variables_with_average_value(
        self, trainable_variables
    ):
        """Overwrite model variables with their moving average values.

        This function overwrites variables on each device.

        Args:
          var_list: list of model variables.
        """
        trainable_variables = [
            v.value if isinstance(v, backend.Variable) else v
            for v in trainable_variables
        ]
        # Override model variable by the stored average value on all devices.
        for var, average_var in zip(
            trainable_variables, self._model_variables_moving_average
        ):
            self._distribution_strategy.extended.update(
                var, lambda a, b: a.assign(b), args=(average_var,)
            )

    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        def update_accumulator(var, grad):
            var.assign(var + grad)

        accumulators = [v.value for v in acc_grads]

        def _distributed_tf_increment_grad_acc(
            distribution, grads, accumulators
        ):
            for grad, var in zip(grads, accumulators):
                distribution.extended.update(
                    var, update_accumulator, args=(grad,), group=False
                )

        tf.__internal__.distribute.interim.maybe_merge_call(
            _distributed_tf_increment_grad_acc,
            self._distribution_strategy,
            grads,
            accumulators,
        )

    def _clip_by_norm(self, values, axes=None):
        # We need to use TF-specific OP to support the case,
        # when `values` are `tf.IndexedSlices`.
        return tf.clip_by_norm(values, self.clipnorm, axes)


def filter_empty_gradients(grads_and_vars):
    """Filter out `(grad, var)` pairs that have a gradient equal to `None`."""
    grads_and_vars = tuple(grads_and_vars)
    if not grads_and_vars:
        return grads_and_vars

    filtered = []
    vars_with_empty_grads = []
    for grad, var in grads_and_vars:
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append((grad, var))
    filtered = tuple(filtered)

    if not filtered:
        variable = ([v.name for _, v in grads_and_vars],)
        raise ValueError(
            f"No gradients provided for any variable: {variable}. "
            f"Provided `grads_and_vars` is {grads_and_vars}."
        )
    if vars_with_empty_grads:
        warnings.warn(
            "Gradients do not exist for variables %s when minimizing the "
            "loss. If you're using `model.compile()`, did you forget to "
            "provide a `loss` argument?",
            ([v.name for v in vars_with_empty_grads]),
        )
    return filtered
