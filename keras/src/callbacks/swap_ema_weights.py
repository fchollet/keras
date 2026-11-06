from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback


@keras_export("keras.callbacks.SwapEMAWeights")
class SwapEMAWeights(Callback):
    """Swaps model weights and EMA weights before and after evaluation.

    This callbacks replaces the model's weight values with the values of
    the optimizer's EMA weights (the exponential moving average of the past
    model weights values, implementing "Polyak averaging") before model
    evaluation, and restores the previous weights after evaluation.

    The `SwapEMAWeights` callback is to be used in conjunction with
    an optimizer that sets `use_ema=True`.

    Note that the weights are swapped in-place in order to save memory.
    The behavior is undefined if you modify the EMA weights
    or model weights in other callbacks.

    Example:

    ```python
    # Remember to set `use_ema=True` in the optimizer
    optimizer = SGD(use_ema=True)
    model.compile(optimizer=optimizer, loss=..., metrics=...)

    # Metrics will be computed with EMA weights
    model.fit(X_train, Y_train, callbacks=[SwapEMAWeights()])

    # If you want to save model checkpoint with EMA weights, you can set
    # `swap_on_epoch=True` and place ModelCheckpoint after SwapEMAWeights.
    model.fit(
        X_train,
        Y_train,
        callbacks=[SwapEMAWeights(swap_on_epoch=True), ModelCheckpoint(...)]
    )
    ```

    Args:
        swap_on_epoch: whether to perform swapping at `on_epoch_begin()`
            and `on_epoch_end()`. This is useful if you want to use
            EMA weights for other callbacks such as `ModelCheckpoint`.
            Defaults to `False`.
    """

    def __init__(self, swap_on_epoch=False):
        super().__init__()
        self.swap_on_epoch = swap_on_epoch

        self._ema_weights_in_model = False

    def _tf_swap_variables(self, optimizer):
        variables = optimizer._trainable_variables
        for var, average_var in zip(
            variables,
            optimizer._model_variables_moving_average,
        ):
            if isinstance(var, backend.Variable):
                var = var.value
            if isinstance(average_var, backend.Variable):
                average_var = average_var.value
            # swap using addition to prevent variable creation
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: a.assign_add(b),
                args=(average_var,),
            )
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: b.assign(a - b),
                args=(average_var,),
            )
            optimizer._distribution_strategy.extended.update(
                var,
                lambda a, b: a.assign(a - b),
                args=(average_var,),
            )

    def _backend_swap_variables(self, optimizer):
        variables = optimizer._trainable_variables
        for var, average_var in zip(
            variables,
            optimizer._model_variables_moving_average,
        ):
            temporary_variable = ops.convert_to_numpy(var)
            var.assign(average_var)
            average_var.assign(temporary_variable)

    def _tf_finalize_ema_values(self, optimizer):
        variables = optimizer._trainable_variables
        for var, average_var in zip(
            variables,
            optimizer._model_variables_moving_average,
        ):
            if isinstance(var, backend.Variable):
                var = var.value
            if isinstance(average_var, backend.Variable):
                average_var = average_var.value
            optimizer._distribution_strategy.extended.update(
                average_var,
                lambda a, b: a.assign(b),
                args=(var,),
            )

    def _backend_finalize_ema_values(self, optimizer):
        variables = optimizer._trainable_variables
        for var, average_var in zip(
            variables,
            optimizer._model_variables_moving_average,
        ):
            average_var.assign(var)

    def _apply_to_ema_optimizers(self, tf_fn, backend_fn):
        if hasattr(self.model.optimizer, "inner_optimizer"):
            # LossScaleOptimizer
            optimizer = self.model.optimizer.inner_optimizer
        else:
            optimizer = self.model.optimizer

        if hasattr(optimizer, "optimizers"):
            has_ema = any(
                getattr(opt, "use_ema", False) for opt in optimizer.optimizers
            )
            if not has_ema:
                raise ValueError(
                    "SwapEMAWeights must be used when at least one inner "
                    "optimizer has `use_ema=True` set. Received a "
                    "MultiOptimizer with all inner optimizers having "
                    "`use_ema=False`."
                )
            for opt in optimizer.optimizers:
                if getattr(opt, "use_ema", False):
                    if not hasattr(opt, "_model_variables_moving_average"):
                        raise ValueError(
                            "SwapEMAWeights must be used when "
                            "`use_ema=True` is set on the optimizer. "
                            f"Received: use_ema={opt.use_ema}"
                        )
                    if backend.backend() == "tensorflow":
                        tf_fn(opt)
                    else:
                        backend_fn(opt)
        else:
            if not hasattr(optimizer, "_model_variables_moving_average"):
                raise ValueError(
                    "SwapEMAWeights must be used when "
                    "`use_ema=True` is set on the optimizer. "
                    f"Received: use_ema={optimizer.use_ema}"
                )
            if backend.backend() == "tensorflow":
                tf_fn(optimizer)
            else:
                backend_fn(optimizer)

    def _swap_variables(self):
        self._apply_to_ema_optimizers(
            self._tf_swap_variables, self._backend_swap_variables
        )

    def _finalize_ema_values(self):
        self._apply_to_ema_optimizers(
            self._tf_finalize_ema_values, self._backend_finalize_ema_values
        )

    def on_epoch_begin(self, epoch, logs=None):
        if self.swap_on_epoch and self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = False

    def on_epoch_end(self, epoch, logs=None):
        if self.swap_on_epoch and not self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = True
            # We need to recover EMA weights from the previously swapped weights
            # in the last epoch. This is because, at the end of the fitting,
            # `finalize_variable_values` will be called to assign
            # `_model_variables_moving_average` to `trainable_variables`.
            if epoch == self.params["epochs"] - 1:
                self._finalize_ema_values()

    def on_test_begin(self, logs=None):
        if not self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = True

    def on_test_end(self, logs=None):
        if self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = False

    def on_predict_begin(self, logs=None):
        if not self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = True

    def on_predict_end(self, logs=None):
        if self._ema_weights_in_model:
            self._swap_variables()
            self._ema_weights_in_model = False
