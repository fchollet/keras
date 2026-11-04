from keras.src import backend
from keras.src import initializers
from keras.src import losses
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend.common import get_keras_mask
from keras.src.metrics.metric import Metric
from keras.src.saving import serialization_lib


def reduce_to_samplewise_values(values, sample_weight, reduce_fn, dtype):
    dtype = dtype or backend.floatx()
    mask = get_keras_mask(values)
    values = ops.cast(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=dtype)

        if mask is not None:
            sample_weight = losses.loss.apply_mask(
                sample_weight, mask, dtype=dtype, reduction="sum"
            )
        # Update dimensions of weights to match with values if possible.
        values, sample_weight = losses.loss.squeeze_or_expand_to_same_rank(
            values, sample_weight
        )
        # Reduce values to same ndim as weight array.
        weight_ndim = len(sample_weight.shape)
        values_ndim = len(values.shape)
        if values_ndim > weight_ndim:
            values = reduce_fn(
                values, axis=list(range(weight_ndim, values_ndim))
            )
        # Broadcast sample_weight. It doesn't change the multiplication below
        # but changes the sample_weight reduction applied later.
        sample_weight = ops.broadcast_to(sample_weight, ops.shape(values))
        values = values * sample_weight
        if weight_ndim > 1:
            sample_weight = reduce_fn(
                sample_weight, axis=list(range(1, weight_ndim))
            )

    values_ndim = len(values.shape)
    if values_ndim > 1:
        values = reduce_fn(values, axis=list(range(1, values_ndim)))
    return values, sample_weight


@keras_export("keras.metrics.Sum")
class Sum(Metric):
    """Compute the (weighted) sum of the given values.

    For example, if `values` is `[1, 3, 5, 7]` then their sum is 16.
    If `sample_weight` was specified as `[1, 1, 0, 0]` then the sum would be 4.

    This metric creates one variable, `total`.
    This is ultimately returned as the sum value.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = metrics.Sum()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    16.0

    >>> m = metrics.Sum()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result()
    4.0
    """

    def __init__(self, name="sum", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.total = self.add_variable(
            shape=(),
            initializer=initializers.Zeros(),
            dtype=self.dtype,
            name="total",
        )

    def update_state(self, values, sample_weight=None):
        values, _ = reduce_to_samplewise_values(
            values, sample_weight, reduce_fn=ops.sum, dtype=self.dtype
        )
        self.total.assign_add(ops.sum(values))

    def reset_state(self):
        self.total.assign(0)

    def result(self):
        return ops.cast(self.total, self.dtype)


@keras_export("keras.metrics.Mean")
class Mean(Metric):
    """Compute the (weighted) mean of the given values.

    For example, if values is `[1, 3, 5, 7]` then the mean is 4.
    If `sample_weight` was specified as `[1, 1, 0, 0]` then the mean would be 2.

    This metric creates two variables, `total` and `count`.
    The mean value returned is simply `total` divided by `count`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

    >>> m = Mean()
    >>> m.update_state([1, 3, 5, 7])
    >>> m.result()
    4.0

    >>> m.reset_state()
    >>> m.update_state([1, 3, 5, 7], sample_weight=[1, 1, 0, 0])
    >>> m.result()
    2.0
    """

    def __init__(self, name="mean", dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.total = self.add_variable(
            shape=(),
            initializer=initializers.Zeros(),
            dtype=self.dtype,
            name="total",
        )
        self.count = self.add_variable(
            shape=(),
            initializer=initializers.Zeros(),
            dtype=self.dtype,
            name="count",
        )

    def update_state(self, values, sample_weight=None):
        values, sample_weight = reduce_to_samplewise_values(
            values, sample_weight, reduce_fn=ops.mean, dtype=self.dtype
        )
        self.total.assign_add(ops.sum(values))
        if sample_weight is not None:
            num_samples = ops.sum(sample_weight)
        elif len(values.shape) >= 1:
            num_samples = ops.shape(values)[0]
        else:
            num_samples = 1
        self.count.assign_add(ops.cast(num_samples, dtype=self.dtype))

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)

    def result(self):
        return ops.divide_no_nan(
            self.total, ops.cast(self.count, dtype=self.dtype)
        )


@keras_export("keras.metrics.MeanMetricWrapper")
class MeanMetricWrapper(Mean):
    """Wrap a stateless metric function with the `Mean` metric.

    You could use this class to quickly build a mean metric from a function. The
    function needs to have the signature `fn(y_true, y_pred)` and return a
    per-sample loss array. `MeanMetricWrapper.result()` will return
    the average metric value across all samples seen so far.

    For example:

    ```python
    def mse(y_true, y_pred):
        return (y_true - y_pred) ** 2

    mse_metric = MeanMetricWrapper(fn=mse)
    ```

    Args:
        fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        **kwargs: Keyword arguments to pass on to `fn`.
    """

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

        # If we are wrapping a Keras loss, register the metric's
        # direction as "down" (needs to be minimized during training).
        if (
            self._fn in losses.ALL_OBJECTS
            or hasattr(self._fn, "__class__")
            and self._fn.__class__ in losses.ALL_OBJECTS
        ):
            self._direction = "down"

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_keras_mask(y_pred)
        values = self._fn(y_true, y_pred, **self._fn_kwargs)
        sample_weight = losses.loss.apply_mask(
            sample_weight, mask, dtype=self.dtype, reduction="sum"
        )
        return super().update_state(values, sample_weight=sample_weight)

    def get_config(self):
        base_config = super().get_config()
        config = {"fn": serialization_lib.serialize_keras_object(self._fn)}
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_keras_object(config)
        return cls(**config)
