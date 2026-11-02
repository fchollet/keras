import numbers
import warnings

import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import set_keras_mask
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.losses.loss import Loss
from keras.src.losses.loss import reduce_weighted_values
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import build_pos_neg_masks
from keras.src.utils.numerical_utils import normalize


def _validate_label_smoothing(label_smoothing):
    """Validate a static `label_smoothing` argument.

    Tensor values are skipped (validated at runtime by the ops themselves).
    """
    if not ops.is_tensor(label_smoothing) or np.isscalar(label_smoothing):
        if isinstance(label_smoothing, (bool, np.bool_)) or not isinstance(
            label_smoothing, numbers.Real
        ):
            raise ValueError(
                "`label_smoothing` must be a float or int. "
                f"Received: label_smoothing={label_smoothing}"
            )
        value = float(label_smoothing)
        if not 0 <= value <= 1:
            raise ValueError(
                "`label_smoothing` must be in the range [0, 1]. "
                f"Received: label_smoothing={label_smoothing}"
            )


class LossFunctionWrapper(Loss):
    def __init__(
        self,
        fn,
        reduction="sum_over_batch_size",
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=name, reduction=reduction, dtype=dtype)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true_y_pred = tree.map_structure(
            squeeze_or_expand_to_same_rank, y_true, y_pred
        )
        y_true = tree.map_structure_up_to(y_true, lambda x: x[0], y_true_y_pred)
        y_pred = tree.map_structure_up_to(y_pred, lambda x: x[1], y_true_y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"fn": serialization_lib.serialize_keras_object(self.fn)})
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return config

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_keras_object(config)
        return cls(**config)

    def __repr__(self):
        return f"<LossFunctionWrapper({self.fn}, kwargs={self._fn_kwargs})>"


@keras_export("keras.losses.MeanSquaredError")
class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    >>> y_true = keras.ops.array([1.0, 0.0, 1.0])
    >>> y_pred = keras.ops.array([0.9, 0.1, 0.8])
    >>> loss = keras.losses.MeanSquaredError()
    >>> loss(y_true, y_pred)
    0.02
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_error",
        dtype=None,
    ):
        super().__init__(
            mean_squared_error, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.MeanAbsoluteError")
class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    >>> y_true = keras.ops.array([1.0, 0.3, 1.0])
    >>> y_pred = keras.ops.array([1.9, 0.3, 1.8])
    >>> loss = keras.losses.MeanAbsoluteError()
    >>> loss(y_true, y_pred)
    0.5666667
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_error",
        dtype=None,
    ):
        super().__init__(
            mean_absolute_error, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    >>> y_true = keras.ops.array([100.0, 200.0, 300.0])
    >>> y_pred = keras.ops.array([90.0, 210.0, 310.0])
    >>> loss = keras.losses.MeanAbsolutePercentageError()
    >>> loss(y_true, y_pred)
    6.111111
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_absolute_percentage_error",
        dtype=None,
    ):
        super().__init__(
            mean_absolute_percentage_error,
            name=name,
            reduction=reduction,
            dtype=dtype,
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="mean_squared_logarithmic_error",
        dtype=None,
    ):
        super().__init__(
            mean_squared_logarithmic_error,
            name=name,
            reduction=reduction,
            dtype=dtype,
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.CosineSimilarity")
class CosineSimilarity(LossFunctionWrapper):
    """Computes the cosine similarity between `y_true` & `y_pred`.

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity
    will be 0 regardless of the proximity between predictions and targets.

    Formula:

    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Args:
        axis: The axis along which the cosine similarity is computed
            (the features axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        axis=-1,
        reduction="sum_over_batch_size",
        name="cosine_similarity",
        dtype=None,
    ):
        super().__init__(
            cosine_similarity,
            name=name,
            reduction=reduction,
            dtype=dtype,
            axis=axis,
        )

    def get_config(self):
        config = super().get_config()
        config.pop("fn")
        return config


@keras_export("keras.losses.Huber")
class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` & `y_pred`.

    Formula:

    ```python
    for x in error:
        if abs(x) <= delta:
            loss.append(0.5 * x^2)
        elif abs(x) > delta:
            loss.append(delta * abs(x) - 0.5 * delta^2)

    loss = mean(loss, axis=-1)
    ```
    See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

    Args:
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        delta=1.0,
        reduction="sum_over_batch_size",
        name="huber_loss",
        dtype=None,
    ):
        if not isinstance(delta, float) or delta <= 0:
            raise ValueError(
                "Invalid value for argument `delta`. Expected a float "
                f"greater than 0. Received: delta={delta}"
            )
        super().__init__(
            huber,
            name=name,
            reduction=reduction,
            dtype=dtype,
            delta=delta,
        )

    def get_config(self):
        config = super().get_config()
        config.pop("fn")
        return config


@keras_export("keras.losses.LogCosh")
class LogCosh(LossFunctionWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    Formula:

    ```python
    error = y_pred - y_true
    logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)`
    ```
    where x is the error `y_pred - y_true`.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="log_cosh",
        dtype=None,
    ):
        super().__init__(log_cosh, name=name, reduction=reduction, dtype=dtype)

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.Hinge")
class Hinge(LossFunctionWrapper):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(1 - y_true * y_pred, 0)
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="hinge",
        dtype=None,
    ):
        super().__init__(hinge, name=name, reduction=reduction, dtype=dtype)

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.SquaredHinge")
class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = square(maximum(1 - y_true * y_pred, 0))
    ```

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="squared_hinge", dtype=None
    ):
        super().__init__(
            squared_hinge, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.CategoricalHinge")
class CategoricalHinge(LossFunctionWrapper):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="categorical_hinge",
        dtype=None,
    ):
        super().__init__(
            categorical_hinge, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.KLDivergence")
class KLDivergence(LossFunctionWrapper):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="kl_divergence", dtype=None
    ):
        super().__init__(
            kl_divergence, name=name, reduction=reduction, dtype=dtype
        )

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.Poisson")
class Poisson(LossFunctionWrapper):
    """Computes the Poisson loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self, reduction="sum_over_batch_size", name="poisson", dtype=None
    ):
        super().__init__(poisson, name=name, reduction=reduction, dtype=dtype)

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.BinaryCrossentropy")
class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
        floating-point value which either represents a
        [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
        when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
        `from_logits=False`).

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` is probabilities (i.e., values in [0, 1]).
        label_smoothing: Float in range [0, 1]. When 0, no smoothing occurs.
            When > 0, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards 0.5. Larger values of
            `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    **Recommended Usage:** (set `from_logits=True`)

    With `compile()` API:

    ```python
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = np.array([0, 1, 0, 0])
    >>> y_pred = np.array([-18.6, 0.51, 2.94, -12.8])
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.8654

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = np.array([[0, 1], [0, 0]])
    >>> y_pred = np.array([[-18.6, 0.51], [2.94, -12.8]])
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.8654
    >>> # Using 'sample_weight' attribute
    >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.243
    >>> # Using 'sum' reduction` type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction="sum")
    >>> bce(y_true, y_pred)
    1.730
    >>> # Using 'none' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=None)
    >>> bce(y_true, y_pred)
    array([0.235, 1.496], dtype=float32)

    **Default Usage:** (set `from_logits=False`)

    >>> # Make the following updates to the above "Recommended Usage" section
    >>> # 1. Set `from_logits=False`
    >>> keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
    >>> # 2. Update `y_pred` to use probabilities instead of logits
    >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_crossentropy",
        dtype=None,
    ):
        super().__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
            }
        )
        return config


@keras_export("keras.losses.BinaryFocalCrossentropy")
class BinaryFocalCrossentropy(LossFunctionWrapper):
    """Computes focal cross-entropy loss between true labels and predictions.

    Binary cross-entropy loss is often used for binary (0 or 1) classification
    tasks. The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
        floating-point value which either represents a
        [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
        when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
        `from_logits=False`).

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a "focal factor" to down-weight easy examples and focus more
    on hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma=0`, this function is
    equivalent to the binary crossentropy loss.

    Args:
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in reference [Lin et al., 2018](
            https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
            `1.0 - alpha`.
        gamma: A focusing parameter used to compute the focal factor, default is
            `2.0` as mentioned in the reference
            [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
        label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs.
            When > `0`, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards `0.5`.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    With the `compile()` API:

    ```python
    model.compile(
        loss=keras.losses.BinaryFocalCrossentropy(
            gamma=2.0, from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = np.array([0, 1, 0, 0])
    >>> y_pred = np.array([-18.6, 0.51, 2.94, -12.8])
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...    gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.691

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=2, from_logits=True)
    >>> loss(y_true, y_pred)
    0.51

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = np.array([[0, 1], [0, 0]])
    >>> y_pred = np.array([[-18.6, 0.51], [2.94, -12.8]])
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.647

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred)
    0.482

    >>> # Using 'sample_weight' attribute with focal effect
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.133

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...      apply_class_balancing=True, gamma=3, from_logits=True)
    >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.097

    >>> # Using 'sum' reduction` type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    1.222

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=4, from_logits=True,
    ...     reduction="sum")
    >>> loss(y_true, y_pred)
    0.914

    >>> # Using 'none' reduction type.
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0017 1.1561], dtype=float32)

    >>> # Apply class weight
    >>> loss = keras.losses.BinaryFocalCrossentropy(
    ...     apply_class_balancing=True, gamma=5, from_logits=True,
    ...     reduction=None)
    >>> loss(y_true, y_pred)
    array([0.0004 0.8670], dtype=float32)
    """

    def __init__(
        self,
        apply_class_balancing=False,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_focal_crossentropy",
        dtype=None,
    ):
        super().__init__(
            binary_focal_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            apply_class_balancing=apply_class_balancing,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        self.apply_class_balancing = apply_class_balancing
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
                "apply_class_balancing": self.apply_class_balancing,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }
        )
        return config


@keras_export("keras.losses.CategoricalCrossentropy")
class CategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes. We expect labels to be provided in a `one_hot` representation. If
    you want to provide labels as integers, please use
    `SparseCategoricalCrossentropy` loss. There should be `num_classes` floating
    point values per feature, i.e., the shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    Standalone usage:

    >>> y_true = np.array([[0, 1, 0], [0, 0, 1]])
    >>> y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> cce = keras.losses.CategoricalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras.losses.CategoricalCrossentropy())
    ```
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_crossentropy",
        dtype=None,
    ):
        super().__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
            }
        )
        return config


@keras_export("keras.losses.CategoricalFocalCrossentropy")
class CategoricalFocalCrossentropy(LossFunctionWrapper):
    """Computes the alpha balanced focal crossentropy loss.

    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    `class_weights`. We expect labels to be provided in a `one_hot`
    representation.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:

    `FL(p_t) = (1 - p_t) ** gamma * log(p_t)`

    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`

    `(1 - p_t) ** gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.

    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)`

    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.

    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CrossEntropy(y_true, y_pred)`

    where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`

    In the snippet below, there is `num_classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `(batch_size, num_classes)`.

    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    Standalone usage:

    >>> y_true = [[0., 1., 0.], [0., 0., 1.]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy()
    >>> cce(y_true, y_pred)
    0.23315276

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.1632

    >>> # Using 'sum' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy(
    ...     reduction="sum")
    >>> cce(y_true, y_pred)
    0.46631

    >>> # Using 'none' reduction type.
    >>> cce = keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=None)
    >>> cce(y_true, y_pred)
    array([3.2058331e-05, 4.6627346e-01], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalFocalCrossentropy())
    ```
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="categorical_focal_crossentropy",
        dtype=None,
    ):
        """Initializes `CategoricalFocalCrossentropy` instance."""
        super().__init__(
            categorical_focal_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }
        )
        return config


@keras_export("keras.losses.SparseCategoricalFocalCrossentropy")
class SparseCategoricalFocalCrossentropy(LossFunctionWrapper):
    """Computes the sparse categorical focal crossentropy loss.

    Use this crossentropy loss function when there are two or more label
    classes and labels are provided as integers. If you want to provide labels
    using a `one-hot` representation, please use
    `CategoricalFocalCrossentropy` loss.

    Focal loss applies a modulating factor to down-weight easy examples and
    focus more on hard examples. For each example, the loss is:

    `FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)`

    where `p_t` is the predicted probability of the true class, `gamma` is the
    focusing parameter, and `alpha_t` is the weight for the true class. When
    `gamma=0`, there is no focal effect on the crossentropy loss.

    Args:
        alpha: A weight balancing factor for all classes. Defaults to `0.25`.
            It can be a list of floats or a scalar. If a list is provided, it
            must have the same length as the number of classes.
        gamma: A focusing parameter. Defaults to `2.0`. It gradually reduces
            the importance given to easy examples.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, `y_pred` is expected to encode a probability distribution.
        ignore_class: Optional integer. The ID of a class to be ignored during
            loss computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255). By default
            (`ignore_class=None`), all classes are considered.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to a different value (via
            `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided,
            then the `compute_dtype` will be utilized.

    Examples:

    >>> y_true = np.array([1, 2])
    >>> y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> loss = keras.losses.SparseCategoricalFocalCrossentropy(
    ...     reduction=None
    ... )
    >>> loss(y_true, y_pred)
    array([3.2058e-05, 4.6627e-01], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalFocalCrossentropy(),
    )
    ```

    An out-of-range, non-ignored label produces a `NaN` loss value. An `alpha`
    vector with a statically known wrong length raises `ValueError`; if its
    length or the class dimension is only known at runtime, a mismatch produces
    `NaN` loss values. This prevents backend-dependent index selection in
    compiled and symbolic execution, where Keras does not have a portable
    runtime assertion primitive.
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        ignore_class=None,
        axis=-1,
        reduction="sum_over_batch_size",
        name="sparse_categorical_focal_crossentropy",
        dtype=None,
    ):
        super().__init__(
            sparse_categorical_focal_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.ignore_class = ignore_class
        self.axis = axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Sparse labels must retain their integer dtype until the loss validates
        # and gathers them. In particular, bfloat16 cannot represent every
        # integer above 256. The generic Loss.__call__ casts both arguments to
        # the compute dtype, which can silently change a valid class ID or an
        # out-of-range ignore sentinel.
        in_mask = get_keras_mask(y_pred)

        with backend.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
            )
            y_true = tree.map_structure(ops.convert_to_tensor, y_true)

            loss_values = self.call(y_true, y_pred)
            out_mask = get_keras_mask(loss_values)

            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None

            return reduce_weighted_values(
                loss_values,
                sample_weight=sample_weight,
                mask=mask,
                reduction=self.reduction,
                dtype=self.dtype,
            )

    def call(self, y_true, y_pred):
        # Shape handling for this loss is relative to its configurable class
        # axis. Bypass LossFunctionWrapper's last-axis rank adjustment and let
        # the focal loss perform its axis-aware validation and squeeze.
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "from_logits": self.from_logits,
                "ignore_class": self.ignore_class,
                "axis": self.axis,
            }
        )
        return config


@keras_export("keras.losses.SparseCategoricalCrossentropy")
class SparseCategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label
    classes.  We expect labels to be provided as integers. If you want to
    provide labels using `one-hot` representation, please use
    `CategoricalCrossentropy` loss.  There should be `# classes` floating point
    values per feature for `y_pred` and a single floating point value per
    feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `num_classes` floating pointing values per example for
    `y_pred`. The shape of `y_true` is `[batch_size]` and the shape of `y_pred`
    is `[batch_size, num_classes]`.

    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to `-1`.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels. Defaults to `0.0`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    >>> y_true = np.array([1, 2])
    >>> y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy()
    >>> scce(y_true, y_pred)
    1.177

    >>> # Calling with 'sample_weight'.
    >>> scce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
    0.814

    >>> # Using 'sum' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy(
    ...     reduction="sum")
    >>> scce(y_true, y_pred)
    2.354

    >>> # Using 'none' reduction type.
    >>> scce = keras.losses.SparseCategoricalCrossentropy(
    ...     reduction=None)
    >>> scce(y_true, y_pred)
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss=keras.losses.SparseCategoricalCrossentropy())
    ```
    """

    def __init__(
        self,
        from_logits=False,
        ignore_class=None,
        reduction="sum_over_batch_size",
        axis=-1,
        label_smoothing=0.0,
        name="sparse_categorical_crossentropy",
        dtype=None,
    ):
        super().__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
            label_smoothing=label_smoothing,
        )

    def get_config(self):
        config = super().get_config()
        config.pop("fn")
        return config


@keras_export("keras.losses.CTC")
class CTC(LossFunctionWrapper):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(self, reduction="sum_over_batch_size", name="ctc", dtype=None):
        super().__init__(ctc, name=name, reduction=reduction, dtype=dtype)

    def get_config(self):
        return Loss.get_config(self)


@keras_export("keras.losses.Dice")
class Dice(LossFunctionWrapper):
    """Computes the Dice loss value between `y_true` and `y_pred`.

    Formula:
    ```python
    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    ```

    Args:
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        axis: Tuple for which dimensions the loss is calculated. Defaults to
            `None`.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Returns:
        Dice loss value.

    Example:

    >>> y_true = [[[[1.0], [1.0]], [[0.0], [0.0]]],
    ...           [[[1.0], [1.0]], [[0.0], [0.0]]]]
    >>> y_pred = [[[[0.0], [1.0]], [[0.0], [1.0]]],
    ...           [[[0.4], [0.0]], [[0.0], [0.9]]]]
    >>> axis = (1, 2, 3)
    >>> loss = keras.losses.Dice(axis=axis, reduction=None)(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.5, 0.75757575], shape=(2,), dtype=float32)

    >>> loss = keras.losses.Dice()(y_true, y_pred)
    >>> assert loss.shape == ()
    >>> loss
    array(0.6164384, shape=(), dtype=float32)

    >>> y_true = np.array(y_true)
    >>> y_pred = np.array(y_pred)
    >>> loss = keras.losses.Dice(axis=axis, reduction=None)(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.5, 0.75757575], shape=(2,), dtype=float32)

    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="dice",
        axis=None,
        dtype=None,
    ):
        super().__init__(
            dice, name=name, reduction=reduction, dtype=dtype, axis=axis
        )
        self.axis = axis

    def get_config(self):
        config = Loss.get_config(self)
        config.update({"axis": self.axis})
        return config


@keras_export("keras.losses.Tversky")
class Tversky(LossFunctionWrapper):
    """Computes the Tversky loss value between `y_true` and `y_pred`.

    This loss function is weighted by the alpha and beta coefficients
    that penalize false positives and false negatives.

    With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
    Dice Loss.

    Args:
        alpha: The coefficient controlling incidence of false positives.
            Defaults to `0.5`.
        beta: The coefficient controlling incidence of false negatives.
            Defaults to `0.5`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Returns:
        Tversky loss value.

    Reference:

    - [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)
    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.5,
        reduction="sum_over_batch_size",
        name="tversky",
        axis=None,
        dtype=None,
    ):
        super().__init__(
            tversky,
            name=name,
            reduction=reduction,
            dtype=dtype,
            alpha=alpha,
            beta=beta,
            axis=axis,
        )
        self.alpha = alpha
        self.beta = beta
        self.axis = axis

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {"alpha": self.alpha, "beta": self.beta, "axis": self.axis}
        )
        return config


@keras_export("keras.losses.Circle")
class Circle(LossFunctionWrapper):
    """Computes Circle Loss between integer labels and L2-normalized embeddings.

    This is a metric learning loss designed to minimize within-class distance
    and maximize between-class distance in a flexible manner by dynamically
    adjusting the penalty strength based on optimization status of each
    similarity score.

    To use Circle Loss effectively, the model should output embeddings without
    an activation function (such as a `Dense` layer with `activation=None`)
    followed by UnitNormalization layer to ensure unit-norm embeddings.

    Args:
        gamma: Scaling factor that determines the largest scale of each
            similarity score. Defaults to `80`.
        margin: The relaxation factor, below this distance, negatives are
        up weighted and positives are down weighted. Similarly, above this
        distance negatives are down weighted and positive are up weighted.
            Defaults to `0.4`.
        remove_diagonal: Boolean, whether to remove self-similarities from the
            positive mask. Defaults to `True`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Examples:

    Usage with the `compile()` API:

    ```python
    model = models.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=None),  # No activation
        keras.layers.UnitNormalization()  # L2 normalization
    ])

    model.compile(optimizer="adam", loss=keras.losses.Circle())
    ```

    Reference:
    - [Yifan Sun et al., 2020](https://arxiv.org/abs/2002.10857)

    """

    def __init__(
        self,
        gamma=80.0,
        margin=0.4,
        remove_diagonal=True,
        reduction="sum_over_batch_size",
        name="circle",
        dtype=None,
    ):
        super().__init__(
            circle,
            name=name,
            reduction=reduction,
            dtype=dtype,
            gamma=gamma,
            margin=margin,
            remove_diagonal=remove_diagonal,
        )
        self.gamma = gamma
        self.margin = margin
        self.remove_diagonal = remove_diagonal

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "gamma": self.gamma,
                "margin": self.margin,
                "remove_diagonal": self.remove_diagonal,
            }
        )
        return config


@keras_export("keras.losses.CategoricalGeneralizedCrossEntropy")
class CategoricalGeneralizedCrossEntropy(LossFunctionWrapper):
    """Computes the Generalized Cross Entropy loss between `y_true` & `y_pred`.

    Generalized Cross Entropy (GCE) is a noise-robust loss function
    that provides better robustness against noisy labels than
    standard cross entropy.
    It generalizes both cross entropy and mean absolute error through
    the parameter q, where values closer to 1 make the loss more robust
    to noisy labels.

    Formula:
    ```python
    loss = (1 - p**q) / q
    ```
    where `p` is the predicted probability for the true class and `q`
    is the noise parameter.

    Args:
        q: Float in range `(0, 1)`. It is the noise parameter.
           Controls the behavior of the loss:
            - As `q` approaches 0: Behaves more like cross entropy
            - As `q` approaches 1: Behaves more like mean absolute error
           Defaults to `0.5`
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the loss instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Example:
    ```python
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]])
    keras.losses.CategoricalGeneralizedCrossEntropy()(y_true, y_pred)
    ```

    References:
        - [Zhang, Sabuncu, 2018](https://arxiv.org/abs/1805.07836)
          ("Generalized Cross Entropy Loss for Training
            Deep Neural Networks with Noisy Labels")
    """

    def __init__(
        self,
        q=0.5,
        reduction="sum_over_batch_size",
        name="categorical_generalized_cross_entropy",
        dtype=None,
    ):
        if not 0 < q < 1:
            raise ValueError("q must be in the interval (0, 1)")
        super().__init__(
            categorical_generalized_cross_entropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            q=q,
        )
        self.q = q

    def get_config(self):
        config = Loss.get_config(self)
        config.update(
            {
                "q": self.q,
            }
        )
        return config


def convert_binary_labels_to_hinge(y_true):
    """Converts binary labels into -1/1 for hinge loss/metric calculation."""
    are_zeros = ops.equal(y_true, 0)
    are_ones = ops.equal(y_true, 1)
    is_binary = ops.all((ops.logical_or(are_zeros, are_ones)))

    def _convert_binary_labels():
        # Convert the binary labels to -1 or 1.
        return 2.0 * y_true - 1.0

    def _return_labels_unconverted():
        # Returns the labels unchanged if they are non-binary
        return y_true

    updated_y_true = ops.cond(
        is_binary, _convert_binary_labels, _return_labels_unconverted
    )
    return updated_y_true


@keras_export(
    [
        "keras.metrics.hinge",
        "keras.losses.hinge",
    ]
)
def hinge(y_true, y_pred):
    """Computes the hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided they will be converted
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, dtype=y_pred.dtype)
    y_true = ops.convert_to_tensor(y_true)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(ops.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


@keras_export(
    [
        "keras.metrics.squared_hinge",
        "keras.losses.squared_hinge",
    ]
)
def squared_hinge(y_true, y_pred):
    """Computes the squared hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
    ```

    Args:
        y_true: The ground truth values. `y_true` values are expected to be -1
            or 1. If binary (0 or 1) labels are provided we will convert them
            to -1 or 1 with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Squared hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.choice([-1, 1], size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.squared_hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    y_true = convert_binary_labels_to_hinge(y_true)
    return ops.mean(
        ops.square(ops.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1
    )


@keras_export(
    [
        "keras.metrics.categorical_hinge",
        "keras.losses.categorical_hinge",
    ]
)
def categorical_hinge(y_true, y_pred):
    """Computes the categorical hinge loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = maximum(neg - pos + 1, 0)
    ```

    where `neg=maximum((1-y_true)*y_pred)` and `pos=sum(y_true*y_pred)`

    Args:
        y_true: The ground truth values. `y_true` values are expected to be
            either `{-1, +1}` or `{0, 1}` (i.e. a one-hot-encoded tensor) with
            shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Categorical hinge loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 3, size=(2,))
    >>> y_true = np.eye(np.max(y_true) + 1)[y_true]
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.categorical_hinge(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    pos = ops.sum(y_true * y_pred, axis=-1)
    neg = ops.max((1.0 - y_true) * y_pred, axis=-1)
    zero = ops.cast(0.0, y_pred.dtype)
    return ops.maximum(neg - pos + 1.0, zero)


@keras_export(
    [
        "keras.metrics.mean_squared_error",
        "keras.losses.mean_squared_error",
        # Legacy aliases
        "keras._legacy.losses.mse",
        "keras._legacy.losses.MSE",
        "keras._legacy.metrics.mse",
        "keras._legacy.metrics.MSE",
    ]
)
def mean_squared_error(y_true, y_pred):
    """Computes the mean squared error between labels and predictions.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred), axis=-1)
    ```

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_squared_error(y_true, y_pred)

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    return ops.mean(ops.square(y_true - y_pred), axis=-1)


@keras_export(
    [
        "keras.metrics.mean_absolute_error",
        "keras.losses.mean_absolute_error",
        # Legacy aliases
        "keras._legacy.losses.MAE",
        "keras._legacy.losses.mae",
        "keras._legacy.metrics.MAE",
        "keras._legacy.metrics.mae",
    ]
)
def mean_absolute_error(y_true, y_pred):
    """Computes the mean absolute error between labels and predictions.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_absolute_error(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    return ops.mean(ops.abs(y_true - y_pred), axis=-1)


@keras_export(
    [
        "keras.metrics.mean_absolute_percentage_error",
        "keras.losses.mean_absolute_percentage_error",
        # Legacy aliases
        "keras._legacy.losses.mape",
        "keras._legacy.losses.MAPE",
        "keras._legacy.metrics.mape",
        "keras._legacy.metrics.MAPE",
    ]
)
def mean_absolute_percentage_error(y_true, y_pred):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
    ```

    Division by zero is prevented by dividing by `maximum(y_true, epsilon)`
    where `epsilon = keras.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean absolute percentage error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:

    >>> y_true = np.random.random(size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_absolute_percentage_error(y_true, y_pred)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    epsilon = ops.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    diff = ops.abs((y_true - y_pred) / ops.maximum(ops.abs(y_true), epsilon))
    return 100.0 * ops.mean(diff, axis=-1)


@keras_export(
    [
        "keras.metrics.mean_squared_logarithmic_error",
        "keras.losses.mean_squared_logarithmic_error",
        # Legacy aliases
        "keras._legacy.losses.msle",
        "keras._legacy.losses.MSLE",
        "keras._legacy.metrics.msle",
        "keras._legacy.metrics.MSLE",
    ]
)
def mean_squared_logarithmic_error(y_true, y_pred):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
    ```

    Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative
    values and 0 values will be replaced with `keras.backend.epsilon()`
    (default to `1e-7`).

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared logarithmic error values with shape = `[batch_size, d0, ..
        dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    """
    epsilon = ops.convert_to_tensor(backend.epsilon())
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    first_log = ops.log(ops.maximum(y_pred, epsilon) + 1.0)
    second_log = ops.log(ops.maximum(y_true, epsilon) + 1.0)
    return ops.mean(ops.square(first_log - second_log), axis=-1)


@keras_export("keras.losses.cosine_similarity")
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:
    ```python
    loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Note that it is a number between -1 and 1. When it is a negative number
    between -1 and 0, 0 indicates orthogonality and values closer to -1
    indicate greater similarity. This makes it usable as a loss function in a
    setting where you try to maximize the proximity between predictions and
    targets. If either `y_true` or `y_pred` is a zero vector, cosine
    similarity will be 0 regardless of the proximity between predictions
    and targets.

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to `-1`.

    Returns:
        Cosine similarity tensor.

    Example:

    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
    [-0., -0.99999994, 0.99999994]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return -ops.sum(y_true * y_pred, axis=axis)


@keras_export(["keras.losses.huber", "keras.metrics.huber"])
def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.

    Formula:
    ```python
    for x in error:
        if abs(x) <= delta:
            loss.append(0.5 * x^2)
        elif abs(x) > delta:
            loss.append(delta * abs(x) - 0.5 * delta^2)

    loss = mean(loss, axis=-1)
    ```
    See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras.losses.huber(y_true, y_pred)
    0.155


    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear. Defaults to `1.0`.

    Returns:
        Tensor with one scalar loss entry per sample.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    delta = ops.convert_to_tensor(delta, dtype=y_pred.dtype)
    error = ops.subtract(y_pred, y_true)
    abs_error = ops.abs(error)
    half = ops.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return ops.mean(
        ops.where(
            abs_error <= delta,
            half * ops.square(error),
            delta * abs_error - half * ops.square(delta),
        ),
        axis=-1,
    )


@keras_export(
    [
        "keras.losses.log_cosh",
        "keras.metrics.log_cosh",
        # Legacy aliases
        "keras._legacy.losses.logcosh",
        "keras._legacy.metrics.logcosh",
    ]
)
def log_cosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    Formula:
    ```python
    loss = mean(log(cosh(y_pred - y_true)), axis=-1)
    ```

    Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small
    `x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works
    mostly like the mean squared error, but will not be so strongly affected by
    the occasional wildly incorrect prediction.

    Example:

    >>> y_true = [[0., 1.], [0., 0.]]
    >>> y_pred = [[1., 1.], [0., 0.]]
    >>> loss = keras.losses.log_cosh(y_true, y_pred)
    0.108

    Args:
        y_true: Ground truth values with shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values with shape = `[batch_size, d0, .. dN]`.

    Returns:
        Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    log2 = ops.convert_to_tensor(ops.log(2.0), dtype=y_pred.dtype)

    def _logcosh(x):
        return x + ops.softplus(x * -2.0) - log2

    return ops.mean(_logcosh(y_pred - y_true), axis=-1)


@keras_export(
    [
        "keras.metrics.kl_divergence",
        "keras.losses.kl_divergence",
        # Legacy aliases
        "keras._legacy.losses.KLD",
        "keras._legacy.losses.kld",
        "keras._legacy.losses.kullback_leibler_divergence",
        "keras._legacy.metrics.KLD",
        "keras._legacy.metrics.kld",
        "keras._legacy.metrics.kullback_leibler_divergence",
    ]
)
def kl_divergence(y_true, y_pred):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

    Formula:

    ```python
    loss = y_true * log(y_true / y_pred)
    ```

    `y_true` and `y_pred` are expected to be probability
    distributions, with values between 0 and 1. They will get
    clipped to the `[0, 1]` range.

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.

    Returns:
        KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.kl_divergence(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_true = ops.clip(y_true, 1e-7, 1)
    >>> y_pred = ops.clip(y_pred, 1e-7, 1)
    >>> assert np.array_equal(
    ...     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, y_pred.dtype)
    y_true = ops.clip(y_true, backend.epsilon(), 1)
    y_pred = ops.clip(y_pred, backend.epsilon(), 1)
    return ops.sum(y_true * ops.log(y_true / y_pred), axis=-1)


@keras_export(
    [
        "keras.metrics.poisson",
        "keras.losses.poisson",
    ]
)
def poisson(y_true, y_pred):
    """Computes the Poisson loss between y_true and y_pred.

    Formula:

    ```python
    loss = y_pred - y_true * log(y_pred)
    ```

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = np.random.randint(0, 2, size=(2, 3))
    >>> y_pred = np.random.random(size=(2, 3))
    >>> loss = keras.losses.poisson(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> y_pred = y_pred + 1e-7
    >>> assert np.allclose(
    ...     loss, np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
    ...     atol=1e-5)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    epsilon = ops.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)
    return ops.mean(y_pred - y_true * ops.log(y_pred + epsilon), axis=-1)


@keras_export(
    [
        "keras.metrics.categorical_crossentropy",
        "keras.losses.categorical_crossentropy",
    ]
)
def categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the categorical crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    _validate_label_smoothing(label_smoothing)

    if y_pred.shape is not None:
        axis = canonicalize_axis(axis, len(y_pred.shape))
        if y_pred.shape[axis] == 1:
            warnings.warn(
                "In loss categorical_crossentropy, expected "
                "y_pred.shape to be (batch_size, num_classes) "
                f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
                "Consider using 'binary_crossentropy' if you only "
                "have 2 classes.",
                SyntaxWarning,
                stacklevel=2,
            )

    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_pred)[axis], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    return ops.categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis
    )


@keras_export(
    [
        "keras.metrics.categorical_focal_crossentropy",
        "keras.losses.categorical_focal_crossentropy",
    ]
)
def categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the categorical focal crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.

    Returns:
        Categorical focal crossentropy loss value.

    Example:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = keras.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if y_pred.shape is not None:
        axis = canonicalize_axis(axis, len(y_pred.shape))
        if y_pred.shape[axis] == 1:
            warnings.warn(
                "In loss categorical_focal_crossentropy, expected "
                "y_pred.shape to be (batch_size, num_classes) "
                f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
                "Consider using 'binary_crossentropy' if you only "
                "have 2 classes.",
                SyntaxWarning,
                stacklevel=2,
            )

    if label_smoothing:
        num_classes = ops.cast(ops.shape(y_pred)[axis], y_pred.dtype)
        y_true = y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    if from_logits:
        y_pred = ops.softmax(y_pred, axis=axis)

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = y_pred / ops.sum(y_pred, axis=axis, keepdims=True)
    output = ops.clip(output, backend.epsilon(), 1.0 - backend.epsilon())

    # Calculate cross entropy
    cce = -y_true * ops.log(output)

    # Calculate factors
    modulating_factor = ops.power(1.0 - output, gamma)
    weighting_factor = ops.multiply(modulating_factor, alpha)

    # Apply weighting factor
    focal_cce = ops.multiply(weighting_factor, cce)
    focal_cce = ops.sum(focal_cce, axis=axis)
    return focal_cce


@keras_export("keras.losses.sparse_categorical_focal_crossentropy")
def sparse_categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    ignore_class=None,
    axis=-1,
):
    """Computes the sparse categorical focal crossentropy loss.

    Args:
        y_true: Tensor of integer true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes. Defaults to `0.25`.
            It can be a list of floats or a scalar. If a list is provided, it
            must have the same length as the number of classes.
        gamma: A focusing parameter. Defaults to `2.0`. It gradually reduces
            the importance given to easy examples. When `gamma=0`, there is no
            focal effect on the crossentropy loss.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, `y_pred` is expected to encode a probability distribution.
        ignore_class: Optional integer. The ID of a class to be ignored during
            loss computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255). By default
            (`ignore_class=None`), all classes are considered.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.

    Returns:
        Sparse categorical focal crossentropy loss value.

        An out-of-range, non-ignored label produces a `NaN` loss value. An
        `alpha` vector with a statically known wrong length raises `ValueError`;
        if its length or the class dimension is only known at runtime, a
        mismatch produces `NaN` loss values. This prevents backend-dependent
        index selection in compiled and symbolic execution, where Keras does
        not have a portable runtime assertion primitive.

    Example:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.sparse_categorical_focal_crossentropy(
    ...     y_true, y_pred
    ... )
    >>> assert loss.shape == (2,)
    >>> loss
    array([3.2058e-05, 4.6627e-01], dtype=float32)
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true)
    if len(y_pred.shape) < 1:
        raise ValueError(
            "Argument `y_pred` must be at least rank 1. "
            f"Received: y_pred.shape={y_pred.shape}"
        )

    axis = canonicalize_axis(axis, len(y_pred.shape))
    if len(y_true.shape) == len(y_pred.shape) and y_true.shape[axis] == 1:
        y_true = ops.squeeze(y_true, axis=axis)

    expected_y_true_shape = y_pred.shape[:axis] + y_pred.shape[axis + 1 :]
    if len(y_true.shape) != len(expected_y_true_shape) or any(
        true_dim is not None and pred_dim is not None and true_dim != pred_dim
        for true_dim, pred_dim in zip(y_true.shape, expected_y_true_shape)
    ):
        raise ValueError(
            "Arguments `y_true` and `y_pred` must have the same shape up "
            "until the class dimension: "
            f"y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}"
        )

    # Promote narrow labels before range checks. Casting the class count or an
    # arbitrary `ignore_class` to the original dtype can overflow at valid
    # boundaries such as 256 classes with uint8 labels. Preserve int64 inputs
    # so very large invalid labels are identified before the gather, while using
    # int32 otherwise to remain compatible with JAX's default dtype mode.
    if standardize_dtype(y_true.dtype) != "int64":
        y_true = ops.cast(y_true, "int32")

    valid_mask = None
    if ignore_class is not None:
        valid_mask = ops.not_equal(y_true, ignore_class)

    num_classes = ops.shape(y_pred)[axis]
    in_range = ops.greater_equal(y_true, 0)
    if num_classes is not None:
        in_range = ops.logical_and(
            in_range,
            ops.less(y_true, ops.cast(num_classes, dtype=y_true.dtype)),
        )
    if valid_mask is None:
        labels_valid = in_range
    else:
        labels_valid = ops.logical_or(ops.logical_not(valid_mask), in_range)

    # Gather only from valid indices. Invalid, non-ignored labels are changed
    # to NaN below. This ops-only behavior is consistent in eager, symbolic,
    # and compiled execution and avoids backend-specific assertions or host
    # synchronization in the loss hot path.
    safe_y_true = ops.where(in_range, y_true, 0)
    safe_y_true = ops.cast(safe_y_true, "int32")
    if from_logits:
        y_pred = ops.softmax(y_pred, axis=axis)

    output = y_pred / ops.sum(y_pred, axis=axis, keepdims=True)
    output = ops.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    true_class_prob = ops.take_along_axis(
        output, ops.expand_dims(safe_y_true, axis=axis), axis=axis
    )
    true_class_prob = ops.squeeze(true_class_prob, axis=axis)

    alpha = ops.convert_to_tensor(alpha, dtype=output.dtype)
    alpha_rank = len(alpha.shape)
    if alpha_rank not in (0, 1):
        raise ValueError(
            "`alpha` must be a scalar or a rank-1 tensor. "
            f"Received: alpha.shape={alpha.shape}"
        )

    alpha_length_valid = None
    if alpha_rank == 0:
        alpha_t = alpha
    else:
        alpha_length = ops.shape(alpha)[0]
        if (
            isinstance(alpha_length, int)
            and isinstance(num_classes, int)
            and alpha_length != num_classes
        ):
            raise ValueError(
                "When `alpha` is a list or rank-1 tensor, it must contain one "
                "value per class. "
                f"Received: alpha.shape={alpha.shape} and "
                f"y_pred.shape={y_pred.shape}"
            )
        elif alpha_length is not None and num_classes is not None:
            # Dynamic dimension(s).
            alpha_length_valid = ops.equal(alpha_length, num_classes)

        alpha_indices = safe_y_true
        if alpha_length is not None:
            alpha = ops.concatenate(
                [alpha, ops.zeros((1,), dtype=alpha.dtype)], axis=0
            )
            alpha_indices = ops.minimum(
                alpha_indices, ops.cast(alpha_length, dtype=alpha_indices.dtype)
            )
        alpha_t = ops.take(alpha, alpha_indices, axis=0)

    loss = ops.negative(
        ops.multiply(
            alpha_t,
            ops.multiply(
                ops.power(1.0 - true_class_prob, gamma),
                ops.log(true_class_prob),
            ),
        )
    )
    nan = ops.cast(float("nan"), dtype=loss.dtype)
    loss = ops.where(labels_valid, loss, nan)
    if alpha_length_valid is not None:
        loss = ops.where(alpha_length_valid, loss, nan)

    if valid_mask is not None:
        loss = ops.where(valid_mask, loss, 0.0)
        set_keras_mask(loss, mask=valid_mask)

    return loss


@keras_export(
    [
        "keras.metrics.sparse_categorical_crossentropy",
        "keras.losses.sparse_categorical_crossentropy",
    ]
)
def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    ignore_class=None,
    axis=-1,
    label_smoothing=0.0,
):
    """Computes the sparse categorical crossentropy loss.

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        ignore_class: Optional integer. The ID of a class to be ignored during
            loss computation. This is useful, for example, in segmentation
            problems featuring a "void" class (commonly -1 or 255) in
            segmentation maps. By default (`ignore_class=None`), all classes are
            considered.
        axis: Defaults to `-1`. The dimension along which the entropy is
            computed.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels. Defaults to
            `0.0`.

    Returns:
        Sparse categorical crossentropy loss value.

    Examples:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.0513, 2.303], dtype=float32)
    """

    if len(y_true.shape) == len(y_pred.shape) and y_true.shape[axis] == 1:
        y_true = ops.squeeze(y_true, axis=axis)

    if ignore_class is not None:
        # `res_shape` is the shape of the per-element loss: `y_pred.shape`
        # with the class axis removed.
        class_axis = canonicalize_axis(axis, len(y_pred.shape))
        y_pred_shape = ops.shape(y_pred)
        res_shape = tuple(
            d for i, d in enumerate(y_pred_shape) if i != class_axis
        )
        valid_mask = ops.not_equal(y_true, ops.cast(ignore_class, y_pred.dtype))
        y_true = ops.where(valid_mask, y_true, 0)
        y_pred = ops.where(ops.expand_dims(valid_mask, axis), y_pred, 0)

    res = ops.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=from_logits,
        axis=axis,
    )

    if label_smoothing > 0:
        # Smoothing the implied one-hot target splits the loss into the
        # hard-label term plus the crossentropy against a uniform target.
        # That second term has a closed form, so no dense target the size
        # of `y_pred` is built.
        if from_logits:
            uniform_res = ops.subtract(
                ops.logsumexp(y_pred, axis=axis), ops.mean(y_pred, axis=axis)
            )
        else:
            # The crossentropy ops normalize probabilities before taking the
            # log, so the same is done here.
            probs = ops.divide(
                y_pred, ops.sum(y_pred, axis=axis, keepdims=True)
            )
            probs = ops.clip(probs, backend.epsilon(), 1.0 - backend.epsilon())
            uniform_res = ops.negative(ops.mean(ops.log(probs), axis=axis))
        res = ops.add(
            ops.multiply(1.0 - label_smoothing, res),
            ops.multiply(label_smoothing, uniform_res),
        )

    if ignore_class is not None:
        valid_mask = ops.reshape(valid_mask, res_shape)
        res = ops.where(valid_mask, res, 0.0)
        set_keras_mask(res, mask=valid_mask)

    return res


@keras_export(
    [
        "keras.metrics.binary_crossentropy",
        "keras.losses.binary_crossentropy",
    ]
)
def binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the binary crossentropy loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.916 , 0.714], dtype=float32)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    _validate_label_smoothing(label_smoothing)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return ops.mean(
        ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    )


@keras_export(
    [
        "keras.metrics.binary_focal_crossentropy",
        "keras.losses.binary_focal_crossentropy",
    ]
)
def binary_focal_crossentropy(
    y_true,
    y_pred,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the binary focal crossentropy loss.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. By default, the focal tensor is computed as follows:

    `focal_factor = (1 - output) ** gamma` for class 1
    `focal_factor = output ** gamma` for class 0
    where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
    effect on the binary crossentropy loss.

    If `apply_class_balancing == True`, this function also takes into account a
    weight balancing factor for the binary classes 0 and 1 as follows:

    `weight = alpha` for class 1 (`target == 1`)
    `weight = 1 - alpha` for class 0
    where `alpha` is a float in the range of `[0, 1]`.

    Args:
        y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
        y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
        apply_class_balancing: A bool, whether to apply weight balancing on the
            binary classes 0 and 1.
        alpha: A weight balancing factor for class 1, default is `0.25` as
            mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary focal crossentropy loss value
        with shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # In this instance, the first sample in the second batch is the
    >>> # 'easier' example.
    >>> focal_loss = keras.losses.binary_focal_crossentropy(
    ...        y_true, y_pred, gamma=2)
    >>> assert loss.shape == (2,)
    >>> focal_loss
    array([0.330, 0.206], dtype=float32)
    >>> # Compare with binary_crossentropy
    >>> bce_loss = keras.losses.binary_focal_crossentropy(
    ...        y_true, y_pred)
    >>> bce_loss
    array([0.916, 0.714], dtype=float32)
    >>> # Binary focal crossentropy loss attributes more importance to the
    >>> # harder example which results in a higher loss for the first batch
    >>> # when normalized by binary cross entropy loss
    >>> focal_loss/bce_loss
    array([0.360, 0.289]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        y_pred = ops.sigmoid(y_pred)

    bce = ops.binary_crossentropy(
        target=y_true,
        output=y_pred,
        from_logits=False,
    )

    # Calculate focal factor
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = ops.power(1.0 - p_t, gamma)

    focal_bce = focal_factor * bce

    if apply_class_balancing:
        weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_bce = weight * focal_bce

    return ops.mean(focal_bce, axis=axis)


@keras_export("keras.losses.ctc")
def ctc(y_true, y_pred):
    """CTC (Connectionist Temporal Classification) loss.

    Args:
        y_true: A tensor of shape `(batch_size, max_length)` containing
            the true labels in integer format. `0` always represents
            the blank/mask index and should not be used for classes.
        y_pred: A tensor of shape `(batch_size, max_length, num_classes)`
            containing logits (the output of your model).
            They should *not* be normalized via softmax.
    """
    if len(ops.shape(y_true)) != 2:
        raise ValueError(
            "Targets `y_true` are expected to be a tensor of shape "
            "`(batch_size, max_length)` in integer format. "
            f"Received: y_true.shape={ops.shape(y_true)}"
        )
    if len(ops.shape(y_pred)) != 3:
        raise ValueError(
            "Logits `y_pred` are expected to be a tensor of shape "
            "`(batch_size, max_length, num_classes)`. "
            f"Received: y_pred.shape={ops.shape(y_pred)}"
        )

    mask_index = 0
    batch_length = ops.shape(y_pred)[0]
    input_length = ops.shape(y_pred)[1]
    input_length = input_length * ops.ones((batch_length,), dtype="int32")
    label_length = ops.cast(
        ops.sum(y_true != mask_index, axis=-1), dtype="int32"
    )

    return ops.ctc_loss(
        y_true, y_pred, label_length, input_length, mask_index=mask_index
    )


@keras_export("keras.losses.dice")
def dice(y_true, y_pred, axis=None):
    """Computes the Dice loss value between `y_true` and `y_pred`.

    Formula:
    ```python
    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    ```

    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        axis: tuple for which dimensions the loss is calculated

    Returns:
        Dice loss value.

    Example:

    >>> y_true = [[[[1.0], [1.0]], [[0.0], [0.0]]],
    ...           [[[1.0], [1.0]], [[0.0], [0.0]]]]
    >>> y_pred = [[[[0.0], [1.0]], [[0.0], [1.0]]],
    ...           [[[0.4], [0.0]], [[0.0], [0.9]]]]
    >>> axis = (1, 2, 3)
    >>> loss = keras.losses.dice(y_true, y_pred, axis=axis)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.5, 0.75757575], shape=(2,), dtype=float32)

    >>> loss = keras.losses.dice(y_true, y_pred)
    >>> assert loss.shape == ()
    >>> loss
    array(0.6164384, shape=(), dtype=float32)

    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    inputs = y_true
    targets = y_pred

    intersection = ops.sum(inputs * targets, axis=axis)
    dice = ops.divide(
        2.0 * intersection,
        ops.sum(y_true, axis=axis)
        + ops.sum(y_pred, axis=axis)
        + backend.epsilon(),
    )

    return 1 - dice


@keras_export("keras.losses.tversky")
def tversky(y_true, y_pred, alpha=0.5, beta=0.5, axis=None):
    """Computes the Tversky loss value between `y_true` and `y_pred`.

    This loss function is weighted by the alpha and beta coefficients
    that penalize false positives and false negatives.

    With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
    Dice Loss.

    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        alpha: coefficient controlling incidence of false positives.
        beta: coefficient controlling incidence of false negatives.
        axis: tuple for which dimensions the loss is calculated.

    Returns:
        Tversky loss value.

    Reference:

    - [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    inputs = y_pred
    targets = y_true

    intersection = ops.sum(inputs * targets, axis=axis)
    fp = ops.sum((1 - targets) * inputs, axis=axis)
    fn = ops.sum(targets * (1 - inputs), axis=axis)

    tversky = ops.divide(
        intersection,
        intersection + fp * alpha + fn * beta + backend.epsilon(),
    )

    return 1 - tversky


@keras_export("keras.losses.circle")
def circle(
    y_true,
    y_pred,
    ref_labels=None,
    ref_embeddings=None,
    remove_diagonal=True,
    gamma=80,
    margin=0.4,
):
    """Computes the Circle loss.

    It is designed to minimize within-class distances and maximize between-class
    distances in L2 normalized embedding space.

    Args:
        y_true: Tensor with ground truth labels in integer format.
        y_pred: Tensor with predicted L2 normalized embeddings.
        ref_labels: Optional integer tensor with labels for reference
            embeddings. If `None`, defaults to `y_true`.
        ref_embeddings: Optional tensor with L2 normalized reference embeddings.
            If `None`, defaults to `y_pred`.
        remove_diagonal: Boolean, whether to remove self-similarities from
            positive mask. Defaults to `True`.
        gamma: Float, scaling factor for the loss. Defaults to `80`.
        margin: Float, relaxation factor for the loss. Defaults to `0.4`.

    Returns:
        Circle loss value.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, "int32")
    ref_embeddings = (
        y_pred
        if ref_embeddings is None
        else ops.convert_to_tensor(ref_embeddings)
    )
    ref_labels = y_true if ref_labels is None else ops.cast(ref_labels, "int32")

    optim_pos = margin
    optim_neg = 1 + margin
    delta_pos = margin
    delta_neg = 1 - margin

    pairwise_cosine_distances = 1 - ops.matmul(
        y_pred, ops.transpose(ref_embeddings)
    )

    pairwise_cosine_distances = ops.maximum(pairwise_cosine_distances, 0.0)
    positive_mask, negative_mask = build_pos_neg_masks(
        y_true,
        ref_labels,
        remove_diagonal=remove_diagonal,
    )
    positive_mask = ops.cast(
        positive_mask, dtype=pairwise_cosine_distances.dtype
    )
    negative_mask = ops.cast(
        negative_mask, dtype=pairwise_cosine_distances.dtype
    )

    pos_weights = optim_pos + pairwise_cosine_distances
    pos_weights = pos_weights * positive_mask
    pos_weights = ops.maximum(pos_weights, 0.0)
    neg_weights = optim_neg - pairwise_cosine_distances
    neg_weights = neg_weights * negative_mask
    neg_weights = ops.maximum(neg_weights, 0.0)

    pos_dists = delta_pos - pairwise_cosine_distances
    neg_dists = delta_neg - pairwise_cosine_distances

    pos_wdists = -1 * gamma * pos_weights * pos_dists
    neg_wdists = gamma * neg_weights * neg_dists

    p_loss = ops.logsumexp(
        ops.where(positive_mask, pos_wdists, float("-inf")),
        axis=1,
    )
    n_loss = ops.logsumexp(
        ops.where(negative_mask, neg_wdists, float("-inf")),
        axis=1,
    )

    circle_loss = ops.softplus(p_loss + n_loss)
    set_keras_mask(circle_loss, circle_loss > 0)
    return circle_loss


@keras_export("keras.losses.categorical_generalized_cross_entropy")
def categorical_generalized_cross_entropy(y_true, y_pred, q):
    """Computes the Generalized Cross Entropy loss.

    Generalized Cross Entropy (GCE) is a noise-robust loss function that
    provides better robustness against noisy labels than standard cross entropy.
    It generalizes both cross entropy and mean absolute error through
    the parameter q, where values closer to 1 make the loss more robust
    to noisy labels.

    Formula:
    ```python
    loss = (1 - p**q) / q
    ```
    where `p` is the predicted probability for the true class and `q`
    is the noise parameter.

    Args:
        y_true: Ground truth labels. Expected to contain *integer class indices*
            with shape `[batch_size]` or `[batch_size, 1]`.
        y_pred: The predicted class probabilities, with shape
            `[batch_size, num_classes]`.
        q: Float in range `(0, 1)`. It is the noise parameter.
           Controls the behavior of the loss:
            - As `q` approaches 0: Behaves more like cross entropy
            - As `q` approaches 1: Behaves more like mean absolute error

    Returns:
        GCE loss values with shape `[batch_size]`.
    ```

    References:
        - [Zhang, Sabuncu, 2018](https://arxiv.org/abs/1805.07836)
          ("Generalized Cross Entropy Loss for Training
            Deep Neural Networks with Noisy Labels")
    """

    # Convert y_true to integer type and one-hot encode
    y_true_one_hot = ops.one_hot(
        ops.cast(y_true, "int"), num_classes=ops.shape(y_pred)[-1]
    )
    y_true_one_hot = ops.cast(y_true_one_hot, y_pred.dtype)
    # Calculate the probability of the true class
    p = ops.sum(y_pred * y_true_one_hot, axis=-1)

    # Compute the GCE loss for q in (0,1)
    gce_loss = (1 - ops.power(p, q)) / q

    return gce_loss
