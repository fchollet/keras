import warnings

from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.losses.losses import log_cosh
from keras.src.losses.losses import mean_absolute_error
from keras.src.losses.losses import mean_absolute_percentage_error
from keras.src.losses.losses import mean_squared_error
from keras.src.losses.losses import mean_squared_logarithmic_error
from keras.src.metrics import reduction_metrics
from keras.src.utils.numerical_utils import normalize


@keras_export("keras.metrics.MeanSquaredError")
class MeanSquaredError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = mean(square(y_true - y_pred))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:
    >>> m = keras.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25
    """

    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(fn=mean_squared_error, name=name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.MeanAbsoluteError")
class MeanAbsoluteError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean absolute error between the labels and predictions.

    Formula:

    ```python
    loss = mean(abs(y_true - y_pred))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.MeanAbsoluteError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError()])
    ```
    """

    def __init__(self, name="mean_absolute_error", dtype=None):
        super().__init__(mean_absolute_error, name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(reduction_metrics.MeanMetricWrapper):
    """Computes mean absolute percentage error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = 100 * mean(abs((y_true - y_pred) / y_true))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:
    >>> m = keras.metrics.MeanAbsolutePercentageError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    250000000.0

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    500000000.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanAbsolutePercentageError()])
    ```
    """

    def __init__(self, name="mean_absolute_percentage_error", dtype=None):
        super().__init__(mean_absolute_percentage_error, name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(reduction_metrics.MeanMetricWrapper):
    """Computes mean squared logarithmic error between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.MeanSquaredLogarithmicError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.12011322

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.24022643

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.MeanSquaredLogarithmicError()])
    ```
    """

    def __init__(self, name="mean_squared_logarithmic_error", dtype=None):
        super().__init__(mean_squared_logarithmic_error, name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_export("keras.metrics.RootMeanSquaredError")
class RootMeanSquaredError(reduction_metrics.Mean):
    """Computes root mean squared error metric between `y_true` and `y_pred`.

    Formula:

    ```python
    loss = sqrt(mean((y_pred - y_true) ** 2))
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.RootMeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.5

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.70710677

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError()])
    ```
    """

    def __init__(self, name="root_mean_squared_error", dtype=None):
        super().__init__(name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(y_true, self._dtype)
        y_pred = ops.convert_to_tensor(y_pred, self._dtype)
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        error_sq = ops.square(y_pred - y_true)
        return super().update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return ops.sqrt(super().result())


@keras_export("keras.metrics.CosineSimilarity")
class CosineSimilarity(reduction_metrics.MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.

    Formula:

    ```python
    loss = sum(l2_norm(y_true) * l2_norm(y_pred))
    ```
    See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) Defaults to `-1`. The dimension along which the cosine
            similarity is computed.

    Examples:

    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> m = keras.metrics.CosineSimilarity(axis=1)
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
    >>> m.result()
    0.49999997

    >>> m.reset_state()
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
    ...                sample_weight=[0.3, 0.7])
    >>> m.result()
    0.6999999

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[keras.metrics.CosineSimilarity(axis=1)])
    ```
    """

    def __init__(self, name="cosine_similarity", dtype=None, axis=-1):
        super().__init__(cosine_similarity, name, dtype=dtype, axis=axis)
        self.axis = axis
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype, "axis": self.axis}


@keras_export("keras.metrics.LogCoshError")
class LogCoshError(reduction_metrics.MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    Formula:

    ```python
    error = y_pred - y_true
    logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
    ```

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    >>> m = keras.metrics.LogCoshError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.10844523

    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result()
    0.21689045

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras.metrics.LogCoshError()])
    ```
    """

    def __init__(self, name="logcosh", dtype=None):
        super().__init__(log_cosh, name, dtype=dtype)
        # Metric should be minimized during optimization.
        self._direction = "down"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


# Adapted from TF-Addons implementation (RSquare class).
@keras_export("keras.metrics.R2Score")
class R2Score(reduction_metrics.Metric):
    """Computes R2 score.

    Formula:

    ```python
    sum_squares_residuals = sum((y_true - y_pred) ** 2)
    sum_squares = sum((y_true - mean(y_true)) ** 2)
    R2 = 1 - sum_squares_residuals / sum_squares
    ```

    This is also called the
    [coefficient of determination](
    https://en.wikipedia.org/wiki/Coefficient_of_determination).

    It indicates how close the fitted regression line
    is to ground-truth data.

    - The highest score possible is 1.0. It indicates that the predictors
        perfectly accounts for variation in the target.
    - A score of 0.0 indicates that the predictors do not
        account for variation in the target.
    - It can also be negative if the model is worse than random.

    This metric can also compute the "Adjusted R2" score.

    Args:
        class_aggregation: Specifies how to aggregate scores corresponding to
            different output classes (or target dimensions),
            i.e. different dimensions on the last axis of the predictions.
            Equivalent to `multioutput` argument in Scikit-Learn.
            Should be one of
            `None` (no aggregation), `"uniform_average"`,
            `"variance_weighted_average"`.
        num_regressors: Number of independent regressors used
            ("Adjusted R2" score). 0 is the standard R2 score.
            Defaults to `0`.
        name: Optional. string name of the metric instance.
        dtype: Optional. data type of the metric result.

    Example:

    >>> y_true = np.array([[1], [4], [3]], dtype=np.float32)
    >>> y_pred = np.array([[2], [4], [4]], dtype=np.float32)
    >>> metric = keras.metrics.R2Score()
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result
    0.57142854
    """

    def __init__(
        self,
        class_aggregation="uniform_average",
        num_regressors=0,
        name="r2_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        valid_class_aggregation_values = (
            None,
            "uniform_average",
            "variance_weighted_average",
        )
        if class_aggregation not in valid_class_aggregation_values:
            raise ValueError(
                "Invalid value for argument `class_aggregation`. Expected "
                f"one of {valid_class_aggregation_values}. "
                f"Received: class_aggregation={class_aggregation}"
            )
        if num_regressors < 0:
            raise ValueError(
                "Invalid value for argument `num_regressors`. "
                "Expected a value >= 0. "
                f"Received: num_regressors={num_regressors}"
            )
        self.class_aggregation = class_aggregation
        self.num_regressors = num_regressors
        self.num_samples = self.add_variable(
            shape=(),
            initializer=initializers.Zeros(),
            name="num_samples",
        )
        self._built = False

    def _build(self, y_true_shape, y_pred_shape):
        if len(y_pred_shape) != 2 or len(y_true_shape) != 2:
            raise ValueError(
                "R2Score expects 2D inputs with shape "
                "(batch_size, output_dim). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        if y_pred_shape[-1] is None or y_true_shape[-1] is None:
            raise ValueError(
                "R2Score expects 2D inputs with shape "
                "(batch_size, output_dim), with output_dim fully "
                "defined (not None). Received input "
                f"shapes: y_pred.shape={y_pred_shape} and "
                f"y_true.shape={y_true_shape}."
            )
        num_classes = y_pred_shape[-1]
        self.squared_sum = self.add_variable(
            name="squared_sum",
            shape=[num_classes],
            initializer=initializers.Zeros(),
        )
        self.sum = self.add_variable(
            name="sum",
            shape=[num_classes],
            initializer=initializers.Zeros(),
        )
        self.total_mse = self.add_variable(
            name="residual",
            shape=[num_classes],
            initializer=initializers.Zeros(),
        )
        self.count = self.add_variable(
            name="count",
            shape=[num_classes],
            initializer=initializers.Zeros(),
        )
        self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.

        Args:
            y_true: The ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional weighting of each example. Can
                be a `Tensor` whose rank is either 0, or the same rank as
                `y_true`, and must be broadcastable to `y_true`.
                Defaults to `1`.

        Returns:
            Update op.
        """
        y_true = ops.convert_to_tensor(y_true, dtype=self._dtype)
        y_pred = ops.convert_to_tensor(y_pred, dtype=self._dtype)
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        if not self._built:
            self._build(y_true.shape, y_pred.shape)

        if sample_weight is None:
            sample_weight = 1

        sample_weight = ops.convert_to_tensor(sample_weight, dtype=self.dtype)

        if len(sample_weight.shape) == 1:
            # Make sure there's a features dimension
            sample_weight = ops.expand_dims(sample_weight, axis=1)

        sample_weight = ops.broadcast_to(sample_weight, ops.shape(y_true))

        sample_weight = ops.cast(sample_weight, y_true.dtype)
        weighted_y_true = y_true * sample_weight
        self.sum.assign(self.sum + ops.sum(weighted_y_true, axis=0))
        self.squared_sum.assign(
            self.squared_sum + ops.sum(y_true * weighted_y_true, axis=0)
        )
        self.total_mse.assign(
            self.total_mse
            + ops.sum(
                (y_true - y_pred) ** 2 * sample_weight,
                axis=0,
            )
        )
        self.count.assign(self.count + ops.sum(sample_weight, axis=0))
        # Count non-zero samples. We count a sample if it has a
        # non-zero weight in at least one output. This avoids N x K inflation
        # for multi-output regression and ensures num_samples is an integer.
        is_nonzero = ops.not_equal(sample_weight, 0.0)
        nonzero_per_sample = ops.any(is_nonzero, axis=-1)
        num_samples_update = ops.sum(ops.cast(nonzero_per_sample, self.dtype))
        self.num_samples.assign_add(num_samples_update)

    def result(self):
        mean = self.sum / self.count
        total = self.squared_sum - self.sum * mean
        # Branch on the state variables themselves (matching sklearn's
        # `force_finite` check on its raw numerator/denominator) rather than
        # on properties of the computed ratio: a NaN in total_mse can also
        # come from unrelated numerical instability (e.g. exploding
        # gradients), and checking isnan(raw_scores) can't tell that case
        # apart from the deliberate 0/0 of a zero-variance perfect
        # prediction. It would silently report a perfect score instead of
        # surfacing the NaN.
        safe_total = ops.where(ops.equal(total, 0.0), 1.0, total)
        raw_scores = 1.0 - (self.total_mse / safe_total)
        raw_scores = ops.where(ops.equal(total, 0.0), 0.0, raw_scores)
        raw_scores = ops.where(ops.equal(self.total_mse, 0.0), 1.0, raw_scores)
        raw_scores = ops.where(ops.isinf(raw_scores), 0.0, raw_scores)

        if self.class_aggregation == "uniform_average":
            r2_score = ops.mean(raw_scores)
        elif self.class_aggregation == "variance_weighted_average":
            weighted_sum = ops.sum(total * raw_scores)
            sum_of_weights = ops.sum(total)
            r2_score = weighted_sum / sum_of_weights
        else:
            r2_score = raw_scores

        if self.num_regressors != 0:
            if self.num_regressors > self.num_samples - 1:
                warnings.warn(
                    "More independent predictors than datapoints "
                    "in adjusted R2 score. Falling back to standard R2 score.",
                    stacklevel=2,
                )
            elif self.num_regressors == self.num_samples - 1:
                warnings.warn(
                    "Division by zero in Adjusted R2 score. "
                    "Falling back to standard R2 score.",
                    stacklevel=2,
                )
            else:
                n = ops.convert_to_tensor(self.num_samples, dtype="float32")
                p = ops.convert_to_tensor(self.num_regressors, dtype="float32")
                num = ops.multiply(
                    ops.subtract(1.0, r2_score), ops.subtract(n, 1.0)
                )
                den = ops.subtract(ops.subtract(n, p), 1.0)
                r2_score = ops.subtract(1.0, ops.divide(num, den))
        return r2_score

    def reset_state(self):
        for v in self.variables:
            v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        config = {
            "name": self.name,
            "dtype": self.dtype,
            "class_aggregation": self.class_aggregation,
            "num_regressors": self.num_regressors,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@keras_export("keras.metrics.cosine_similarity")
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:

    ```python
    similarity = sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to `-1`.

    Returns:
        Cosine similarity tensor.

    Example:

    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> sim = keras.metrics.cosine_similarity(y_true, y_pred, axis=-1)
    [0., 0.99999994, -0.99999994]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return ops.sum(y_true * y_pred, axis=axis)
