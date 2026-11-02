from keras.src import backend
from keras.src import dtype_policies
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import get_keras_mask
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils.naming import auto_name


@keras_export(["keras.Loss", "keras.losses.Loss"])
class Loss(KerasSaveable):
    """Loss base class.

    This is the class to subclass in order to create new custom losses.

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

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation using `y_true`,
        `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            return ops.mean(ops.square(y_pred - y_true), axis=-1)
    ```
    """

    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=None):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)
        self._dtype_policy = dtype_policies.get(dtype or backend.floatx())
        self._dtype = self._dtype_policy.compute_dtype

    @property
    def dtype(self):
        return self._dtype

    def __call__(self, y_true, y_pred, sample_weight=None):
        in_mask = get_keras_mask(y_pred)

        with backend.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
            )
            y_true = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true
            )

            losses = self.call(y_true, y_pred)
            out_mask = get_keras_mask(losses)

            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None

            return reduce_weighted_values(
                losses,
                sample_weight=sample_weight,
                mask=mask,
                reduction=self.reduction,
                dtype=self.dtype,
            )

    def call(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name, "reduction": self.reduction}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _obj_type(self):
        return "Loss"


def standardize_reduction(reduction):
    allowed = {
        "sum_over_batch_size",
        "sum",
        None,
        "none",
        "mean",
        "mean_with_sample_weight",
    }
    if reduction not in allowed:
        raise ValueError(
            "Invalid value for argument `reduction`. "
            f"Expected one of {allowed}. Received: "
            f"reduction={reduction}"
        )
    return reduction


def squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = ops.expand_dims(x2, axis=-1)
            else:
                x1 = ops.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = ops.expand_dims(x1, axis=-1)
            else:
                x2 = ops.squeeze(x2, axis=-1)
    return x1, x2


def reduce_values(values, sample_weight=None, reduction="sum_over_batch_size"):
    if (
        reduction is None
        or reduction == "none"
        or tuple(values.shape) == ()
        or tuple(values.shape) == (0,)
    ):
        return values
    loss = ops.sum(values)
    if reduction in ("sum_over_batch_size", "mean", "mean_with_sample_weight"):
        if reduction == "mean_with_sample_weight" and sample_weight is not None:
            divisor = ops.cast(ops.sum(sample_weight), loss.dtype)
        else:
            divisor = ops.cast(
                ops.prod(
                    ops.convert_to_tensor(ops.shape(values), dtype="int32")
                ),
                loss.dtype,
            )
        loss = ops.divide_no_nan(loss, divisor)
        loss = scale_loss_for_distribution(loss)
    return loss


def reduce_weighted_values(
    values,
    sample_weight=None,
    mask=None,
    reduction="sum_over_batch_size",
    dtype=None,
):
    reduction = standardize_reduction(reduction)

    values = ops.convert_to_tensor(values, dtype=dtype)
    if sample_weight is not None:
        sample_weight = ops.convert_to_tensor(sample_weight, dtype=dtype)
    if mask is not None:
        mask = ops.convert_to_tensor(mask, dtype=dtype)

    # Merge mask and sample weight into sample weight.
    sample_weight = apply_mask(
        sample_weight, mask, dtype=values.dtype, reduction=reduction
    )

    if sample_weight is not None:
        sample_weight = ops.cast(sample_weight, values.dtype)
        # Update dimensions of `sample_weight` to match `losses`.
        values, sample_weight = squeeze_or_expand_to_same_rank(
            values, sample_weight
        )
        values = values * sample_weight

    # Apply reduction function to the individual weighted losses.
    loss = reduce_values(values, sample_weight, reduction)
    return loss


def apply_mask(sample_weight, mask, dtype, reduction):
    """Applies any mask on predictions to sample weights."""
    if mask is not None:
        if reduction in ("mean", "sum_over_batch_size"):
            # Valid entries have weight `total/valid`, while invalid ones
            # have 0. When summed over batch, they will be reduced to:
            #
            # mean(loss * sample_weight * total / valid)
            #   = sum(loss * sample_weight * total / valid) / total
            #   = sum(loss * sample_weight) / total * total / valid
            #   = sum(loss * sample_weight) / valid
            total = ops.cast(
                ops.prod(ops.convert_to_tensor(ops.shape(mask), dtype="int32")),
                dtype,
            )
            valid = ops.sum(ops.cast(mask, dtype=dtype))  # May be 0!
            mask_weight = ops.divide_no_nan(total, valid)
            float_mask = ops.where(ops.cast(mask, "bool"), mask_weight, 0)
        else:
            float_mask = ops.cast(mask, dtype=dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, dtype=dtype)
            float_mask, sample_weight = squeeze_or_expand_to_same_rank(
                float_mask, sample_weight
            )
            sample_weight *= float_mask
        else:
            sample_weight = float_mask
    return sample_weight


def scale_loss_for_distribution(value):
    """Scales the given value by the number of replicas in the strategy.

    Currently, this function is only effective when using the tensorflow backend
    and `tf.distribute`.
    """
    if backend.backend() == "tensorflow":
        import tensorflow as tf

        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        if num_replicas > 1:
            value = ops.multiply(
                value, ops.cast(1.0 / num_replicas, value.dtype)
            )
    return value


def unscale_loss_for_distribution(value):
    """Unscales the given value by the number of replicas in the strategy.

    Currently, this function is only effective when using the tensorflow backend
    and `tf.distribute`.
    """
    if backend.backend() == "tensorflow":
        import tensorflow as tf

        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        if num_replicas > 1:
            value = ops.multiply(value, ops.cast(num_replicas, value.dtype))
    return value
