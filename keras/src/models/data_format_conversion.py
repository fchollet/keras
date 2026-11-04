"""Convert built/pretrained models between channels_first and channels_last."""

import warnings

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers import Input
from keras.src.layers.layer import Layer
from keras.src.models.cloning import clone_model
from keras.src.models.functional import Functional
from keras.src.models.sequential import Sequential

# Layer class names whose configs carry a literal `data_format` field that
# this function knows how to flip.
_DATA_FORMAT_LAYERS = {
    "Conv1D",
    "Conv2D",
    "Conv3D",
    "Conv1DTranspose",
    "Conv2DTranspose",
    "Conv3DTranspose",
    "DepthwiseConv1D",
    "DepthwiseConv2D",
    "SeparableConv1D",
    "SeparableConv2D",
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "GlobalMaxPooling1D",
    "GlobalMaxPooling2D",
    "GlobalMaxPooling3D",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "AdaptiveMaxPooling1D",
    "AdaptiveMaxPooling2D",
    "AdaptiveMaxPooling3D",
    "AdaptiveAveragePooling1D",
    "AdaptiveAveragePooling2D",
    "AdaptiveAveragePooling3D",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "Flatten",
    "Resizing",
    "Rescaling",
    "CenterCrop",
}

# Layers whose channel axis is recorded as an `axis` field that needs to be
# flipped between -1 (channels_last) and 1 (channels_first).
_CHANNEL_AXIS_LAYERS = {
    "BatchNormalization",
    "GroupNormalization",
    "LayerNormalization",
    "RMSNormalization",
    "UnitNormalization",
    "SpectralNormalization",
}

# Layers whose semantics depend on data_format in a way this function
# cannot rewrite automatically. We warn so the user knows to audit.
_AMBIGUOUS_LAYERS = {
    "Reshape",
    "Permute",
    "Concatenate",
    "Lambda",
}


def _flipped_data_format(target):
    if target == "channels_last":
        return "channels_first"
    return "channels_last"


def _move_channel_dim(shape, to_format):
    """Move the channel dim of `shape` to the position implied by `to_format`.

    `shape` is a tuple/list of dims including the leading batch dim (`None`
    typically). For 4D inputs, channels_last is `(N, H, W, C)` and
    channels_first is `(N, C, H, W)`. For rank-3 (1D conv inputs) and rank-5
    (3D conv inputs) the same rule applies — position 1 vs position -1.
    """
    shape = list(shape)
    if len(shape) < 3:
        return tuple(shape)
    if to_format == "channels_first":
        # Move last → position 1.
        channel = shape.pop(-1)
        shape.insert(1, channel)
    else:
        # Move position 1 → last.
        channel = shape.pop(1)
        shape.append(channel)
    return tuple(shape)


def _flip_axis_for_target(axis, source_rank, target_data_format):
    """Compute the channel axis for `target_data_format`.

    The new axis is `1` for `channels_first` and `-1` for `channels_last`.
    `source_rank` is only used to normalize the original `axis` if the caller
    needs it; the new axis is the same regardless of rank.
    """
    del axis, source_rank  # Retained for future use.
    return 1 if target_data_format == "channels_first" else -1


def _make_clone_function(target_data_format, ambiguous_seen):
    def clone_function(layer):
        config = layer.get_config()
        cls_name = layer.__class__.__name__

        if cls_name in _DATA_FORMAT_LAYERS and "data_format" in config:
            config["data_format"] = target_data_format
            return layer.__class__.from_config(config)

        if cls_name in _CHANNEL_AXIS_LAYERS and "axis" in config:
            # If the original `axis` is the channel axis (the common case),
            # remap it to the new channel position. We treat any single-int
            # axis as the channel axis; a tuple/list `axis` is normalization
            # over multiple dims and is left untouched.
            if isinstance(config["axis"], int):
                config["axis"] = _flip_axis_for_target(
                    config["axis"], None, target_data_format
                )
            return layer.__class__.from_config(config)

        if cls_name in _AMBIGUOUS_LAYERS:
            ambiguous_seen.add(cls_name)

        return layer.__class__.from_config(config)

    return clone_function


@keras_export("keras.models.convert_data_format")
def convert_data_format(model, target_data_format):
    """Convert a built or pretrained model between channels_first and
    channels_last.

    Walks every layer of a `Sequential` or `Functional` model, flips the
    `data_format` (or channel `axis`) of each layer that carries one,
    rebuilds the model with the appropriately reshaped input, and copies the
    original weights over. Layer weights themselves are unchanged — only the
    layer-level data-format is rewired and the input/output channel position
    is moved.

    Args:
        model: A built `Sequential` or `Functional` model. Subclassed models
            are not supported.
        target_data_format: `"channels_last"` or `"channels_first"`. If the
            model is already in this format, it is returned unchanged.

    Returns:
        A new `Model` whose layers all use `target_data_format` and whose
        input/output shapes have the channel dim moved accordingly.

    Example:

    ```python
    # `m` was trained with channels_last (the Keras default).
    m_cf = keras.models.convert_data_format(m, "channels_first")

    x_cl = np.random.uniform(size=(1, 224, 224, 3))
    x_cf = np.transpose(x_cl, (0, 3, 1, 2))

    y_cl = m(x_cl)
    y_cf = m_cf(x_cf)
    # `y_cl` and `y_cf` agree up to a channel-axis transpose.
    ```

    Notes:
    - Layers whose data-format dependence cannot be inferred from the config
      alone — `Reshape`, `Permute`, `Concatenate`, `Lambda` — are cloned as
      is and a warning is emitted listing the affected layer classes. Audit
      any such layers manually after conversion.
    - Subclassed models (those that override `__init__` / `call` with custom
      logic) are not supported because their internal structure is opaque to
      the cloner.
    """
    if target_data_format not in ("channels_last", "channels_first"):
        raise ValueError(
            "`target_data_format` must be one of `'channels_last'` or "
            f"`'channels_first'`. Received: {target_data_format!r}"
        )
    if not isinstance(model, (Sequential, Functional)):
        raise TypeError(
            "`convert_data_format` only supports Sequential or Functional "
            f"models. Received: model of type {type(model).__name__}"
        )

    # Detect the source data format from the first layer that carries it.
    source = _detect_source_format(model)
    if source == target_data_format:
        return model

    ambiguous_seen = set()
    clone_function = _make_clone_function(target_data_format, ambiguous_seen)

    # `clone_model` does not run `clone_function` on Input/InputLayer
    # instances, so build replacement inputs ourselves with the channel dim
    # moved to the right position. The `input_tensors` argument shape that
    # `clone_model` expects differs between Sequential (single tensor) and
    # Functional (matches `model.input` structure).
    def _rebuilt_input(inp):
        new_shape = tuple(
            _move_channel_dim(
                (None,) + tuple(inp.shape[1:]), target_data_format
            )
        )[1:]
        return Input(shape=new_shape, dtype=inp.dtype)

    if isinstance(model, Sequential):
        new_inputs = _rebuilt_input(model.inputs[0])
    else:
        new_inputs = tree.map_structure(_rebuilt_input, model.input)

    new_model = clone_model(
        model, clone_function=clone_function, input_tensors=new_inputs
    )

    # Copy weights: layer configs are equivalent in shape (Keras conv kernels
    # don't depend on data_format and BN/LN/GN gammas/betas are 1D), so a
    # straight one-to-one copy works.
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        if old_layer.weights:
            new_layer.set_weights(old_layer.get_weights())

    if ambiguous_seen:
        warnings.warn(
            "convert_data_format encountered layer types whose data-format "
            "dependence cannot be inferred from the config: "
            f"{sorted(ambiguous_seen)}. These layers were cloned as is — "
            "audit them manually to confirm the conversion is correct.",
            stacklevel=2,
        )
    return new_model


def _detect_source_format(model):
    for layer in model.layers:
        cfg = layer.get_config()
        df = cfg.get("data_format")
        if df in ("channels_last", "channels_first"):
            return df
    # Fall back to the channel position implied by the input shape.
    try:
        shape = tuple(model.input.shape)
    except (AttributeError, ValueError):
        return None
    if len(shape) >= 3:
        # Heuristic: in channels_first, position 1 is a small integer
        # (typically <= 8). When in doubt we default to channels_last.
        return None
    return None


def _is_layer(obj):
    return isinstance(obj, Layer)
