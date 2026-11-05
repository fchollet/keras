import functools
import warnings

from keras.src import layers
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel import tensor_layout


def analyze_dense_layer(layer, expansion_threshold=1.5):
    """Classifies a Dense layer based on its input/output dimensions.

    This function uses a heuristic to determine if a Dense layer acts as an
    'up_projection' (expansion), a 'down_projection' (contraction), or a
    standard 'dense' layer.

    Args:
        layer: The Keras Dense layer instance to analyze.
        expansion_threshold: Multiplier threshold to classify
            expansion/contraction.

    Returns:
        str: One of 'up_projection', 'down_projection', or 'dense'.
    """
    input_dim = None
    output_dim = None

    kernel = getattr(layer, "kernel", getattr(layer, "_kernel", None))
    if kernel is not None:
        if len(kernel.shape) == 2:
            input_dim = kernel.shape[0]
            output_dim = kernel.shape[1]
        elif len(kernel.shape) == 3:
            # For EinsumDense with 3D kernels (e.g., (E, N, H))
            # Input is usually the first dimension.
            input_dim = kernel.shape[0]
            output_dim = kernel.shape[1] * kernel.shape[2]

    if output_dim is None and hasattr(layer, "units"):
        output_dim = layer.units

    if (
        input_dim is None
        and hasattr(layer, "input_shape")
        and layer.input_shape
    ):
        input_shape = layer.input_shape
        if (
            isinstance(input_shape, list)
            and input_shape
            and not isinstance(input_shape[0], int)
        ):
            input_shape = input_shape[0]
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 1:
            input_dim = input_shape[-1]

    if input_dim is None or output_dim is None:
        return "dense"

    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return "up_projection"
    elif is_contraction:
        return "down_projection"
    else:
        return "dense"


def _reduce_sum(x):
    """Performs an all-reduce sum operation across the 'model' mesh axis."""
    return distribution_lib.all_reduce(x, op="sum", axis_name="model")


def _gather(x, axis):
    """Performs an all-gather operation across the 'model' mesh axis."""
    return distribution_lib.all_gather(x, axis=axis, axis_name="model")


def _get_variable_key(variable):
    """Get a stable key for a variable, preferring path if available."""
    # Keras Variables have a stable 'path' attribute once built into a model.
    # Paths are preferred as they are stable across serialization.
    if hasattr(variable, "path") and variable.path:
        return variable.path
    return id(variable)


def _get_layer_key(layer):
    """Get a stable key for a layer, preferring path if available."""
    if hasattr(layer, "path") and layer.path:
        return layer.path
    return id(layer)


def _apply_layer_sharding_rules(
    layer, device_count, state_rules, output_rules, expansion_threshold=1.5
):
    """Applies sharding rules to a single layer based on its type.

    Args:
        layer: The Keras layer instance to configure.
        device_count: The number of devices available for tensor parallelism.
        state_rules: Dictionary mapping variable paths/IDs to sharding
        functions.
        output_rules: Dictionary mapping layer paths to output communication
        functors.
        expansion_threshold: Threshold to classify Dense layers.
    """
    if not getattr(layer, "built", False):
        return

    def split_rule(dim):
        return functools.partial(
            tensor_layout.split_tensor_for_parallelism,
            device_count=device_count,
            dim=dim,
        )

    def gather_rule(axis):
        return functools.partial(_gather, axis=axis)

    layer_key = _get_layer_key(layer)

    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer, expansion_threshold)

        if mlp_type == "up_projection":
            state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[_get_variable_key(layer.bias)] = split_rule(dim=0)
            # Column-parallel (up) usually requires no gathering if succeeded by
            # Row-parallel (down), keeping it gather-free for Megatron chaining.

        elif mlp_type == "down_projection":
            state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=0)
            if layer.use_bias:
                warnings.warn(
                    f"Layer {layer_key} is classified as a down-projection "
                    "but has 'use_bias=True'. In row-parallel sharding, "
                    "adding a replicated bias before reduction is "
                    "mathematically incorrect (it will be summed N times). "
                    "Ensure your model handles bias after all-reduce."
                )
            output_rules[layer_key] = _reduce_sum

        else:
            state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[_get_variable_key(layer.bias)] = split_rule(dim=0)
            output_rules[layer_key] = gather_rule(axis=-1)

    elif isinstance(layer, layers.EinsumDense):
        # Heuristic for Attention Projections vs MLP
        # WARNING: Relying on string heuristics like name/equation is brittle.
        if "attention_output" in layer.name:  # Contraction / Row-Parallel
            state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=0)
            if hasattr(layer, "bias") and layer.bias is not None:
                warnings.warn(
                    f"EinsumDense layer {layer_key} (attention_output) "
                    "has a bias. Row-parallel sharding requires handling "
                    "bias after all-reduce."
                )
            output_rules[layer_key] = _reduce_sum
        elif (
            "h" in layer.equation.split("->")[1]
            or "attention" in layer.name
            or "query" in layer.name
            or "key" in layer.name
            or "value" in layer.name
        ):
            # Expansion / Column-Parallel for Query/Key/Value
            state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[_get_variable_key(layer.bias)] = split_rule(dim=0)
        else:
            # Generic EinsumDense (like FFN in some models)
            mlp_type = analyze_dense_layer(layer, expansion_threshold)
            if mlp_type == "up_projection":
                state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=1)
                if hasattr(layer, "bias") and layer.bias is not None:
                    state_rules[_get_variable_key(layer.bias)] = split_rule(
                        dim=0
                    )
            elif mlp_type == "down_projection":
                state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=0)
                if hasattr(layer, "bias") and layer.bias is not None:
                    warnings.warn(
                        f"EinsumDense layer {layer_key} (down_projection) "
                        "has a bias. Row-parallel sharding requires handling "
                        "bias after all-reduce."
                    )
                output_rules[layer_key] = _reduce_sum
            else:
                state_rules[_get_variable_key(layer.kernel)] = split_rule(dim=1)
                if hasattr(layer, "bias") and layer.bias is not None:
                    state_rules[_get_variable_key(layer.bias)] = split_rule(
                        dim=0
                    )
                output_rules[layer_key] = gather_rule(axis=-1)

    elif (
        isinstance(layer, layers.Embedding)
        or "Embedding" in layer.__class__.__name__
        or hasattr(layer, "embeddings")
    ):
        embeddings_var = getattr(layer, "embeddings", None)
        if embeddings_var is None:
            # Try to find it in weights by path suffix
            embeddings_var = next(
                (
                    w
                    for w in getattr(layer, "weights", [])
                    if getattr(w, "path", "").endswith("/embeddings")
                ),
                None,
            )

        if embeddings_var is not None:
            # Shard along the embedding dimension (Column-parallel equivalence)
            # to avoid out-of-bounds errors during lookup with standard
            # Embedding layers.
            state_rules[_get_variable_key(embeddings_var)] = split_rule(dim=1)
            # All-gather to reconstruct the full embedding dimension
            output_rules[layer_key] = gather_rule(axis=-1)
        else:
            warnings.warn(
                f"Embedding layer {layer_key} found but embeddings variable "
                "unidentified."
            )

    elif isinstance(layer, layers.Dropout):
        # Parallel RNG handling for dropout
        output_rules[layer_key] = "parallel_dropout"


def get_default_config(model, device_ids, expansion_threshold=1.5):
    """Generates a default tensor parallelism configuration for a model.

    This function traverses the model's layer hierarchy and
    automatically generates a `LayoutMap`. This map contains:
    1.  `state_rules`: How to shard the weights of supported layers.
    2.  `output_rules`: How to synchronize outputs of these layers.

    Args:
        model: The Keras model to configure.
        device_ids: A list of device identifiers to use for distribution.
        expansion_threshold: Threshold to classify Dense layers.

    Returns:
        ParallelLayoutMap: Configuration populated with `state_rules`
        and `output_rules`.
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}

    for layer in model._flatten_layers(recursive=True, include_self=True):
        _apply_layer_sharding_rules(
            layer, device_count, state_rules, output_rules, expansion_threshold
        )

    return tensor_layout.ParallelLayoutMap(
        state_rules=state_rules, output_rules=output_rules
    )
