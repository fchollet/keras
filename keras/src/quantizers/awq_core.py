"""AWQ core functionality for layer-wise quantization.

This module provides the orchestration logic for applying AWQ quantization
to transformer models in a layer-by-layer fashion.
"""

from contextlib import contextmanager

from absl import logging

from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.quantizers.awq import AWQCalibrator
from keras.src.quantizers.calibration_run import _execution_stages
from keras.src.quantizers.calibration_run import calibration_no_grad_scope
from keras.src.quantizers.calibration_run import find_layers_in_block
from keras.src.quantizers.calibration_run import get_dataloader
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.utils import should_quantize_layer


@contextmanager
def stream_activations(layers_map, awq_objects, execution_trace=None):
    """Context manager to capture activations for AWQ calibration.

    Registers a calibration capture (`keras.src.quantizers.capture`) on
    each layer, which the dispatch machinery runs before every forward
    pass, whichever forward that is, to capture activation statistics for
    computing per-channel scaling factors. Nothing on the layers is
    rebound.

    Args:
        layers_map: Dict[str, Layer]. Mapping from layer names to layers.
        awq_objects: Dict[str, AWQCalibrator]. Mapping from names to AWQ
            calibrators.
        execution_trace: Optional dict. When provided, each layer's FIRST
            hook invocation records `{name: (call_index, input_tensor)}`
            into it - the block-level execution order and the identity of
            the input each layer consumes. Used to derive within-block
            quantization stages (see `apply_awq_layerwise`). The recorded
            tensors are only kept alive for identity comparison; callers
            should drop the trace after use.

    Yields:
        None: The captures are active only within the `with` block.
    """
    call_counter = [0]

    def create_capture(name):
        def capture(inp):
            if execution_trace is not None and name not in execution_trace:
                # Record block-level execution order and the input tensor's
                # identity on the first call (a live reference is kept so
                # the id cannot be recycled while tracing).
                execution_trace[name] = (call_counter[0], inp)
                call_counter[0] += 1
            awq_objects[name].observe(inp)

        return capture

    with calibration_scope(
        {layer: create_capture(name) for name, layer in layers_map.items()}
    ):
        yield


def apply_awq_layerwise(dataloader, config, structure, filters=None):
    """Apply AWQ quantization layer-by-layer to a Keras model.

    This function processes the model sequentially, one block at a time:
    1. Captures activation statistics through calibration data forward pass
    2. Uses activation magnitudes to determine weight saliency
    3. Finds optimal per-channel scales via grid search
    4. Quantizes weights with AWQ scaling

    Args:
        dataloader: Calibration data as numpy array.
        config: AWQConfig instance.
        structure: Dict with 'pre_block_layers' and 'sequential_blocks'.
        filters: Optional layer filters.
    """
    num_samples = config.num_samples
    logging.info("Starting AWQ quantization...")

    pre_layers = structure.get("pre_block_layers", [])
    transformer_blocks = structure.get("sequential_blocks", [])

    if not transformer_blocks:
        raise ValueError(
            "No sequential blocks found in the structure to quantize."
        )

    # Process inputs through pre-block layers (e.g., embedding)
    inputs = []
    for batch in dataloader:
        batch = ops.convert_to_tensor(batch, dtype="int32")
        for layer in pre_layers:
            batch = layer(batch)
        inputs.append(batch)

    num_samples = min(num_samples, len(inputs))
    progbar = keras_utils.Progbar(target=len(transformer_blocks))

    for block_idx, block in enumerate(transformer_blocks):
        logging.info(f"Quantizing Block {block_idx}")
        sub_layers_map = find_layers_in_block(block)

        # Apply filters
        final_sub_layers_map = {}
        for name, layer in sub_layers_map.items():
            if not should_quantize_layer(layer, filters):
                continue
            final_sub_layers_map[name] = layer

        sub_layers_map = final_sub_layers_map

        if not sub_layers_map:
            logging.info(
                f"  No quantizable layers found in block {block_idx}. Skipping."
            )
        else:
            logging.info(f"Found layers: {list(sub_layers_map.keys())}")

            # Create an AWQ calibrator for each layer
            awq_objects = {
                name: AWQCalibrator(layer, config)
                for name, layer in sub_layers_map.items()
            }

            def run_calibration_sweep(layers, objects, trace=None):
                with stream_activations(layers, objects, execution_trace=trace):
                    for sample_idx in range(num_samples):
                        current_input = inputs[sample_idx]
                        if ops.ndim(current_input) == 2:
                            current_input = ops.expand_dims(
                                current_input, axis=0
                            )
                        _ = block(current_input)

            # Capture activation statistics
            execution_trace = {}
            run_calibration_sweep(
                sub_layers_map, awq_objects, trace=execution_trace
            )

            # Quantize the block's layers in execution-order stages ("true
            # sequential", matching the GPTQ path): after each stage is
            # quantized, downstream stages' activation statistics are
            # re-estimated so their scale searches see the quantized
            # upstream activations they will actually get at inference.
            stages = _execution_stages(sub_layers_map, execution_trace)
            del execution_trace

            for stage_idx, stage_names in enumerate(stages):
                if stage_idx > 0:
                    stage_map = {
                        name: sub_layers_map[name] for name in stage_names
                    }
                    stage_objects = {}
                    for name in stage_names:
                        # Drop the stale statistics before allocating new ones.
                        del awq_objects[name]
                        awq_objects[name] = AWQCalibrator(
                            sub_layers_map[name], config
                        )
                        stage_objects[name] = awq_objects[name]
                    run_calibration_sweep(stage_map, stage_objects)

                for name in stage_names:
                    logging.info(f"Quantizing {name}...")
                    awq_objects.pop(name).quantize()

            del awq_objects

        # Generate inputs for next block
        if block_idx < len(transformer_blocks) - 1:
            logging.info(f"Generating inputs for block {block_idx + 1}...")
            next_block_inputs = []
            for sample_idx in range(num_samples):
                current_input = inputs[sample_idx]
                if len(current_input.shape) == 2:
                    current_input = ops.expand_dims(current_input, axis=0)
                output = block(current_input)[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

        progbar.update(current=block_idx + 1)

    logging.info("AWQ quantization complete.")


def awq_quantize(config, quantization_layer_structure, filters=None):
    """Main entry point for AWQ quantization.

    Args:
        config: AWQConfig instance.
        quantization_layer_structure: Model structure dictionary.
        filters: Optional layer filters.
    """
    if config.dataset is None or config.tokenizer is None:
        raise ValueError(
            "AWQ quantization requires a dataset and tokenizer. "
            "Please provide them in the AWQConfig."
        )

    if quantization_layer_structure is None:
        raise ValueError(
            "For 'awq' mode, a valid quantization structure must be provided "
            "either via `config.quantization_layer_structure` or by overriding "
            "`model.get_quantization_layer_structure(mode)`. The structure "
            "should be a dictionary with keys 'pre_block_layers' and "
            "'sequential_blocks'."
        )

    # Load calibration data
    dataloader = get_dataloader(
        config.tokenizer,
        config.sequence_length,
        config.dataset,
        num_samples=config.num_samples,
    )

    with calibration_no_grad_scope():
        apply_awq_layerwise(
            dataloader[: config.num_samples],
            config,
            quantization_layer_structure,
            filters=filters,
        )
