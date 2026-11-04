"""The run scope of the calibration modes: one pass over one dataset.

GPTQ and AWQ share one driver, `CalibrationRun`. What differs between the
modes is declared, not coded twice: the calibrator class comes from the
mode's `CalibrationStrategy`, the forward batch size from its config, and
the undersampling threshold from the calibrator.
"""

import math
import warnings
from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
from absl import logging

from keras.src import backend
from keras.src import ops
from keras.src import utils as keras_utils
from keras.src.layers import Dense
from keras.src.layers import EinsumDense
from keras.src.quantizers.capture import calibration_scope
from keras.src.quantizers.utils import should_quantize_layer


def calibration_no_grad_scope():
    """Returns a context manager that disables gradient tracking.

    Calibration is inference-only: it runs forward passes to accumulate
    statistics and then assigns quantized values to variables. On the torch
    backend those forwards would otherwise build autograd graphs, and
    because per-sample activations are retained across the whole
    calibration loop (and AWQ stashes activation samples for its clipping
    search), the graphs and every intermediate activation stay alive until
    the block completes - enough to exhaust GPU memory on models that fit
    comfortably otherwise. JAX and TensorFlow build no such graphs in eager
    mode, so this is a no-op there.
    """
    if backend.backend() == "torch":
        import torch

        return torch.no_grad()
    return nullcontext()


# Seed of the offset of the first calibration window.
_WINDOW_OFFSET_SEED = 42


def get_dataloader(tokenizer, sequence_length, dataset, num_samples=128):
    """Cuts `num_samples` token windows out of the calibration dataset.

    The dataset is tokenized into one token stream, repeated if it is too
    short, and windows of `sequence_length` tokens are taken at a regular
    stride from a fixed offset. All processing happens on the CPU.
    Documents are joined with no separator, so a window can span two of
    them, and the start offsets wrap around the end of the token stream.

    Args:
        tokenizer: The tokenizer to use for text splitting.
        sequence_length: The length of each input sequence.
        dataset: The dataset to sample from.
        num_samples: The number of samples to generate.

    Returns:
        np.ndarray of shape (num_samples, 1, sequence_length), dtype int32.
    """
    if not hasattr(dataset, "__iter__") or isinstance(dataset, (str, bytes)):
        raise TypeError(
            "The `dataset` argument must be an iterable (e.g., a list of "
            "strings, a generator, or a NumPy array). Got type: "
            f"{type(dataset).__name__}. Please pass the loaded dataset "
            "directly."
        )

    dataset_list = list(dataset)
    if not dataset_list:
        raise ValueError("Provided dataset is empty.")

    if isinstance(dataset_list[0], str):
        dataset_list = [tokenizer.tokenize(s) for s in dataset_list]
    all_tokens = np.concatenate(
        [ops.convert_to_numpy(s).reshape(-1) for s in dataset_list]
    ).astype(np.int32, copy=False)

    required_tokens = num_samples * sequence_length
    if all_tokens.size < required_tokens:
        repeats = math.ceil(required_tokens / max(1, all_tokens.size))
        all_tokens = np.tile(all_tokens, repeats)

    max_start = all_tokens.size - sequence_length
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens to form one sample of length {sequence_length} "
            f"(have {all_tokens.size})."
        )

    # A stride that covers the stream roughly uniformly, from an offset
    # derived from a fixed seed. Python's `hash()` must not be used here:
    # it is randomized per process, so the windows, and every quantization
    # result, would differ between runs.
    stride = max(1, (max_start + 1) // num_samples)
    offset = int(
        np.random.default_rng(_WINDOW_OFFSET_SEED).integers(0, max_start + 1)
    )
    starts = (offset + np.arange(num_samples, dtype=np.int64) * stride) % (
        max_start + 1
    )
    # `sliding_window_view` avoids building a big index matrix.
    windows = np.lib.stride_tricks.sliding_window_view(
        all_tokens, sequence_length
    )
    return windows[starts][:, None, :]


def find_layers_in_block(block):
    """
    Finds all Dense and EinsumDense layers in a transformer block.

    Args:
        block: A Keras layer representing a transformer block.
    Returns:
        A dict mapping layer paths to the corresponding Dense or EinsumDense
    """
    found_layers = {}
    for sub_layer in block._flatten_layers():
        # A quantizable layer may own sub-layers (e.g. a `Layer`
        # activation), so no leaf filtering here — collect every Dense/
        # EinsumDense reachable inside the block.
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[sub_layer.path] = sub_layer
    return found_layers


def _execution_stages(block, layers, batch):
    """Groups a block's layers into sequential quantization stages.

    Reference GPTQ implementations quantize a block's sub-layers in
    topological order ("true sequential"): the statistics of a layer are
    taken after the layers upstream of it are quantized, so its solve is
    computed against the activations it sees at inference.

    One forward pass of `batch` records the order the layers first run in
    and the input each consumes. Consecutive layers that consume the same
    input tensor (the query/key/value projections, an MLP's gate/up pair)
    share a stage, since quantizing one cannot change the others' inputs.
    The layers that do not run in that pass form the last stage.

    Args:
        block: The block the layers belong to.
        layers: The layers to group.
        batch: One batch of the block's inputs.

    Returns:
        List of lists of layers, one list per stage, in execution order.
    """
    trace = {}

    def recorder(layer):
        def record(inputs):
            # The first call only. The trace keeps the input alive, so its
            # identity cannot be reused within the pass.
            trace.setdefault(id(layer), (layer, inputs))

        return record

    with calibration_scope({layer: recorder(layer) for layer in layers}):
        block(batch)
    stages, stage_input = [], None
    for layer, inputs in trace.values():
        if not stages or inputs is not stage_input:
            stages.append([])
        stages[-1].append(layer)
        stage_input = inputs
    untraced = [layer for layer in layers if id(layer) not in trace]
    if untraced:
        stages.append(untraced)
    return stages


# The keys of a layer structure. A structure without `pre_block_layers`
# hands the token ids to the first block.
_STRUCTURE_KEYS = ("pre_block_layers", "sequential_blocks")
_ACCEPTED_KEYS = "'sequential_blocks' and, optionally, 'pre_block_layers'"


def _read_layer_structure(structure):
    """Returns the pre-block layers and the blocks of a layer structure.

    A layer structure is a dict with the key `"sequential_blocks"`, the
    blocks to calibrate in order, and optionally `"pre_block_layers"`, the
    layers that map the token ids to the inputs of the first block. Each
    value is a list or a tuple of layers. Any other key is refused, so a
    misspelled key raises instead of being dropped: a structure with
    `"pre_block_layer"` would feed the raw token ids to the blocks.

    Args:
        structure: The layer structure.

    Returns:
        `(pre_block_layers, blocks)`, two lists.
    """
    if not isinstance(structure, Mapping):
        raise ValueError(
            "The quantization layer structure must be a dict with the keys "
            f"{_ACCEPTED_KEYS}. Received an object of type "
            f"`{type(structure).__name__}`."
        )
    unknown = [key for key in structure if key not in _STRUCTURE_KEYS]
    if unknown:
        raise ValueError(
            f"The quantization layer structure has unknown keys {unknown}. "
            f"It takes the keys {_ACCEPTED_KEYS}. Received the keys "
            f"{list(structure)}."
        )
    if "sequential_blocks" not in structure:
        raise ValueError(
            "The quantization layer structure has no 'sequential_blocks' "
            f"key. It takes the keys {_ACCEPTED_KEYS}. Received the keys "
            f"{list(structure)}."
        )
    for key, value in structure.items():
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"The '{key}' of the quantization layer structure must be a "
                "list or a tuple of layers. Received an object of type "
                f"`{type(value).__name__}`."
            )
    return (
        list(structure.get("pre_block_layers", [])),
        list(structure["sequential_blocks"]),
    )


class CalibrationRun:
    """One calibration pass of a model's sequential blocks over a dataset.

    `Model.quantize` resolves the layer structure and the mode's
    `CalibrationStrategy` creates the run for it (`calibrate`). The
    constructor checks the blocks, the dataset and the tokenizer, and
    builds the dataloader.

    `run()` materializes the activations behind the prefix layers once,
    then walks the blocks in order. For each block it groups the layers
    this mode left pending into execution-order stages ("true
    sequential"). For each stage in turn it builds a calibrator per
    layer, from the run's config, passes the block's inputs through the
    block to them, and quantizes the stage's layers, so a later stage
    observes the quantized output of the stages before it. It then runs
    the calibrated block to produce the next block's inputs, up to the
    last block with a pending layer. The activations are kept as the
    batches the blocks run on. A run with no pending layer does nothing.

    Args:
        strategy: The `CalibrationStrategy` of the mode. It supplies the
            calibrator class.
        config: The mode's config. It holds the dataset and the
            tokenizer, the solver parameters, and the number of samples
            and the batch size of the run.
        structure: Dict with keys `"pre_block_layers"` and
            `"sequential_blocks"`.
        filters: Optional filters that exclude layers from quantization.
    """

    def __init__(self, strategy, config, structure, filters=None):
        self.strategy = strategy
        self.config = config
        self.filters = filters
        self.pre_block_layers, self.blocks = _read_layer_structure(structure)
        if not self.blocks:
            raise ValueError(
                "No sequential blocks found in the provided structure to "
                "quantize."
            )
        if config.dataset is None or config.tokenizer is None:
            raise ValueError(
                f"{strategy.name.upper()} quantization requires a dataset "
                "and a tokenizer. Please provide them in the "
                f"`{strategy.config_cls.__name__}`."
            )
        self.dataloader = get_dataloader(
            config.tokenizer,
            config.sequence_length,
            config.dataset,
            num_samples=config.num_samples,
        )
        # AWQ calibrates one sample at a time; its config has no batch size.
        self.batch_size = int(getattr(config, "calibration_batch_size", 1))
        # Layers whose statistics saw too few calibration tokens relative
        # to their input width, and layers that observed no input,
        # collected across all blocks for one warning each.
        self.undersampled = []
        self.unreached = []

    def run(self):
        """Calibrates and quantizes every block, in order."""
        pending = [self._pending_layers(block) for block in self.blocks]
        if not any(pending):
            logging.info("No layers are pending calibration. Skipping.")
            return
        # The blocks after the last one with a pending layer need no inputs.
        last = max(i for i, layers in enumerate(pending) if layers)
        logging.info("Starting model quantization...")
        with calibration_no_grad_scope():
            inputs = self._prefix_outputs(self.dataloader)
            progbar = keras_utils.Progbar(target=last + 1)
            for block_idx, block in enumerate(self.blocks[: last + 1]):
                logging.info(f"Quantizing Block {block_idx}")
                self._calibrate_block(block_idx, block, inputs)
                if block_idx < last:
                    logging.info(
                        f"Generating inputs for block {block_idx + 1}..."
                    )
                    inputs = self._next_inputs(block, inputs)
                progbar.update(current=block_idx + 1)
        self._warn_undersampled()
        self._warn_unreached()
        logging.info("Quantization process complete.")

    def _prefix_outputs(self, dataloader):
        # The pre-block layers run one sample at a time; their outputs are
        # stacked into the batches the blocks run on.
        outputs = []
        for batch in dataloader:
            batch = ops.convert_to_tensor(batch, dtype="int32")
            for layer in self.pre_block_layers:
                batch = layer(batch)
            outputs.append(batch)
        return [
            ops.concatenate(outputs[start : start + self.batch_size], axis=0)
            for start in range(0, len(outputs), self.batch_size)
        ]

    def _pending_layers(self, block):
        # Only the layers this mode left pending: a layer quantized in
        # another mode, or already calibrated, has no float kernel to solve.
        return {
            name: layer
            for name, layer in find_layers_in_block(block).items()
            if layer.quantization_mode == self.strategy.name
            and layer.calibration_pending
            and should_quantize_layer(layer, self.filters)
        }

    def _calibrator(self, layer):
        return self.strategy.calibrator_cls(layer, self.config)

    def _calibrate_block(self, block_idx, block, inputs):
        layers = self._pending_layers(block)
        if not layers:
            logging.info(
                f"  No quantizable layers found in block {block_idx}. Skipping."
            )
            return
        logging.info(f"Found layers: {list(layers)}")
        # Quantize the block's layers in execution-order stages ("true
        # sequential", as in reference GPTQ): a stage observes the block
        # after the stages before it are quantized, so its solves are
        # computed against the activations the layers see at inference.
        stages = _execution_stages(block, list(layers.values()), inputs[0])
        for stage in stages:
            self._calibrate_stage(block, stage, inputs)

    def _calibrate_stage(self, block, layers, inputs):
        # The statistics live for this call only, so the run holds those
        # of one stage at a time.
        calibrators = [self._calibrator(layer) for layer in layers]
        with calibration_scope({c.layer: c.observe for c in calibrators}):
            for batch in inputs:
                block(batch)
        for calibrator in calibrators:
            name = calibrator.layer.path
            if calibrator.num_samples:
                self._tally_undersampling(name, calibrator)
            else:
                self.unreached.append(name)
            logging.info(f"Quantizing {name}...")
            calibrator.quantize()

    def _tally_undersampling(self, name, calibrator):
        threshold = calibrator.warn_tokens_per_row
        if threshold is None:
            return
        tokens = int(calibrator.num_samples)
        rows = int(calibrator.rows)
        if tokens < threshold * rows:
            self.undersampled.append((name, tokens, rows))

    def _next_inputs(self, block, inputs):
        # Each batch's output goes to the next block as it is: the first
        # output of a block that returns several.
        next_inputs = []
        for batch in inputs:
            output = block(batch)
            if isinstance(output, (list, tuple)):
                output = output[0]
            next_inputs.append(output)
        return next_inputs

    def _warn_undersampled(self):
        if not self.undersampled:
            return
        warnings.warn(
            self.strategy.calibrator_cls.undersampling_warning(
                self.undersampled
            ),
            stacklevel=2,
        )

    def _warn_unreached(self):
        if not self.unreached:
            return
        warnings.warn(
            f"{self.strategy.name.upper()} calibration observed no input "
            f"for {len(self.unreached)} layer(s), so their weights are "
            "quantized without calibration statistics: "
            f"{', '.join(self.unreached)}. A layer that the blocks do not "
            "run on the calibration data, such as a layer that runs only "
            "in training, is never observed. To keep such a layer in "
            "float, exclude it with `filters`.",
            stacklevel=2,
        )
