import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers import strategy_registry
from keras.src.quantizers.calibration_run import _stack_calibration_batch
from keras.src.quantizers.calibration_run import find_layers_in_block
from keras.src.quantizers.calibration_run import get_dataloader
from keras.src.quantizers.calibration_run import gptq_quantize
from keras.src.quantizers.calibration_run import stream_hessians
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import tiny_calibration_model
from keras.src.quantizers.quantization_test_utils import token_dataset
from keras.src.utils.rng_utils import set_random_seed

VOCAB_SIZE = 100


class MockTokenizer:
    """A mock tokenizer that mimics the real API for testing."""

    def tokenize(self, text):
        return [ord(c) % VOCAB_SIZE for c in "".join(text)]

    def __call__(self, text):
        return self.tokenize(text)


class PassthroughTokenizer:
    """A tokenizer for datasets that are already token ids."""

    def tokenize(self, x):
        return np.asarray(x, dtype=np.int32)


class EmptyBlock(layers.Layer):
    """A block that contains no quantizable layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()

    def call(self, inputs):
        return self.ln(inputs)


class TransformerBlock(layers.Layer):
    """A toy transformer block with a quantizable Dense layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(128)

    def call(self, inputs):
        return self.dense(inputs)


def _get_model_with_backbone(
    has_transformer_layers=True, embedding_name="embedding"
):
    """Creates a KerasHub-style model with a backbone."""

    class Backbone(layers.Layer):
        def __init__(self, vocab_size, embedding_dim=128, **kwargs):
            super().__init__(**kwargs)
            # Use direct assignment
            setattr(
                self,
                embedding_name,
                layers.Embedding(vocab_size, embedding_dim),
            )

            # Keep track of layers in a list for the call method
            self.transformer_layers = []
            if has_transformer_layers:
                self.transformer_layers.append(TransformerBlock())

        def call(self, inputs):
            x = getattr(self, embedding_name)(inputs)
            for layer in self.transformer_layers:
                x = layer(x)
            return x

    class Model(models.Model):
        def __init__(self, vocab_size, **kwargs):
            super().__init__(**kwargs)
            # Pass configuration directly
            self.backbone = Backbone(vocab_size=vocab_size)
            self.classifier = layers.Dense(1, activation="sigmoid")

        def call(self, inputs):
            x = self.backbone(inputs)
            x = layers.GlobalAveragePooling1D()(x)
            return self.classifier(x)

    model = Model(vocab_size=VOCAB_SIZE)
    rng = np.random.default_rng(seed=42)
    dummy_input = rng.normal(loc=0, scale=1, size=(2, 64)).astype(np.float32)

    _ = model(dummy_input)
    return model


@pytest.mark.requires_trainable_backend
class TestGPTQCore(testing.TestCase):
    def test_shape_and_dtype_strings(self):
        """Test the shape and dtype of the output for string inputs."""
        tok = MockTokenizer()
        dataset = ["a b c d e f g", "h i j k"]
        seq_len, n = 5, 7

        out = get_dataloader(tok, seq_len, dataset, num_samples=n)
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_shape_and_dtype_pretokenized(self):
        """Test the shape and dtype of the output for pre-tokenized inputs."""
        # Pre-tokenized inputs; mixed shapes (1, L) and (L,)
        seqs = [
            np.array([[1, 2, 3, 4]], dtype=np.int64),
            np.array([5, 6], dtype=np.int64),
        ]
        tok = MockTokenizer()
        seq_len, n = 3, 4

        out = get_dataloader(tok, seq_len, seqs, num_samples=n)
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_get_dataloader_error_scenarios(self):
        """Tests error cases for get_dataloader."""
        with pytest.raises(ValueError, match="Provided dataset is empty"):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset=[],
                num_samples=10,
            )
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be an iterable.*Got type: str.*"
            "Please pass the loaded dataset directly.",
        ):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset="wikitext2",
                num_samples=10,
            )

    def test_calibration_batching_produces_identical_hessian(self):
        """The Hessian accumulated during calibration must be identical
        whether calibration samples are streamed one at a time or in
        batches. `GPTQCalibrator.observe` flattens activations to
        `[-1, features]`, so batching only changes the number of forward
        passes, not the math."""
        d_model = 16
        seq_len = 8
        num_samples = 8

        block = TransformerBlock()  # contains a single Dense(128) layer.
        # Trigger a build so the inner Dense layer's kernel exists.
        _ = block(ops.zeros((1, seq_len, d_model)))

        rng = np.random.default_rng(0)
        samples = [
            ops.convert_to_tensor(
                rng.standard_normal((1, seq_len, d_model)).astype("float32")
            )
            for _ in range(num_samples)
        ]

        def accumulate(batch_size):
            layers_map = find_layers_in_block(block)
            gptq_objects = {
                name: calibrate_layer(
                    layer, calibration_config("gptq"), solve=False
                )
                for name, layer in layers_map.items()
            }
            with stream_hessians(layers_map, gptq_objects):
                for start in range(0, num_samples, batch_size):
                    batch = _stack_calibration_batch(
                        samples[start : start + batch_size]
                    )
                    _ = block(batch)
            return {name: obj.hessian for name, obj in gptq_objects.items()}

        unbatched = accumulate(batch_size=1)
        batched = accumulate(batch_size=4)

        self.assertEqual(set(unbatched.keys()), set(batched.keys()))
        self.assertGreater(len(unbatched), 0)
        for name in unbatched:
            self.assertAllClose(
                unbatched[name], batched[name], rtol=1e-5, atol=1e-5
            )

    def test_apply_gptq_on_multi_block_model(self):
        """Tests quantization on a model with multiple blocks."""
        model = models.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, 128),
                TransformerBlock(),
                TransformerBlock(),
            ]
        )
        model.build(input_shape=(None, 10))

        layer_structure = {
            "pre_block_layers": [model.layers[0]],
            "sequential_blocks": [model.layers[1], model.layers[2]],
        }

        config = GPTQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(),
            group_size=32,
            quantization_layer_structure=layer_structure,
        )
        model.quantize("gptq", config=config)

    @parameterized.named_parameters(
        (
            "no_embedding_layer",
            models.Sequential([layers.Dense(10)]),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "no_transformer_blocks",
            models.Sequential(
                [layers.Embedding(VOCAB_SIZE, 10), layers.Dense(10)]
            ),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "backbone_no_layers",
            _get_model_with_backbone(has_transformer_layers=False),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "backbone_no_embedding",
            _get_model_with_backbone(embedding_name="wrong_name"),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
    )
    def test_apply_gptq_with_unsupported_architectures(
        self, model, error_message
    ):
        """Tests that quantize fails correctly for various unsupported
        model architectures."""
        if not model.built:
            model.build(input_shape=(None, 10))

        config = GPTQConfig(dataset=["test"], tokenizer=MockTokenizer())
        with self.assertRaisesRegex(ValueError, error_message):
            # We pass None as structure to trigger the error
            gptq_quantize(config, quantization_layer_structure=None)


class TestExecutionStages(testing.TestCase):
    def test_stages_group_by_shared_input_and_order(self):
        from keras.src.quantizers.calibration_run import _execution_stages

        x_attn, x_out, x_mlp, x_down = (
            object(),
            object(),
            object(),
            object(),
        )
        trace = {
            "query": (0, x_attn),
            "key": (1, x_attn),
            "value": (2, x_attn),
            "attention_output": (3, x_out),
            "gate": (4, x_mlp),
            "up": (5, x_mlp),
            "down": (6, x_down),
        }
        stages = _execution_stages(list(trace), trace)
        self.assertEqual(
            stages,
            [
                ["query", "key", "value"],
                ["attention_output"],
                ["gate", "up"],
                ["down"],
            ],
        )

    def test_untraced_layers_join_first_stage(self):
        from keras.src.quantizers.calibration_run import _execution_stages

        x = object()
        trace = {"a": (0, x)}
        stages = _execution_stages(["ghost", "a"], trace)
        self.assertEqual(stages, [["ghost", "a"]])

    def test_no_trace_single_stage(self):
        from keras.src.quantizers.calibration_run import _execution_stages

        stages = _execution_stages(["a", "b"], {})
        self.assertEqual(stages, [["a", "b"]])


class TestDataloaderReproducibility(testing.TestCase):
    def test_strided_offset_is_process_stable(self):
        """The strided sampling offset must not depend on PYTHONHASHSEED.

        The offset was previously derived via `hash(("gptq-calib", seed))`;
        Python randomizes string hashing per process, so calibration
        windows - and therefore every quantization result - silently
        differed between runs despite the fixed seed. These golden values
        pin the numpy-based derivation (seed=42, 1000 tokens, seq len 8,
        4 samples); the old hash-based offset only reproduces them under
        one specific PYTHONHASHSEED by coincidence.
        """
        out = get_dataloader(
            PassthroughTokenizer(),
            8,
            [np.arange(1000, dtype=np.int32)],
            num_samples=4,
        )
        self.assertEqual(out.shape, (4, 1, 8))
        self.assertAllClose(out[:, 0, 0], np.array([88, 336, 584, 832]))
        self.assertAllClose(out[0, 0], np.arange(88, 96))

    def test_windows_wrap_around_the_end_of_the_stream(self):
        # 33 tokens, 16 windows of 2: stride 2 from offset 2. The last
        # window would start at 32, past the last valid start (31), so it
        # wraps to 0.
        out = get_dataloader(
            PassthroughTokenizer(), 2, [np.arange(33)], num_samples=16
        )
        self.assertEqual(out[:, 0, 0].tolist(), [*range(2, 31, 2), 0])
        self.assertAllClose(out[:, 0, 1], out[:, 0, 0] + 1)


def _tiny_model(mode, num_samples=4, dtype=None, **kwargs):
    """Embedding -> block -> pooled head.

    The block is `[Dense(16, relu), Dense(8)]`.
    """
    set_random_seed(123)
    seq_len, vocab_size, embed_dim = 16, 48, 8
    block_layers = [
        layers.Dense(16, activation="relu", dtype=dtype),
        layers.Dense(embed_dim, dtype=dtype),
    ]
    model, structure = tiny_calibration_model(
        block_layers,
        vocab_size=vocab_size,
        sequence_length=seq_len,
        embed_dim=embed_dim,
        dtype=dtype,
    )
    if mode == "awq":
        kwargs["num_grid_points"] = 5
    rng = np.random.default_rng(7)
    config = calibration_config(
        mode,
        dataset=token_dataset(num_samples, seq_len, vocab_size, rng),
        tokenizer=lambda text: text,
        num_samples=num_samples,
        sequence_length=seq_len,
        group_size=8,
        quantization_layer_structure=structure,
        **kwargs,
    )
    return model, config


class CalibrationRunTest(testing.TestCase):
    """The pieces the calibration drivers share, once per mode."""

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_resolution_error_names_the_received_policy(self, mode):
        layer = layers.Dense(4)
        layer.build((None, 3))
        strategy = strategy_registry.get_strategy(mode)
        with self.assertRaisesRegex(
            ValueError,
            f"{mode.upper()} quantization.*Received: dtype_policy=<.*float32",
        ):
            strategy.resolve_group_size(layer, None)


@pytest.mark.requires_trainable_backend
class CalibrationRunModelTest(testing.TestCase):
    """The drivers as `model.quantize` runs them."""

    @parameterized.named_parameters(("gptq", "gptq"), ("awq", "awq"))
    def test_run_under_a_bfloat16_policy(self, mode):
        # The solve runs in float32 when the variables are `bfloat16`. On
        # the same weights and data, the result matches a float32 run up to
        # bfloat16 rounding and the codes that rounding flips.
        model, config = _tiny_model(mode, dtype="bfloat16")
        reference, reference_config = _tiny_model(mode)
        reference.set_weights(model.get_weights())
        model.quantize(mode, config=config)
        reference.quantize(mode, config=reference_config)
        block = config.quantization_layer_structure["sequential_blocks"][0]
        for layer in block.layers:
            self.assertFalse(layer.calibration_pending)
        outputs = ops.convert_to_numpy(
            ops.cast(model(config.dataset[0]), "float32")
        )
        expected = ops.convert_to_numpy(reference(config.dataset[0]))
        error = np.linalg.norm(outputs - expected) / np.linalg.norm(expected)
        self.assertLess(error, 0.2)
