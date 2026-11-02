"""Tests for AWQ quantization."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.quantizers.awq import AWQCalibrator
from keras.src.quantizers.awq import _get_weight_scale
from keras.src.quantizers.awq import awq_quantize_matrix
from keras.src.quantizers.awq import awq_search_best_clip
from keras.src.quantizers.awq import awq_search_optimal_scales
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.quantization_test_utils import calibrate_layer
from keras.src.quantizers.quantization_test_utils import calibration_config
from keras.src.quantizers.quantization_test_utils import tiny_calibration_model
from keras.src.quantizers.quantization_test_utils import token_dataset

# Shared RNG instance for reproducible tests
RNG = np.random.default_rng(seed=42)


class MockTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size=100, seq_len=64):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def tokenize(self, text):
        # Simple character-based tokenization
        tokens = [ord(c) % self.vocab_size for c in str(text)]
        # Pad or truncate to seq_len
        if len(tokens) < self.seq_len:
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        return ops.array([tokens], dtype="int32")

    def __call__(self, text):
        return self.tokenize(text)


@pytest.mark.requires_trainable_backend
class AWQAlgorithmTest(testing.TestCase):
    """Test AWQ algorithm core functionality."""

    def test_scale_search_returns_valid_scales(self):
        """Test that scale search returns valid positive scales."""
        weights = RNG.standard_normal((32, 16)).astype("float32")
        activations = ops.abs(
            ops.add(RNG.standard_normal((16,)).astype("float32"), 0.1)
        )

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=10, group_size=-1
        )

        self.assertEqual(scales.shape, (16,))
        # All scales should be positive
        self.assertTrue(ops.all(ops.greater(scales, 0)))

    def test_scale_search_with_zero_activations(self):
        """Test scale search handles near-zero activations."""
        weights = ops.array(RNG.standard_normal((32, 16)).astype("float32"))
        # Some activations are very small
        activations = np.abs(RNG.standard_normal((16,)).astype("float32"))
        activations[:5] = 1e-10
        activations = ops.array(activations)

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=10, group_size=-1
        )

        # Should handle gracefully without NaN or Inf
        self.assertFalse(ops.any(ops.isnan(scales)))
        self.assertFalse(ops.any(ops.isinf(scales)))

    def test_get_weight_scale_shape_and_range(self):
        """Weight statistic is per-in-channel and normalized to (0, 1]."""
        weights = RNG.standard_normal((32, 64)).astype("float32")
        for group_size in (-1, 16):
            w_stat = _get_weight_scale(weights, group_size)
            self.assertEqual(w_stat.shape, (64,))
            self.assertTrue(ops.all(ops.greater(w_stat, 0)))
            self.assertTrue(ops.all(ops.less_equal(w_stat, 1.0 + 1e-6)))

    def test_search_best_clip_shapes(self):
        """Clip search returns a per-group bound not exceeding the max."""
        weights = RNG.standard_normal((64, 32)).astype("float32")
        scales = ops.add(
            ops.abs(RNG.standard_normal((32,)).astype("float32")), 0.1
        )
        weights_scaled = ops.multiply(weights, scales)
        sample = RNG.standard_normal((128, 32)).astype("float32")

        best_max, gs, n_group = awq_search_best_clip(
            weights_scaled, sample, scales, group_size=8, n_grid=20
        )
        self.assertEqual(gs, 8)
        self.assertEqual(n_group, 4)
        self.assertEqual(best_max.shape, (64, 4, 1))
        # Bound must be positive and never exceed the original per-group max.
        w_grouped = ops.reshape(weights_scaled, (64, 4, 8))
        org_max = ops.max(ops.abs(w_grouped), axis=-1, keepdims=True)
        self.assertTrue(ops.all(ops.greater(best_max, 0)))
        self.assertTrue(
            ops.all(ops.less_equal(best_max, ops.add(org_max, 1e-6)))
        )

    def test_clip_reduces_per_group_reconstruction_error(self):
        """Clipping must not increase the per-group output recon error.

        The clip search selects, per (output channel, group), the shrink factor
        minimizing the reconstruction error of that group's partial output
        against the (unclipped) scaled weights. Because the search grid always
        includes the no-shrink candidate, this error can only decrease. This
        test measures that exact objective with vs without clipping on weights
        with injected outliers.
        """
        from keras.src.quantizers.quantizers import dequantize_with_sz_map

        keras.utils.set_random_seed(0)
        out_features, in_features, group_size = 64, 128, 32
        n_group = in_features // group_size

        weights = RNG.standard_normal((out_features, in_features)).astype(
            "float32"
        )
        # Inject weight outliers on a few output rows so clipping matters.
        weights[::16] *= 6.0
        sample = RNG.standard_normal((256, in_features)).astype("float32")
        x_stat = ops.mean(ops.abs(sample), axis=0)

        def _per_group_error(apply_clip):
            q, sc, z, aw, gi = awq_quantize_matrix(
                weights,
                x_stat,
                num_grid_points=10,
                group_size=group_size,
                apply_clip=apply_clip,
                activation_sample=sample,
            )
            # Same FP reference for both: unclipped scaled weights.
            w_scaled = ops.multiply(weights, aw)
            x_scaled = ops.divide(sample, aw)
            w_hat = dequantize_with_sz_map(q, sc, z, ops.cast(gi, "int32"))
            xg = ops.reshape(x_scaled, (-1, n_group, group_size))
            wg_fp = ops.reshape(w_scaled, (out_features, n_group, group_size))
            wg_q = ops.reshape(w_hat, (out_features, n_group, group_size))
            fp = ops.einsum("rng,ong->orn", xg, wg_fp)
            qt = ops.einsum("rng,ong->orn", xg, wg_q)
            return float(ops.sum(ops.square(ops.subtract(qt, fp)))), aw

        err_no_clip, aw_no = _per_group_error(False)
        err_clip, aw_clip = _per_group_error(True)
        # Scale search is independent of clipping, so scales must match.
        self.assertAllClose(aw_no, aw_clip, atol=1e-6)
        # The clip search can only reduce this objective.
        self.assertLessEqual(err_clip, err_no_clip + 1e-4)
        # On outlier weights it should strictly help.
        self.assertLess(err_clip, err_no_clip)

    def test_quantize_matrix_shapes(self):
        """Test that quantize_matrix returns correct shapes."""
        # weights_transpose has shape [out_features, in_features]
        weights = ops.array(RNG.standard_normal((32, 16)).astype("float32"))
        activations = ops.add(
            ops.abs(RNG.standard_normal((16,)).astype("float32")), 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=10, group_size=-1
        )

        # Quantized shape: [out_features, in_features]
        self.assertEqual(quantized.shape, (32, 16))
        # Scale shape: [out_features, num_groups]
        self.assertEqual(scale.shape, (32, 1))
        # AWQ scales: per-channel for input features
        self.assertEqual(awq_scales.shape, (16,))
        # AWQ zero shape: [out_features, num_groups]
        self.assertEqual(zero.shape, (32, 1))
        # Group indices
        self.assertEqual(g_idx.shape, (16,))

    def test_quantize_matrix_with_grouping(self):
        """Test quantize_matrix with group size."""
        # Use dimensions divisible by group_size for cleaner test
        weights = ops.array(RNG.standard_normal((64, 32)).astype("float32"))
        activations = ops.add(
            ops.abs(RNG.standard_normal((32,)).astype("float32")), 0.1
        )

        # Test per-channel mode (group_size=-1) which is well-supported
        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=8
        )

        # Quantized shape: [out_features, in_features]
        self.assertEqual(quantized.shape, (64, 32))
        # Scale shape: [out_features, num_groups]
        self.assertEqual(scale.shape, (64, 4))  # 32 in_features / 8 group_size
        # AWQ scales: per-channel for input features
        self.assertEqual(awq_scales.shape, (32,))
        # AWQ zero shape: [out_features, num_groups]
        self.assertEqual(zero.shape, (64, 4))
        # Group indices
        self.assertEqual(g_idx.shape, (32,))

        # Check g_idx values
        self.assertEqual(ops.max(g_idx), 3)  # 4 groups: 0,1,2,3
        self.assertEqual(awq_scales.shape, (32,))

    def test_quantize_matrix_grouped_shapes(self):
        """Test awq_quantize_matrix with positive group_size.

        This is a regression test for the InvalidArgumentError that occurred
        when group_size != -1 due to shape mismatch in broadcasting.
        """
        out_features = 768
        in_features = 768
        group_size = 128
        n_groups = in_features // group_size  # 6 groups

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.array(
            np.abs(RNG.standard_normal((in_features,)).astype("float32")) + 0.1
        )

        quantized, scale, zero, awq_scales, g_idx = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=group_size
        )

        # Quantized should match input shape
        self.assertEqual(quantized.shape, (out_features, in_features))
        # Scale should be [out_features, n_groups]
        self.assertEqual(scale.shape, (out_features, n_groups))
        # Zero should be [out_features, n_groups]
        self.assertEqual(zero.shape, (out_features, n_groups))
        # AWQ scales should be per-input-channel
        self.assertEqual(awq_scales.shape, (in_features,))
        # g_idx should be [in_features]
        self.assertEqual(g_idx.shape, (in_features,))

        # Verify g_idx values
        expected_g_idx = ops.floor_divide(ops.arange(in_features), group_size)
        self.assertAllEqual(g_idx, expected_g_idx)

    def test_quantize_matrix_grouped_no_nan_inf(self):
        """Test grouped quantization produces no NaN or Inf values."""
        out_features = 256
        in_features = 512
        group_size = 64

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        quantized, scale, _, awq_scales, _ = awq_quantize_matrix(
            weights, activations, num_grid_points=5, group_size=group_size
        )

        # Check for NaN/Inf in all outputs
        self.assertFalse(ops.any(ops.isnan(quantized)))
        self.assertFalse(ops.any(ops.isinf(quantized)))
        self.assertFalse(ops.any(ops.isnan(scale)))
        self.assertFalse(ops.any(ops.isinf(scale)))
        self.assertFalse(ops.any(ops.isnan(awq_scales)))
        self.assertFalse(ops.any(ops.isinf(awq_scales)))

    def test_scale_search_grouped_quantization(self):
        """Test awq_search_optimal_scales with grouped quantization."""
        out_features = 128
        in_features = 256
        group_size = 32

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        scales = awq_search_optimal_scales(
            weights, activations, num_grid_points=5, group_size=group_size
        )

        # Scales should be [in_features]
        self.assertEqual(scales.shape, (in_features,))
        # All scales should be positive
        self.assertTrue(ops.all(ops.greater(scales, 0)))
        # No NaN or Inf
        self.assertFalse(ops.any(ops.isnan(scales)))
        self.assertFalse(ops.any(ops.isinf(scales)))

    @parameterized.named_parameters(
        ("group_8", 8),
        ("group_16", 16),
        ("group_32", 32),
        ("group_64", 64),
        ("group_128", 128),
    )
    def test_quantize_matrix_various_group_sizes(self, group_size):
        """Test awq_quantize_matrix with various group sizes."""
        out_features = 64
        in_features = 128
        n_groups = in_features // group_size

        weights = ops.array(
            RNG.standard_normal((out_features, in_features)).astype("float32")
        )
        activations = ops.add(
            ops.abs(RNG.standard_normal((in_features,)).astype("float32")), 0.1
        )

        _, scale, zero, _, _ = awq_quantize_matrix(
            weights, activations, num_grid_points=3, group_size=group_size
        )

        self.assertEqual(
            scale.shape,
            (out_features, n_groups),
            f"Failed for group_size={group_size}",
        )
        self.assertEqual(
            zero.shape,
            (out_features, n_groups),
            f"Failed for group_size={group_size}",
        )


@pytest.mark.requires_trainable_backend
class AWQLayerTest(testing.TestCase):
    """Test AWQ class for layer quantization."""

    def test_awq_on_dense_layer(self):
        """Test AWQ on a Dense layer."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            num_grid_points=10,
        )

        layer.quantize(config=config)

        # Simulate activation capture
        calibration_data = RNG.standard_normal((64, 16)).astype("float32")
        calibrator = calibrate_layer(
            layer, config, calibration_data, solve=False
        )

        self.assertEqual(calibrator.num_samples, 64)
        # Activation magnitudes should be non-negative
        self.assertTrue(
            ops.all(ops.greater_equal(calibrator.activation_magnitudes, 0))
        )

    def test_awq_activation_accumulation(self):
        """Test that activation magnitudes accumulate as a running mean.

        The reference AWQ statistic is the per-channel mean of |x|. The running
        update must be equivalent to computing that mean over all rows seen so
        far, regardless of how the rows are split into batches.
        """
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)

        # A second batch with a different row count exercises the
        # weighting.
        batch1 = RNG.standard_normal((10, 16)).astype("float32")
        batch2 = ops.add(RNG.standard_normal((30, 16)).astype("float32"), 1.0)
        calibrator = calibrate_layer(layer, config, batch1, batch2, solve=False)

        # Running mean must equal the mean of |x| over all rows.
        combined = ops.concatenate([ops.abs(batch1), ops.abs(batch2)], axis=0)
        expected_mean = ops.mean(combined, axis=0)
        self.assertEqual(calibrator.num_samples, 40)
        self.assertAllClose(
            calibrator.activation_magnitudes, expected_mean, atol=1e-6
        )

    def test_awq_clip_sample_capture(self):
        """Test that a bounded activation sample is stashed for clipping."""
        from keras.src.quantizers.awq import MAX_CLIP_SAMPLE_ROWS

        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=5
        )
        layer.quantize(config=config)
        calibrator = calibrate_layer(layer, config, solve=False)

        # Feed more rows than the cap across several batches.
        for _ in range(4):
            batch = RNG.standard_normal((MAX_CLIP_SAMPLE_ROWS // 2, 16)).astype(
                "float32"
            )
            calibrator.observe(batch)

        # The stash is bounded by MAX_CLIP_SAMPLE_ROWS.
        self.assertEqual(calibrator._clip_sample_rows, MAX_CLIP_SAMPLE_ROWS)
        total_rows = int(
            sum(int(ops.shape(s)[0]) for s in calibrator._clip_samples)
        )
        self.assertEqual(total_rows, MAX_CLIP_SAMPLE_ROWS)

    def test_awq_clip_disabled_skips_sample(self):
        """Test that apply_clip=False does not stash a sample."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=-1,
            num_grid_points=5,
            apply_clip=False,
        )
        layer.quantize(config=config)
        calibrator = calibrate_layer(
            layer,
            config,
            RNG.standard_normal((32, 16)).astype("float32"),
            solve=False,
        )
        self.assertEqual(calibrator._clip_sample_rows, 0)
        self.assertEqual(calibrator._clip_samples, [])

    def test_awq_layer_variables_created(self):
        """Test that AWQ layer variables are properly created."""
        layer = layers.Dense(32)
        layer.build(input_shape=(None, 16))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=10
        )
        layer.quantize(config=config)

        # Check that AWQ-specific variables exist
        self.assertTrue(hasattr(layer, "quantized_kernel"))
        self.assertTrue(hasattr(layer, "kernel_scale"))
        self.assertIsNotNone(layer.kernel_zero)
        self.assertTrue(hasattr(layer, "awq_scales"))
        self.assertIsNotNone(layer.g_idx)
        self.assertTrue(layer.calibration_pending)


@pytest.mark.requires_trainable_backend
class AWQIntegrationTest(testing.TestCase):
    """Integration tests for AWQ quantization."""

    def test_dense_layer_quantize_awq(self):
        """Test Dense layer can be quantized with AWQ."""
        layer = layers.Dense(64)
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=16, num_grid_points=5
        )
        layer.quantize(config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_einsum_dense_layer_quantize_awq(self):
        """Test EinsumDense layer can be quantized with AWQ."""
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(64,))
        layer.build(input_shape=(None, 32))

        config = AWQConfig(
            dataset=None, tokenizer=None, group_size=-1, num_grid_points=5
        )
        layer.quantize(config=config)

        # Check layer is properly configured
        self.assertEqual(layer.quantization_mode, "awq")
        self.assertTrue(hasattr(layer, "awq_scales"))

    def test_model_quantize_requires_structure(self):
        """Test model.quantize requires structure for AWQ."""
        model = models.Sequential([layers.Dense(10, input_shape=(5,))])
        model.build()

        config = AWQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(vocab_size=100, seq_len=5),
        )

        with self.assertRaisesRegex(ValueError, "quantization structure"):
            model.quantize(config=config)

    @pytest.mark.requires_trainable_backend
    def test_awq_save_load_round_trip_einsum_dense_block(self):
        """Regression test for a RecursionError when saving an AWQ model.

        Saving an AWQ-quantized model used to raise a RecursionError because
        `AWQConfig.get_config` serialized `quantization_layer_structure`,
        which holds live model layers, forming a reference cycle
        (layer -> config -> layer). This builds a tiny model whose quantized
        block contains both `Dense` and `EinsumDense` (unlike the Dense-only
        round trip in `AWQAccuracyTest`), saves it, reloads it, and checks
        the predictions are preserved exactly.
        """
        vocab_size, seq_len, embed_dim = 32, 8, 4
        model, structure = tiny_calibration_model(
            [
                layers.Dense(embed_dim, activation="relu"),
                layers.EinsumDense(
                    "abc,cd->abd", output_shape=(seq_len, embed_dim)
                ),
            ],
            vocab_size=vocab_size,
            sequence_length=seq_len,
            embed_dim=embed_dim,
            head_units=2,
        )
        (embedding,) = structure["pre_block_layers"]
        head = model.layers[-1]

        rng = np.random.default_rng(seed=21)
        config = calibration_config(
            "awq",
            dataset=token_dataset(3, seq_len, vocab_size, rng),
            tokenizer=lambda text: text,
            num_samples=2,
            sequence_length=seq_len,
            group_size=4,
            num_grid_points=5,
            quantization_layer_structure=structure,
        )

        # Layers outside the structure (embedding, pooling, head) are not
        # quantized at all, so the round-trip can be compared exactly.
        model.quantize("awq", config=config)
        self.assertIsNone(getattr(head, "quantization_mode", None))

        # The embedding only supports int8/int4; `quantize` must reject the
        # unsupported mode without stashing a stale AWQ config on it.
        self.assertIsNone(embedding.quantization_config)

        x_eval = rng.integers(0, vocab_size, size=(2, seq_len)).astype("int32")
        y_quantized = model.predict(x_eval)

        # This `save` used to raise a RecursionError.
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        restored = saving.load_model(path)
        y_restored = restored.predict(x_eval)
        self.assertAllClose(y_quantized, y_restored)

        # The quantized block state survives the round-trip.
        restored_block = next(
            l for l in restored.layers if isinstance(l, models.Sequential)
        )
        restored_dense = restored_block.layers[0]
        self.assertEqual(
            getattr(restored_dense, "quantization_mode", None), "awq"
        )
        self.assertTrue(hasattr(restored_dense, "quantized_kernel"))
        self.assertIsNone(
            restored_dense.quantization_config.quantization_layer_structure
        )


# Constants for end-to-end tests
VOCAB_SIZE = 1000
SEQ_LEN = 128
NUM_SAMPLES = 16
NUM_CLASSES = 32

CALIBRATION_TEXT = """
AWQ (Activation-aware Weight Quantization) is an efficient and accurate
low-bit weight quantization method for LLMs. AWQ is based on the observation
that weights are not equally important: protecting only 1% of salient weights
can greatly reduce quantization error. To find salient weights, AWQ looks at
the activation distribution, not weights. Salient weights are those that
correspond to channels with larger activation magnitudes. AWQ then applies
per-channel scaling to protect salient weights during quantization.
The key insight is that for a weight channel, if the corresponding activation
channel has large values, quantizing that weight channel will incur large
error. By scaling up salient weight channels before quantization and scaling
down during inference, AWQ can significantly reduce quantization error
while maintaining the same effective computation.
"""


def _mean_kl(p, q):
    """Compute mean KL divergence between two probability distributions."""
    eps = 1e-8
    p = ops.clip(p, eps, 1.0)
    q = ops.clip(q, eps, 1.0)
    return ops.mean(
        ops.sum(ops.multiply(p, ops.subtract(ops.log(p), ops.log(q))), axis=-1)
    )


def _top1_match_rate(a_logits, b_logits):
    """Calculate top-1 match rate between two sets of logits."""
    return ops.mean(
        ops.equal(ops.argmax(a_logits, axis=-1), ops.argmax(b_logits, axis=-1))
    )


def _get_sequence_classifier():
    """Create a transformer-based sequence classifier for testing."""
    embed_dim = 32
    num_heads = 4
    ff_dim = 32

    class SimpleTransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
            super().__init__(**kwargs)
            self.att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads
            )
            self.ffn = models.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            attention_output = self.att(inputs, inputs)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            return self.layernorm2(out1 + ffn_output)

    inputs = layers.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, embed_dim)(inputs)
    x = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    outputs = layers.Dense(NUM_CLASSES)(x)
    return models.Model(inputs, outputs)


def _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Character-based tokenizer for testing."""

    def _pad_or_trim_1d(ids, length):
        ids = ops.ravel(ops.array(ids, "int64"))
        if len(ids) < length:
            ids = ops.concatenate(
                [ids, ops.zeros(length - len(ids), dtype=ids.dtype)]
            )
        else:
            ids = ids[:length]
        return ids

    def _tok(x):
        if isinstance(x, str):
            ids = ops.convert_to_tensor(
                np.fromiter((ord(c) % vocab_size for c in x), dtype=np.int64)
            )
        else:
            ids = np.asarray(x, dtype=np.int64)
        ids = _pad_or_trim_1d(ids, seq_len)
        return ids[None, :]

    _tok.tokenize = _tok
    return _tok


def _string_dataset(
    long_text, num_samples=NUM_SAMPLES, sequence_length=SEQ_LEN
):
    """Yield string slices for calibration."""
    length = max(1, len(long_text) - sequence_length)
    for _ in range(num_samples):
        start = RNG.integers(0, length) if length > 1 else 0
        yield long_text[start : start + sequence_length]


@pytest.mark.requires_trainable_backend
class AWQAccuracyTest(testing.TestCase):
    """End-to-end accuracy preservation tests for AWQ quantization."""

    @parameterized.named_parameters(
        ("per_channel", -1, 20, 0.5, 0.30),
        ("group_16", 16, 10, 0.4, 0.40),
    )
    def test_awq_transformer_accuracy(
        self, group_size, num_grid_points, min_top1, max_kl
    ):
        """Test that AWQ quantization preserves model accuracy.

        This test:
        1. Creates a transformer-based sequence classifier
        2. Gets baseline (full precision) predictions
        3. Applies AWQ quantization with calibration data
        4. Compares quantized predictions against baseline
        5. Validates top-1 match rate and KL divergence bounds
        """
        keras.utils.set_random_seed(123)

        # Build calibration dataset
        calibration_set = list(_string_dataset(CALIBRATION_TEXT, NUM_SAMPLES))
        self.assertNotEmpty(calibration_set)

        # Build model and tokenizer
        model = _get_sequence_classifier()
        tokenizer = _char_tokenizer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)

        # Build eval batch from same distribution as calibration
        batch_size = min(8, len(calibration_set))
        eval_samples = [
            calibration_set[RNG.integers(0, len(calibration_set))]
            for _ in range(batch_size)
        ]
        x_eval = ops.concatenate([tokenizer(s) for s in eval_samples], axis=0)

        # Get baseline predictions (full precision)
        y_ref = model.predict(x_eval)

        # Define layer structure for AWQ
        embedding_layer = model.layers[1]
        transformer_block = model.layers[2]

        layer_structure = {
            "pre_block_layers": [embedding_layer],
            "sequential_blocks": [transformer_block],
        }

        # Configure AWQ
        awq_config = AWQConfig(
            dataset=calibration_set,
            tokenizer=tokenizer,
            num_samples=NUM_SAMPLES,
            sequence_length=SEQ_LEN,
            group_size=group_size,
            num_grid_points=num_grid_points,
            quantization_layer_structure=layer_structure,
        )

        # Quantize model with AWQ
        model.quantize(config=awq_config)

        # Get post-quantization predictions
        y_q = model.predict(x_eval)

        # Calculate accuracy metrics
        top1_match = _top1_match_rate(y_ref, y_q)

        p_ref = ops.softmax(y_ref)
        p_q = ops.softmax(y_q)
        kl = _mean_kl(p_ref, p_q)

        # Validate accuracy preservation
        self.assertGreaterEqual(
            float(top1_match),
            min_top1,
            f"Top-1 agreement too low for group_size={group_size}: "
            f"{float(top1_match):.3f}",
        )
        self.assertLessEqual(
            float(kl),
            max_kl,
            f"KL divergence too high for group_size={group_size}: "
            f"{float(kl):.3f}",
        )

    @parameterized.named_parameters(
        ("per_channel", -1, 0.35),
        ("group_16", 16, 0.35),
        ("group_32", 32, 0.35),
        ("group_64", 64, 0.35),
        ("group_128", 128, 0.35),
    )
    def test_awq_accuracy_various_group_sizes(
        self, group_size, max_relative_mse
    ):
        """Test AWQ accuracy across various group sizes.

        Verifies that quantizing a single layer maintains reasonable
        output reconstruction error and correct variable shapes.
        """
        in_features = 128
        out_features = 64

        keras.utils.set_random_seed(42)

        # Create fresh layer for each test
        layer = layers.Dense(out_features)
        layer.build(input_shape=(None, in_features))

        # Create data
        calibration_data = RNG.standard_normal((64, in_features)).astype(
            "float32"
        )
        test_data = RNG.standard_normal((16, in_features)).astype("float32")

        # Get original output
        original_output = layer(test_data)

        # Configure and quantize
        config = AWQConfig(
            dataset=None,
            tokenizer=None,
            group_size=group_size,
            num_grid_points=5,
        )
        calibrate_layer(layer, config, calibration_data)

        # Verify layer variables have correct shapes for grouped quantization
        if group_size > 0:
            n_groups = in_features // group_size
            self.assertEqual(
                layer.kernel_scale.shape,
                (n_groups, out_features),
                f"kernel_scale shape mismatch for group_size={group_size}",
            )
            self.assertEqual(
                layer.kernel_zero.shape,
                (n_groups, out_features),
                f"kernel_zero shape mismatch for group_size={group_size}",
            )

        # Verify output
        quantized_output = layer(test_data)

        # Should have no NaN/Inf
        self.assertFalse(
            ops.any(ops.isnan(quantized_output)),
            f"NaN in output for group_size={group_size}",
        )
        self.assertFalse(
            ops.any(ops.isinf(quantized_output)),
            f"Inf in output for group_size={group_size}",
        )

        # Should maintain reasonable accuracy
        mse = ops.mean(
            ops.power(ops.subtract(original_output, quantized_output), 2)
        )
        original_var = ops.var(original_output)
        relative_mse = ops.divide(mse, ops.add(original_var, 1e-8))

        self.assertLess(
            relative_mse,
            max_relative_mse,
            f"Accuracy too low for group_size={group_size}: "
            f"relative_mse={relative_mse:.4f}",
        )

    def test_awq_save_load_round_trip(self):
        """Full AWQ quantize -> save -> load round trip.

        Only the Dense layers inside the structure's ``sequential_blocks``
        are quantized; the embedding and classifier head stay untouched,
        and predictions are reproduced after a save/load cycle. Uses small
        local dimensions to keep the grid search fast.
        """
        keras.utils.set_random_seed(123)
        seq_len = 16
        vocab = 48
        num_classes = 4
        embed_dim = 8
        model, structure = tiny_calibration_model(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(embed_dim),
            ],
            vocab_size=vocab,
            sequence_length=seq_len,
            embed_dim=embed_dim,
            head_units=num_classes,
        )
        (embedding,) = structure["pre_block_layers"]
        (block,) = structure["sequential_blocks"]
        head = model.layers[-1]

        rng = np.random.default_rng(seed=7)
        tokenizer = _char_tokenizer(vocab_size=vocab, seq_len=seq_len)
        config = calibration_config(
            "awq",
            dataset=token_dataset(4, seq_len, vocab, rng),
            tokenizer=tokenizer,
            group_size=8,
            num_samples=4,
            sequence_length=seq_len,
            num_grid_points=5,
            quantization_layer_structure=structure,
        )

        model.quantize("awq", config=config)

        # In-structure Dense layers are quantized and calibrated.
        for dense in block.layers:
            self.assertEqual(dense.quantization_mode, "awq")
            self.assertFalse(dense.calibration_pending)

        # Out-of-structure layers must stay completely untouched.
        self.assertIsNone(getattr(head, "quantization_mode", None))
        self.assertFalse(hasattr(head, "quantized_kernel"))
        self.assertIsNone(getattr(embedding, "quantization_mode", None))
        self.assertFalse(hasattr(embedding, "quantized_kernel"))

        # Predictions survive a save/load round trip.
        eval_rng = np.random.default_rng(seed=99)
        x_eval = eval_rng.integers(0, vocab, size=(4, seq_len), dtype=np.int32)
        y_before = model.predict(x_eval)

        path = os.path.join(self.get_temp_dir(), "awq_model.keras")
        model.save(path)
        reloaded = saving.load_model(path)
        y_after = reloaded.predict(x_eval)

        self.assertAllClose(y_before, y_after)

    def test_awq_calibration_hooks_fire_during_model_quantize(self):
        """Calibration hooks must run during `model.quantize("awq")`.

        Regression test: `model.quantize` switches layers to their quantized
        dtype policy before calibration, after which `Operation.__call__`
        dispatches to `quantized_call` instead of `call`. The calibration
        hooks used to patch `call` only, so they never fired: activation
        magnitudes stayed zero and the AWQ scale search was
        activation-blind. Asserts the hook actually runs and records
        non-zero activation magnitudes.
        """
        keras.utils.set_random_seed(123)
        seq_len = 16
        vocab = 48
        embed_dim = 8

        block = models.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        inputs = layers.Input(shape=(seq_len,), dtype="int32")
        embedding = layers.Embedding(vocab, embed_dim)
        x = embedding(inputs)
        x = block(x)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(4)(x)
        model = models.Model(inputs, outputs)

        rng = np.random.default_rng(seed=7)
        dataset = [
            rng.integers(0, vocab, size=(1, seq_len), dtype=np.int32)
            for _ in range(4)
        ]
        tokenizer = _char_tokenizer(vocab_size=vocab, seq_len=seq_len)

        config = AWQConfig(
            dataset=dataset,
            tokenizer=tokenizer,
            group_size=8,
            num_samples=4,
            sequence_length=seq_len,
            num_grid_points=5,
            quantization_layer_structure={
                "pre_block_layers": [embedding],
                "sequential_blocks": [block],
            },
        )

        hook_calls = [0]
        max_magnitude = [0.0]
        original_update = AWQCalibrator.observe

        def spy_update(awq_self, inp):
            hook_calls[0] += 1
            result = original_update(awq_self, inp)
            magnitudes = ops.convert_to_numpy(awq_self.activation_magnitudes)
            max_magnitude[0] = max(
                max_magnitude[0], float(np.abs(magnitudes).max())
            )
            return result

        AWQCalibrator.observe = spy_update
        try:
            model.quantize("awq", config=config)
        finally:
            AWQCalibrator.observe = original_update

        self.assertGreater(hook_calls[0], 0)
        # All-zero magnitudes would be replaced with ones, making the
        # scale search activation-blind.
        self.assertGreater(max_magnitude[0], 0.0)

    def _tiny_awq_model_and_config(self, num_samples=4):
        """Embedding -> [Dense(16, relu), Dense(8)] block, tiny AWQ config."""
        keras.utils.set_random_seed(123)
        seq_len, vocab, embed_dim = 16, 48, 8

        block = models.Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        inputs = layers.Input(shape=(seq_len,), dtype="int32")
        embedding = layers.Embedding(vocab, embed_dim)
        x = embedding(inputs)
        x = block(x)
        x = layers.GlobalAveragePooling1D()(x)
        model = models.Model(inputs, layers.Dense(4)(x))

        rng = np.random.default_rng(seed=7)
        dataset = [
            rng.integers(0, vocab, size=(1, seq_len), dtype=np.int32)
            for _ in range(num_samples)
        ]
        config = AWQConfig(
            dataset=dataset,
            tokenizer=_char_tokenizer(vocab_size=vocab, seq_len=seq_len),
            group_size=8,
            num_samples=num_samples,
            sequence_length=seq_len,
            num_grid_points=5,
            quantization_layer_structure={
                "pre_block_layers": [embedding],
                "sequential_blocks": [block],
            },
        )
        return model, config

    def test_awq_recalibrates_downstream_stages(self):
        """AWQ must re-capture downstream statistics between stages.

        The two chained Dense layers form two execution stages. The first
        sweep captures both layers (2 hook calls per sample); after the
        first Dense is quantized, the second Dense's statistics must be
        re-captured against the quantized upstream activations (1 more
        call per sample). A single-sweep implementation records only
        2 * num_samples calls.
        """
        num_samples = 4
        model, config = self._tiny_awq_model_and_config(num_samples)

        hook_calls = [0]
        original_update = AWQCalibrator.observe

        def spy_update(awq_self, inp):
            hook_calls[0] += 1
            return original_update(awq_self, inp)

        AWQCalibrator.observe = spy_update
        try:
            model.quantize("awq", config=config)
        finally:
            AWQCalibrator.observe = original_update

        self.assertEqual(hook_calls[0], 3 * num_samples)

    def test_awq_calibration_runs_without_grad_tracking(self):
        """Calibration forwards must not build autograd graphs on torch.

        The per-sample activations are retained across the whole
        calibration loop, so retained graphs previously accumulated every
        intermediate activation and exhausted GPU memory.
        """
        if backend.backend() != "torch":
            self.skipTest("gradient tracking is specific to torch")
        model, config = self._tiny_awq_model_and_config()

        graph_free = [True]
        original_update = AWQCalibrator.observe

        def spy_update(awq_self, inp):
            if getattr(inp, "grad_fn", None) is not None:
                graph_free[0] = False
            return original_update(awq_self, inp)

        AWQCalibrator.observe = spy_update
        try:
            model.quantize("awq", config=config)
        finally:
            AWQCalibrator.observe = original_update

        self.assertTrue(graph_free[0])
