import math
import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.quantizers.report import QuantizationReport
from keras.src.testing import test_case
from keras.src.testing import test_utils
from keras.src.testing.test_utils import named_product


class ReversibleEmbeddingTest(test_case.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", {"axis": -1}, {"axis": -1}),
        (
            "int4",
            "int4",
            {"axis": -1, "value_range": (-8, 7), "output_dtype": "int8"},
            {"axis": -1},
        ),
        ("int8_weight_only", "int8", {"axis": -1}, None),
    )
    def test_reversible_embedding_quantize(
        self, mode, weight_quantizer_args, activation_quantizer_args
    ):
        """Test ReversibleEmbedding quantization with QuantizationConfig."""
        layer = layers.ReversibleEmbedding(
            input_dim=10, output_dim=6, tie_weights=True
        )
        layer.build((None,))

        weight_quantizer = AbsMaxQuantizer(**weight_quantizer_args)
        if activation_quantizer_args is not None:
            activation_quantizer = AbsMaxQuantizer(**activation_quantizer_args)
        else:
            activation_quantizer = None

        if mode == "int8":
            config = Int8QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
            )
        elif mode == "int4":
            # Custom quantizers require per-channel mode (block_size=None)
            config = Int4QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=activation_quantizer,
                block_size=None,
            )

        layer.quantize(mode, config=config)

        if activation_quantizer_args is not None:
            # Verify inputs_quantizer is set correctly
            self.assertIsInstance(layer.inputs_quantizer, AbsMaxQuantizer)
        else:
            # Verify inputs_quantizer is None
            self.assertIsNone(layer.inputs_quantizer)

        # Verify reverse call works
        x = np.random.random((2, 6)).astype("float32")
        y = layer(x, reverse=True)
        self.assertEqual(y.shape, (2, 10))

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_reversible_embedding_basics(self, tie_weights):
        self.run_layer_test(
            layers.ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "logit_soft_cap": 50,
            },
            input_data=np.random.randint(low=0, high=100, size=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=1 if tie_weights else 2,
        )

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_saving(self, tie_weights):
        input_data = np.random.randint(low=0, high=100, size=(4, 10))
        model = models.Sequential(
            [
                layers.ReversibleEmbedding(
                    input_dim=100,
                    output_dim=32,
                    tie_weights=tie_weights,
                )
            ]
        )
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model_output = model(input_data)
        model.save(path)
        restored_model = saving.load_model(path)
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)

    def test_correctness(self):
        layer = layers.ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

        layer = layers.ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 4.0, 6.0]]))

        layer = layers.ReversibleEmbedding(
            input_dim=3, output_dim=2, logit_soft_cap=5
        )
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 3.320184, 4.168273]]))

    def test_reverse_dtype(self):
        embedding = layers.ReversibleEmbedding(100, 16, reverse_dtype="float32")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDType(output_data, "float32")

        if backend.backend() == "torch":
            import torch

            if not torch.cuda.is_available():
                self.skipTest("Torch CPU does not support float16")

        embedding = layers.ReversibleEmbedding(100, 16, reverse_dtype="float16")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDType(output_data, "float16")

    @parameterized.named_parameters(
        named_product(mode=("int4", "int8"), tie_weights=(False, True))
    )
    def test_quantize_int(self, mode, tie_weights):
        layer = layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)
        layer.build()
        x = np.random.randint(0, 9, size=(64, 3))
        x_reverse = np.random.uniform(size=(64, 16)).astype("float32")
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)
        layer.quantize(mode)

        # Verify the dtype of the weights.
        if not tie_weights:
            # The reverse_embeddings's dtype is int8, despite the int4
            # quantization, because we pack the int4 values into int8.
            self.assertDType(layer.reverse_embeddings, "int8")
            self.assertDType(
                layer.reverse_embeddings_scale, layer.variable_dtype
            )

        # Verify the correctness of the outputs.
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)  # A weak correctness test
        self.assertLess(mse_reverse, 1e-2)  # A weak correctness test

        # Check model save / load round-trip.
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Check weights-only save / load round-trip.
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential(
            [layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)]
        )
        new_model.build((None, 3))
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        named_product(mode=("float8", "gptq"), tie_weights=(True, False))
    )
    def test_quantize_unsupported_mode_no_mutation(self, mode, tie_weights):
        # `ReversibleEmbedding` only supports int8/int4. Other modes must be
        # rejected before any state is mutated, so that `Model.quantize`
        # records the layer as skipped instead of leaving it half-quantized.
        layer = layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)
        layer.build()
        x = np.random.randint(0, 9, size=(4, 3))
        x_reverse = np.random.uniform(size=(4, 16)).astype("float32")
        y_before = layer(x)
        y_reverse_before = layer(x_reverse, reverse=True)
        original_dtype_policy = layer.dtype_policy

        if mode == "gptq":
            config = GPTQConfig(dataset=None, tokenizer=None)
        else:
            config = None
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            layer.quantize(mode, config=config)

        # No state was stashed or replaced by the failed quantization.
        self.assertIsNone(layer.quantization_config)
        self.assertFalse(getattr(layer, "_is_quantized", False))
        self.assertEqual(layer.dtype_policy, original_dtype_policy)
        self.assertDType(layer.embeddings, layer.variable_dtype)
        if not tie_weights:
            self.assertDType(layer.reverse_embeddings, layer.variable_dtype)

        # The layer still works in both directions with unchanged outputs.
        self.assertAllClose(layer(x), y_before)
        self.assertAllClose(layer(x_reverse, reverse=True), y_reverse_before)

        # The failed attempt must not block a supported quantization.
        layer.quantize("int8")
        self.assertTrue(layer._is_quantized)

    @parameterized.named_parameters(named_product(tie_weights=(True, False)))
    def test_model_quantize_unsupported_mode_skips_with_reason(
        self, tie_weights
    ):
        # When quantizing a whole model with a mode the embedding does not
        # support, the embedding must be skipped (and reported as such) while
        # the rest of the model is quantized and remains usable.
        embedding = layers.ReversibleEmbedding(10, 16, tie_weights=tie_weights)
        dense = layers.Dense(8)
        model = models.Sequential([embedding, dense])
        x = np.random.randint(0, 9, size=(4, 3))
        model.build((None, 3))

        # `ternary` is supported by `Dense` but not by the embedding.
        report = model.quantize("ternary", verbose=False)

        # The embedding was skipped with the "no support" reason and the
        # report lists it; the dense layer was quantized.
        unsupported = report.skipped_by_reason(
            QuantizationReport.SKIP_NO_SUPPORT
        )
        self.assertIn(embedding.path or embedding.name, unsupported)
        quantized_paths = [path for path, _, _ in report.quantized]
        self.assertIn(dense.path or dense.name, quantized_paths)

        # The embedding was left completely untouched.
        self.assertIsNone(embedding.quantization_config)
        self.assertFalse(getattr(embedding, "_is_quantized", False))
        self.assertDType(embedding.embeddings, embedding.variable_dtype)
        if not tie_weights:
            self.assertDType(
                embedding.reverse_embeddings, embedding.variable_dtype
            )

        # The model still runs, and so does the reverse path.
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (4, 3, 8))
        x_reverse = np.random.uniform(size=(4, 16)).astype("float32")
        y_reverse = embedding(x_reverse, reverse=True)
        self.assertEqual(ops.shape(y_reverse), (4, 10))

    @staticmethod
    def _build_reversible_for_mode(mode, tie_weights):
        layer = layers.ReversibleEmbedding(64, 32, tie_weights=tie_weights)
        layer.build()
        if mode != "none":
            layer.quantize(mode)
        return layer

    @parameterized.named_parameters(
        named_product(mode=("none", "int8", "int4"), tie_weights=(True, False))
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_serialization_round_trip(self, mode, tie_weights):
        source = self._build_reversible_for_mode(mode, tie_weights)
        test_utils.randomize_serialized_variables(source)

        store = {}
        source.save_own_variables(store)
        # Variables (incl. `reverse_*` for untied weights) are keyed by
        # consecutive integer positions in the mode's serialization spec --
        # the on-disk contract.
        n_expected = len(test_utils.serialized_variable_names(source))
        self.assertEqual(set(store.keys()), {str(i) for i in range(n_expected)})

        target = self._build_reversible_for_mode(mode, tie_weights)
        target.load_own_variables(store)
        test_utils.assert_serialized_variables_equal(self, source, target)

    @parameterized.named_parameters(
        ("int8_tie_weights", "int8_from_mixed_bfloat16", True, 0, 2),
        ("int8_untie_weights", "int8_from_mixed_bfloat16", False, 0, 4),
        ("int4_tie_weights", "int4_from_mixed_bfloat16", True, 0, 2),
        ("int4_untie_weights", "int4_from_mixed_bfloat16", False, 0, 4),
    )
    def test_quantize_dtype_argument(
        self,
        dtype,
        tie_weights,
        num_trainable_weights,
        num_non_trainable_weights,
    ):
        self.run_layer_test(
            layers.ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "dtype": dtype,
            },
            input_data=np.random.randint(low=0, high=100, size=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_non_trainable_variables=num_non_trainable_weights,
        )

    def test_reversible_embedding_int8_custom_quantizer(self):
        """
        Test custom quantizer serialization for reversible embedding layer with
        int8 quantization.
        """
        # Setup
        weight_range = (-20, 20)
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=-1, value_range=weight_range),
        )

        # Build & Quantize
        layer = layers.ReversibleEmbedding(input_dim=100, output_dim=16)
        layer.build(None)
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.ReversibleEmbedding.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )
        quantizer = new_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertAllEqual(quantizer.value_range, weight_range)

    def test_masking(self):
        layer = layers.ReversibleEmbedding(3, 2, mask_zero=True)
        layer.build()

        out = layer(np.array(([2, 1, 0])))
        mask = get_keras_mask(out)
        self.assertAllClose(mask, np.array([True, True, False]))

        out = layer(np.array(([[1.0, 2.0], [0.0, 0.0]])), reverse=True)
        mask = get_keras_mask(out)
        self.assertIsNone(mask)

    @parameterized.named_parameters(
        named_product(
            block_size=(64, 128, None, -1),
            tie_weights=(True, False),
        )
    )
    def test_int4_quantization_block_size(self, block_size, tie_weights):
        """Test int4 quantization with different block_size configurations."""

        input_dim, output_dim = 100, 256
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()

        x = np.random.randint(0, input_dim, size=(4, 8))
        x_reverse = np.random.random((4, output_dim)).astype("float32")
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)

        # Create config with specified block_size
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify block_size is stored
        self.assertEqual(
            layer._quantized_weight().scheme.group_size,
            None if block_size in (None, -1) else block_size,
        )

        # Verify embeddings_scale shape
        if block_size is None or block_size == -1:
            expected_scale_shape = (input_dim,)
        else:
            n_groups = math.ceil(output_dim / block_size)
            expected_scale_shape = (input_dim, n_groups)

        self.assertEqual(layer.embeddings_scale.shape, expected_scale_shape)

        # Verify reverse_embeddings_scale shape if not tied
        if not tie_weights:
            if block_size is None or block_size == -1:
                expected_reverse_scale_shape = (input_dim,)
            else:
                n_groups = math.ceil(output_dim / block_size)
                expected_reverse_scale_shape = (n_groups, input_dim)

            self.assertEqual(
                layer.reverse_embeddings_scale.shape,
                expected_reverse_scale_shape,
            )

        # Verify outputs are reasonable
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)
        self.assertLess(mse_reverse, 1e-2)

    @parameterized.named_parameters(
        named_product(
            block_size=(64, 128, None),
            tie_weights=(True, False),
        )
    )
    def test_int4_block_size_serialization(self, block_size, tie_weights):
        """Test that block_size is preserved through serialization."""
        input_dim, output_dim = 50, 128
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Get output before serialization
        x = np.random.randint(0, input_dim, size=(2, 8))
        y_before = layer(x)

        # Save and load model to test full serialization roundtrip
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(),
            f"int4_block_size_rev_emb_model_{tie_weights}.keras",
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify block_size is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertIsInstance(
            loaded_layer.quantization_config, Int4QuantizationConfig
        )
        self.assertEqual(
            loaded_layer.quantization_config.block_size, block_size
        )

        # Verify reverse_embeddings_zero is preserved for untied grouped
        if not tie_weights and block_size is not None:
            self.assertIsNotNone(loaded_layer.reverse_embeddings_zero)
            self.assertAllClose(
                loaded_layer.reverse_embeddings_zero,
                layer.reverse_embeddings_zero,
            )

        # Verify outputs match after deserialization
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

    @parameterized.named_parameters(
        ("tie_grouped", True, 64),
        ("tie_perchannel", True, None),
        ("untie_grouped", False, 64),
        ("untie_perchannel", False, None),
    )
    def test_int4_grouped_vs_perchannel_scale_shapes(
        self, tie_weights, block_size
    ):
        """Test that grouped and per-channel have different scale shapes."""

        input_dim, output_dim = 100, 256

        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim, tie_weights=tie_weights
        )
        layer.build()
        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        if block_size is None or block_size == -1:
            # Per-channel
            expected_scale_shape = (input_dim,)
            expected_reverse_scale_shape = (input_dim,)
        else:
            # Grouped
            n_groups = math.ceil(output_dim / block_size)
            expected_scale_shape = (input_dim, n_groups)
            expected_reverse_scale_shape = (n_groups, input_dim)

        self.assertEqual(layer.embeddings_scale.shape, expected_scale_shape)

        if not tie_weights:
            self.assertEqual(
                layer.reverse_embeddings_scale.shape,
                expected_reverse_scale_shape,
            )
            # Check reverse_embeddings_zero shape for grouped quantization
            if block_size is not None and block_size != -1:
                self.assertIsNotNone(layer.reverse_embeddings_zero)
                self.assertEqual(
                    layer.reverse_embeddings_zero.shape,
                    expected_reverse_scale_shape,
                )
            else:
                self.assertIsNone(layer.reverse_embeddings_zero)

    @parameterized.named_parameters(
        ("grouped_block_4", 4),
        ("grouped_block_8", 8),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_subchannel_g_idx_created(self, block_size):
        """Test that g_idx is created for sub-channel int4 quantization."""
        input_dim, output_dim = 10, 16
        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Verify g_idx is created
        self.assertIsNotNone(layer.g_idx)

        # Verify g_idx shape (output_dim for embedding)
        self.assertEqual(layer.g_idx.shape, (output_dim,))

        # g_idx is integer group metadata stored as float32 (see build).
        self.assertDType(layer.g_idx, "float32")

        # Verify g_idx values (should map each column to its group)
        expected_g_idx = np.arange(output_dim) // block_size
        self.assertAllClose(layer.g_idx, expected_g_idx)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_perchannel_no_g_idx(self):
        """Test that per-channel int4 does NOT create g_idx."""
        layer = layers.ReversibleEmbedding(input_dim=10, output_dim=16)
        layer.build()

        config = Int4QuantizationConfig(block_size=None)  # Per-channel
        layer.quantize("int4", config=config)

        # Verify g_idx is NOT created for per-channel
        self.assertIsNone(layer.g_idx)

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_subchannel_g_idx_serialization(self):
        """Test that g_idx is properly serialized and deserialized."""
        input_dim, output_dim = 10, 16
        block_size = 8

        layer = layers.ReversibleEmbedding(
            input_dim=input_dim, output_dim=output_dim
        )
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        y_before = layer(x)
        g_idx_before = ops.convert_to_numpy(layer.g_idx)

        # Save and load
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "rev_embedding_int4_g_idx_model.keras"
        )
        model.save(temp_filepath)
        loaded_model = saving.load_model(temp_filepath)

        # Verify g_idx is preserved
        loaded_layer = loaded_model.layers[0]
        self.assertIsNotNone(loaded_layer.g_idx)
        self.assertAllClose(loaded_layer.g_idx, g_idx_before)

        # Verify outputs match
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)


class ReversibleEmbeddingConsistencyTest(test_case.TestCase):
    """The reverse projection follows the forward table in every mode."""

    def _config(self, mode, **kwargs):
        if mode == "int8":
            return Int8QuantizationConfig(**kwargs)
        return Int4QuantizationConfig(block_size=-1, **kwargs)

    def _layers(self, mode, tie_weights, config=None, **kwargs):
        # A float reference and a quantized copy with identical weights.
        reference = layers.ReversibleEmbedding(
            12, 6, tie_weights=tie_weights, **kwargs
        )
        reference.build((None,))
        quantized = layers.ReversibleEmbedding(
            12, 6, tie_weights=tie_weights, **kwargs
        )
        quantized.build((None,))
        for name in ("_embeddings", "reverse_embeddings"):
            if hasattr(reference, name):
                getattr(quantized, name).assign(getattr(reference, name))
        quantized.quantize(mode, config=config)
        return reference, quantized

    @parameterized.named_parameters(
        named_product(mode=["int8", "int4"], tie_weights=[True, False])
    )
    def test_custom_weight_quantizer_applies_to_both_tables(
        self, mode, tie_weights
    ):
        # A user quantizer used to crash the untied reverse table, which
        # was quantized with the forward axis.
        if mode == "int8":
            quantizer = AbsMaxQuantizer(axis=-1)
        else:
            quantizer = AbsMaxQuantizer(
                axis=-1, value_range=(-8, 7), output_dtype="int8"
            )
        reference, quantized = self._layers(
            mode,
            tie_weights,
            config=self._config(mode, weight_quantizer=quantizer),
        )
        x = np.random.random((3, 6)).astype("float32")
        self.assertAllClose(
            quantized(x, reverse=True),
            reference(x, reverse=True),
            atol=0.2,
            rtol=0.2,
        )

    @parameterized.named_parameters(("int8", "int8"), ("int4", "int4"))
    def test_tied_reverse_includes_lora_delta(self, mode):
        # The LoRA update reaches the reverse projection of a tied layer, as
        # it does in float mode and in the forward lookup.
        _, layer = self._layers(mode, tie_weights=True)
        x = np.random.random((3, 6)).astype("float32")
        before = ops.convert_to_numpy(layer(x, reverse=True))
        layer.enable_lora(2)
        rng = np.random.default_rng(0)
        layer.lora_embeddings_a.assign(rng.random((12, 2)) - 0.5)
        layer.lora_embeddings_b.assign(rng.random((2, 6)) - 0.5)
        after = ops.convert_to_numpy(layer(x, reverse=True))
        a = ops.convert_to_numpy(layer.lora_embeddings_a)
        b = ops.convert_to_numpy(layer.lora_embeddings_b)
        expected = (layer.lora_alpha / layer.lora_rank) * (x @ b.T @ a.T)
        self.assertAllClose(
            after - before,
            expected,
            atol=1e-5,
            rtol=1e-5,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    @parameterized.named_parameters(("int8", "int8"), ("int4", "int4"))
    def test_mask_survives_quantization(self, mode):
        _, layer = self._layers(mode, tie_weights=True, mask_zero=True)
        outputs = layer(np.array([[0, 3, 5]]))
        mask = get_keras_mask(outputs)
        self.assertIsNotNone(mask)
        self.assertAllClose(mask, [[False, True, True]])

    @parameterized.named_parameters(("int8", "int8"), ("int4", "int4"))
    def test_reverse_dtype_is_honored(self, mode):
        reference, quantized = self._layers(
            mode, tie_weights=True, reverse_dtype="bfloat16"
        )
        x = np.random.random((3, 6)).astype("float32")
        logits = quantized(x, reverse=True)
        self.assertEqual(standardize_dtype(logits.dtype), "bfloat16")
        self.assertAllClose(
            logits, reference(x, reverse=True), atol=0.2, rtol=0.2
        )

    @parameterized.named_parameters(
        named_product(mode=["int8", "int4"], tie_weights=[True, False])
    )
    def test_reverse_outputs_survive_save_load(self, mode, tie_weights):
        _, layer = self._layers(mode, tie_weights)
        model = models.Sequential([layers.Input((6,)), layer])
        x = np.random.random((3, 6)).astype("float32")
        expected = model.layers[-1](x, reverse=True)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path)
        restored = saving.load_model(path)
        self.assertAllClose(restored.layers[-1](x, reverse=True), expected)

    def test_int4_policy_build_is_weight_only(self):
        # An int4 layer built from a dtype policy is weight-only on the
        # reverse projection, like every other int4 path.
        layer = layers.ReversibleEmbedding(12, 6, dtype="int4_from_float32")
        layer.build((None,))
        self.assertIsNone(layer.inputs_quantizer)
