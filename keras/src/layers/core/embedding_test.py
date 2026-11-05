import math
import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import constraints
from keras.src import export
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import quantizers
from keras.src import saving
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.quantizers.awq_config import AWQConfig
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.quantization_config import Int4QuantizationConfig
from keras.src.quantizers.quantization_config import Int8QuantizationConfig
from keras.src.quantizers.quantizers import AbsMaxQuantizer
from keras.src.testing import test_case
from keras.src.testing import test_utils


class EmbeddingTest(test_case.TestCase):
    @parameterized.named_parameters(
        ("int8", "int8", {"axis": -1}),
        (
            "int4",
            "int4",
            {"axis": -1, "value_range": (-8, 7), "output_dtype": "int8"},
        ),
        ("int8_custom", "int8", {"axis": -1}),
    )
    def test_embedding_quantize_config(self, mode, weight_quantizer_args):
        """Test Embedding quantization with QuantizationConfig."""
        layer = layers.Embedding(input_dim=10, output_dim=6)
        layer.build((None,))

        weight_quantizer = AbsMaxQuantizer(**weight_quantizer_args)
        if mode == "int8":
            config = Int8QuantizationConfig(
                weight_quantizer=weight_quantizer, activation_quantizer=None
            )
        elif mode == "int4":
            # Custom quantizers require per-channel mode (block_size=None)
            config = Int4QuantizationConfig(
                weight_quantizer=weight_quantizer,
                activation_quantizer=None,
                block_size=None,
            )

        layer.quantize(mode, config=config)

        # Verify weights are quantized
        self.assertEqual(standardize_dtype(layer._embeddings.dtype), "int8")
        self.assertTrue(hasattr(layer, "embeddings_scale"))

        # Verify call works
        x = np.random.randint(0, 10, size=(2, 3))
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 6))

    def test_embedding_basics(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 4, "output_dim": 3},
            input_shape=(2,),
            input_dtype="int32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4, "mask_zero": True},
            input_shape=(2, 3),
            input_dtype="int64",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_sparse(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4},
            input_shape=(2, 3),
            input_dtype="int32",
            input_sparse=True,
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    @pytest.mark.skipif(
        not backend.SUPPORTS_RAGGED_TENSORS,
        reason="Backend does not support ragged tensors.",
    )
    def test_ragged(self):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 5, "output_dim": 4},
            input_shape=(2, 3),
            input_dtype="int32",
            input_ragged=True,
            expected_output_shape=(2, None, 4),
            expected_output_ragged=True,
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            # run_training_check=False,
        )

    def test_correctness(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array([2, 1, 0]))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_correctness_sparse(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x = tf.SparseTensor([[0, 0], [1, 2]], [2, 1], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            x = jax_sparse.BCOO(([2, 1], [[0, 0], [1, 2]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        self.assertAllClose(
            layer(x),
            np.array(
                [
                    [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
                ]
            ),
        )

    def test_masking(self):
        layer = layers.Embedding(input_dim=3, output_dim=2, mask_zero=True)
        layer.build()
        out = layer.compute_mask(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([True, True, False]))

    def test_compute_mask_no_masking(self):
        layer = layers.Embedding(input_dim=3, output_dim=2, mask_zero=False)
        input_data = np.array([2, 1, 0])
        mask = layer.compute_mask(input_data)
        self.assertIsNone(mask)

    def test_embedding_constraints(self):
        layer = layers.Embedding(3, 2, embeddings_constraint="non_neg")
        layer.build((None, 2))
        self.assertIsInstance(layer.embeddings.constraint, constraints.NonNeg)

    def test_invalid_input_dim(self):
        for bad_value in (3.7, 4.0, "3", None, True, False, -1, 0):
            with self.assertRaisesRegex(
                ValueError, "`input_dim` must be a positive integer"
            ):
                layers.Embedding(input_dim=bad_value, output_dim=8)
        with self.assertRaisesRegex(
            TypeError, "`input_dim` must be a positive integer"
        ):
            layers.Embedding.from_config({"input_dim": 3.7, "output_dim": 8})

    def test_invalid_output_dim(self):
        for bad_value in (3.7, 4.0, "3", None, True, False, -1, 0):
            with self.assertRaisesRegex(
                ValueError, "`output_dim` must be a positive integer"
            ):
                layers.Embedding(input_dim=10, output_dim=bad_value)
        with self.assertRaisesRegex(
            TypeError, "`output_dim` must be a positive integer"
        ):
            layers.Embedding.from_config({"input_dim": 10, "output_dim": 3.7})

    def test_weights_constructor_arg(self):
        layer = layers.Embedding(3, 4, weights=np.ones((3, 4)))
        self.assertAllClose(layer.embeddings.numpy(), np.ones((3, 4)))
        layer = layers.Embedding(3, 4, weights=[np.ones((3, 4))])
        self.assertAllClose(layer.embeddings.numpy(), np.ones((3, 4)))

    @pytest.mark.requires_trainable_backend
    def test_enable_lora(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_lora(4)
        self.assertLen(layer.trainable_weights, 2)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 3)
        self.assertDType(layer.lora_embeddings_a, "float32")
        self.assertDType(layer.lora_embeddings_b, "float32")

        # Try eager call
        x = np.random.randint(0, 9, size=(64, 3))
        y = np.random.random((64, 3, 16))
        _ = layer(x[:2])

        init_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        init_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()

        # Try calling fit()
        model = models.Sequential(
            [
                layer,
            ]
        )
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        final_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_embeddings_value - final_lora_a_embeddings_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_embeddings_value - final_lora_b_embeddings_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "lora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-lora model
        new_model = models.Sequential(
            [
                layers.Input((3,), dtype="int32"),
                layers.Embedding(10, 16),
            ]
        )
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_lora_with_alpha(self):
        # Create an `Embedding` layer without specifying `lora_rank`
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build((None,))  # Build the layer

        # Set the base embeddings to known values.
        base_emb = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        )
        layer.embeddings.assign(base_emb)

        # Enable LoRA with a custom alpha: `rank`=2, `lora_alpha`=3.0.
        layer.enable_lora(2, lora_alpha=3.0)
        self.assertEqual(layer.lora_rank, 2)
        self.assertEqual(layer.lora_alpha, 3.0)

        # Manually assign known values to lora weights.
        a_val = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
        b_val = np.array([[0.5, 0.5], [0.6, 0.6]], dtype=np.float32)
        layer.lora_embeddings_a.assign(a_val)
        layer.lora_embeddings_b.assign(b_val)

        # Compute the expected delta.
        # Scaling factor: (3.0 / 2) = 1.5
        effective_delta = 1.5 * np.matmul(a_val, b_val)
        expected_embeddings = base_emb + effective_delta

        # Verify that the effective embeddings match expectation.
        actual_embeddings = ops.convert_to_numpy(layer.embeddings)
        self.assertAllClose(
            actual_embeddings, expected_embeddings, tpu_atol=1e-3, tpu_rtol=1e-3
        )

    def test_lora_rank_argument(self):
        self.run_layer_test(
            layers.Embedding,
            init_kwargs={"input_dim": 5, "output_dim": 4, "lora_rank": 2},
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_lora_alpha_argument(self):
        layer = layers.Embedding(
            input_dim=5, output_dim=4, lora_rank=2, lora_alpha=16
        )
        layer.build()
        self.assertEqual(layer.lora_alpha, 16)
        self.assertEqual(layer.get_config()["lora_alpha"], 16)

        # The forward pass scales the update by `lora_alpha / lora_rank`.
        # Every value, product and sum is exact in bfloat16, so the check
        # also holds where matmuls round their inputs to bfloat16 (TPU).
        embeddings = np.arange(20, dtype="float32").reshape((5, 4)) / 4 - 2
        lora_a = (
            np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, -1]], "float32") / 2
        )
        lora_b = np.ones((2, 4), dtype="float32")
        layer._embeddings.assign(embeddings)
        layer.lora_embeddings_a.assign(lora_a)
        layer.lora_embeddings_b.assign(lora_b)
        x = np.array([[0, 1, 4], [2, 3, 0]], dtype="int32")
        expected = (embeddings + (16 / 2) * lora_a @ lora_b)[x]
        self.assertAllClose(layer(x), expected)

        # A layer built from the config keeps `lora_alpha`.
        restored = layers.Embedding.from_config(layer.get_config())
        restored.build()
        self.assertEqual(restored.lora_alpha, 16)
        restored.set_weights(layer.get_weights())
        self.assertAllClose(restored(x), expected)

    def test_enable_lora_with_embeddings_constraint(self):
        layer = layers.Embedding(
            input_dim=10, output_dim=16, embeddings_constraint="max_norm"
        )
        with self.assertRaisesRegex(
            ValueError, "incompatible with embedding constraints"
        ):
            layer.enable_lora(rank=2)

    def test_enable_lora_when_already_enabled(self):
        layer = layers.Embedding(input_dim=10, output_dim=16)
        layer.build()
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(ValueError, "lora is already enabled"):
            layer.enable_lora(rank=2)

    @pytest.mark.requires_trainable_backend
    def test_enable_dora(self):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_dora(4)
        self.assertLen(layer.trainable_weights, 3)
        self.assertLen(layer.non_trainable_weights, 1)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, 4)
        self.assertDType(layer.dora_embeddings_a, "float32")
        self.assertDType(layer.dora_embeddings_b, "float32")
        self.assertDType(layer.dora_magnitude, "float32")

        # Try eager call
        x = np.random.randint(0, 9, size=(64, 3))
        y = np.random.random((64, 3, 16))
        _ = layer(x[:2])

        init_dora_a_embeddings_value = layer.dora_embeddings_a.numpy()
        init_dora_b_embeddings_value = layer.dora_embeddings_b.numpy()
        init_dora_magnitude_value = layer.dora_magnitude.numpy()

        # Try calling fit()
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_dora_a_embeddings_value = layer.dora_embeddings_a.numpy()
        final_dora_b_embeddings_value = layer.dora_embeddings_b.numpy()
        final_dora_magnitude_value = layer.dora_magnitude.numpy()

        diff_a = np.max(
            np.abs(init_dora_a_embeddings_value - final_dora_a_embeddings_value)
        )
        diff_b = np.max(
            np.abs(init_dora_b_embeddings_value - final_dora_b_embeddings_value)
        )
        diff_m = np.max(
            np.abs(init_dora_magnitude_value - final_dora_magnitude_value)
        )

        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)
        self.assertGreater(diff_m, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(self.get_temp_dir(), "dora_model.keras")
        model.save(temp_filepath)

        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].dora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "dora_model.weights.h5"
        )
        model.save_weights(temp_filepath)

        # Load the file into a fresh, non-dora model
        new_model = models.Sequential(
            [
                layers.Input((3,), dtype="int32"),
                layers.Embedding(10, 16),
            ]
        )
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

        # Try loading a normal checkpoint into a dora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    def test_enable_dora_with_alpha(self):
        layer = layers.Embedding(input_dim=3, output_dim=2)
        layer.build((None,))

        base_emb = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        )
        layer._embeddings.assign(base_emb)

        layer.enable_dora(2, dora_alpha=3.0)
        self.assertEqual(layer.dora_rank, 2)
        self.assertEqual(layer.dora_alpha, 3.0)

        a_val = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=np.float32)
        b_val = np.array([[0.5, 0.5], [0.6, 0.6]], dtype=np.float32)
        layer.dora_embeddings_a.assign(a_val)
        layer.dora_embeddings_b.assign(b_val)

        # Manually compute expected effective embeddings.
        expected_delta = 1.5 * np.matmul(a_val, b_val)
        W_combined = base_emb + expected_delta
        norm = np.sqrt(np.sum(np.square(W_combined), axis=0))
        expected_embeddings = ops.convert_to_numpy(layer.dora_magnitude) * (
            W_combined / norm
        )

        actual_embeddings = ops.convert_to_numpy(layer.embeddings)
        self.assertAllClose(
            actual_embeddings, expected_embeddings, tpu_atol=1e-3, tpu_rtol=1e-3
        )

    def test_dora_rank_argument(self):
        self.run_layer_test(
            layers.Embedding,
            init_kwargs={"input_dim": 5, "output_dim": 4, "dora_rank": 2},
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=1,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )

    def test_dora_lora_mutual_exclusivity(self):
        layer = layers.Embedding(input_dim=10, output_dim=16)
        layer.build()
        layer.enable_lora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "LoRA is already enabled. Cannot enable DoRA."
        ):
            layer.enable_dora(rank=2)

        layer2 = layers.Embedding(input_dim=10, output_dim=16)
        layer2.build()
        layer2.enable_dora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "dora is already enabled. Cannot enable LoRA."
        ):
            layer2.enable_lora(rank=2)

    def test_enable_dora_with_embeddings_constraint(self):
        layer = layers.Embedding(
            input_dim=10, output_dim=16, embeddings_constraint="max_norm"
        )
        with self.assertRaisesRegex(
            ValueError, "DoRA is incompatible with embedding constraints"
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_on_unbuilt_layer(self):
        layer = layers.Embedding(input_dim=10, output_dim=16)
        with self.assertRaisesRegex(
            ValueError, "Cannot enable dora on a layer that isn't yet built."
        ):
            layer.enable_dora(rank=2)

    def test_enable_dora_when_already_enabled(self):
        layer = layers.Embedding(input_dim=10, output_dim=16)
        layer.build()
        layer.enable_dora(rank=2)
        with self.assertRaisesRegex(
            ValueError, "dora is already enabled. This can only be done once"
        ):
            layer.enable_dora(rank=2)

    def test_enable_lora_after_int4_quantization_freezes_variable(self):
        # In int4 mode the `embeddings` property returns a freshly unpacked
        # tensor rather than the variable, so freezing must go through
        # `_embeddings`. Backends whose tensors reject attribute assignment
        # raise `AttributeError` here otherwise.
        layer = layers.Embedding(input_dim=10, output_dim=16)
        layer.build()
        layer.quantize("int4")
        layer.enable_lora(rank=2)
        self.assertFalse(layer._embeddings.trainable)

    # Test quantization-related methods.

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
    )
    def test_quantize_int(self, mode):
        layer = layers.Embedding(10, 16)
        layer.build()
        x = np.random.randint(0, 9, size=(64, 3))
        y_float = layer(x)
        layer.quantize(mode)

        # Verify the dtype of the weights.
        # The embeddings's dtype is int8, despite the int4 quantization, because
        # we pack the int4 values into int8.
        self.assertEqual(standardize_dtype(layer._embeddings.dtype), "int8")
        self.assertEqual(
            standardize_dtype(layer.embeddings_scale.dtype),
            layer.variable_dtype,
        )

        # Verify the unpacked embeddings for int4 quantization.
        if mode == "int4":
            self.assertAllClose(
                layer.embeddings,
                quantizers.unpack_int4(
                    layer._embeddings, layer.output_dim, axis=-1
                ),
            )

        # Verify the correctness of the outputs.
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 1e-3)  # A weak correctness test

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
        new_model = models.Sequential([layers.Embedding(10, 16)])
        new_model.build((None, 3))
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
    )
    def test_quantize_on_unbuilt_layer(self, mode):
        layer = layers.Embedding(10, 16)
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            layer.quantize(mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
    )
    def test_quantize_on_subclass(self, mode):
        class MyEmbedding(layers.Embedding):
            pass

        layer = MyEmbedding(10, 16)
        layer.build()
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode)

        layer.quantize(mode, type_check=False)  # No error

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("int4", "int4"),
    )
    def test_quantize_when_already_quantized(self, mode):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.quantize(mode)
        for m in ("int8", "int4"):
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

        layer = layers.Embedding(10, 16, dtype=f"{mode}_from_float32")
        layer.build()
        for m in ("int8", "int4"):
            with self.assertRaisesRegex(
                ValueError, "is already quantized with dtype_policy="
            ):
                layer.quantize(m)

    @parameterized.named_parameters(
        ("int8", "int8_from_float32", 2),
        ("int4", "int4_from_float32", 2),  # embeddings + scale
    )
    def test_quantize_by_setting_dtype_policy(
        self, policy, expected_num_variables
    ):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.dtype_policy = policy
        self.assertLen(layer.variables, expected_num_variables)

    @parameterized.named_parameters(
        ("int7", "int7"),
        ("float7", "float7"),
    )
    def test_quantize_invalid_mode(self, mode):
        layer = layers.Embedding(10, 16)
        layer.build()
        x = np.random.randint(0, 9, size=(1, 3))
        # dtype_policy should not be altered by failed quantization
        original_dtype_policy = layer.dtype_policy

        # Test quantize
        with self.assertRaisesRegex(ValueError, "Invalid quantization mode."):
            layer.quantize(mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_build
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            layer.quantized_build((None, 2), mode)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

        # Test quantized_call
        with self.assertRaisesRegex(
            NotImplementedError, "Invalid quantization mode."
        ):
            # Explicitly set quantization_mode
            layer._dtype_policy._quantization_mode = mode
            layer.quantized_call(x)
        self.assertEqual(layer.dtype_policy, original_dtype_policy)

    @parameterized.named_parameters(
        ("gptq", "gptq"),
        ("awq", "awq"),
    )
    def test_quantize_unsupported_calibration_modes(self, mode):
        # Embedding only supports int8/int4. GPTQ/AWQ must be rejected
        # without stashing a stale quantization config on the layer.
        layer = layers.Embedding(10, 16)
        layer.build()
        if mode == "gptq":
            config = GPTQConfig(dataset=None, tokenizer=None)
        else:
            config = AWQConfig(dataset=None, tokenizer=None)
        with self.assertRaises(NotImplementedError):
            layer.quantize(mode, config=config)
        self.assertIsNone(layer.quantization_config)

    @parameterized.named_parameters(
        ("int8", "int8_from_mixed_bfloat16", 0, 2),
        (
            "int4",
            "int4_from_mixed_bfloat16",
            0,
            2,
        ),  # per-channel (no zero point)
    )
    def test_quantize_dtype_argument(
        self, dtype, num_trainable_weights, num_non_trainable_weights
    ):
        self.run_layer_test(
            layers.Embedding,
            {"input_dim": 4, "output_dim": 3, "dtype": dtype},
            input_shape=(2,),
            input_dtype="int32",
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
        )
        self.run_layer_test(
            layers.Embedding,
            {
                "input_dim": 5,
                "output_dim": 4,
                "mask_zero": True,
                "dtype": dtype,
            },
            input_shape=(2, 3),
            input_dtype="int64",
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=num_trainable_weights,
            expected_num_non_trainable_weights=num_non_trainable_weights,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )

    @parameterized.named_parameters(
        ("int8", "int8", 2, 2, 4),
        ("int4", "int4", 2, 4, 6),  # +2 for embeddings_zero + g_idx
    )
    @pytest.mark.requires_trainable_backend
    def test_quantize_lora_integration(
        self,
        mode,
        num_trainable_weights,
        num_non_trainable_weights,
        num_torch_params,
    ):
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.enable_lora(4)
        layer.quantize(mode)
        self.assertLen(layer.trainable_weights, num_trainable_weights)
        self.assertLen(layer.non_trainable_weights, num_non_trainable_weights)
        if backend.backend() == "torch":
            self.assertLen(layer.torch_params, num_torch_params)

        # Try calling fit()
        init_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        init_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        x = np.random.randint(0, 9, size=(64, 3))
        y = np.random.random((64, 3, 16))
        model = models.Sequential([layer])
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y)

        final_lora_a_embeddings_value = layer.lora_embeddings_a.numpy()
        final_lora_b_embeddings_value = layer.lora_embeddings_b.numpy()
        diff_a = np.max(
            np.abs(init_lora_a_embeddings_value - final_lora_a_embeddings_value)
        )
        diff_b = np.max(
            np.abs(init_lora_b_embeddings_value - final_lora_b_embeddings_value)
        )
        self.assertGreater(diff_a, 0.0)
        self.assertGreater(diff_b, 0.0)

        # Try saving and reloading the model
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.keras"
        )
        model.save(temp_filepath)
        new_model = saving.load_model(temp_filepath)
        self.assertTrue(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try saving and reloading the model's weights only
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_lora_model.weights.h5"
        )
        model.save_weights(temp_filepath)
        new_model = models.Sequential(
            [layers.Input((3,), dtype="int32"), layers.Embedding(10, 16)]
        )
        new_model.quantize(mode)
        new_model.load_weights(temp_filepath)
        self.assertFalse(new_model.layers[0].lora_enabled)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Try loading a normal checkpoint into a lora model
        new_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x), atol=0.5)

        # Test export and TFSMLayer reloading when using tensorflow backend
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
            ref_input = tf.random.normal((32, 3))
            ref_output = model(ref_input)
            model.export(temp_filepath, format="tf_saved_model")
            reloaded_layer = export.TFSMLayer(temp_filepath)
            self.assertAllClose(
                reloaded_layer(ref_input), ref_output, atol=1e-7
            )
            self.assertLen(reloaded_layer.weights, len(model.weights))
            self.assertLen(
                reloaded_layer.trainable_weights, len(model.trainable_weights)
            )
            self.assertLen(
                reloaded_layer.non_trainable_weights,
                len(model.non_trainable_weights),
            )

    def test_legacy_load_own_variables(self):
        # In previous versions, `load_own_variables` accepted a store with
        # numeric keys.
        float32_store = {
            "0": np.random.random((10, 16)).astype("float32"),
        }
        int8_store = {
            "0": np.random.randint(-128, 127, size=(10, 16), dtype="int8"),
            "1": np.random.random((10,)).astype("float32"),
        }
        int4_store = {
            "0": np.random.randint(-128, 127, size=(10, 8), dtype="int8"),
            "1": np.random.random((10,)).astype("float32"),
        }

        # Test float32 layer.
        layer = layers.Embedding(10, 16)
        layer.build()
        layer.load_own_variables(float32_store)
        self.assertAllClose(layer._embeddings, float32_store["0"])

        # Test int8-quantized layer.
        layer = layers.Embedding(10, 16, dtype="int8_from_float32")
        layer.build()
        layer.load_own_variables(int8_store)
        self.assertAllClose(layer._embeddings, int8_store["0"])
        self.assertAllClose(layer.embeddings_scale, int8_store["1"])

        # Test int4-quantized layer.
        layer = layers.Embedding(10, 16, dtype="int4_from_float32")
        layer.build()
        layer.load_own_variables(int4_store)
        self.assertAllClose(layer._embeddings, int4_store["0"])
        self.assertAllClose(layer.embeddings_scale, int4_store["1"])

    @staticmethod
    def _build_embedding_for_mode(mode, input_dim=64, output_dim=64):
        """Builds an `Embedding` populated for `mode`'s serialization spec."""
        if mode == "none":
            layer = layers.Embedding(input_dim, output_dim)
            layer.build()
        elif mode == "int8":
            layer = layers.Embedding(
                input_dim, output_dim, dtype="int8_from_float32"
            )
            layer.build()
        elif mode == "int4_per_channel":
            layer = layers.Embedding(
                input_dim, output_dim, dtype="int4_from_float32"
            )
            layer.build()
        elif mode == "int4_grouped":
            layer = layers.Embedding(input_dim, output_dim)
            layer.build()
            layer.quantize("int4", config=Int4QuantizationConfig(block_size=16))
        else:
            raise ValueError(f"Unhandled test mode: {mode}")
        return layer

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_serialization_round_trip(self):
        for mode in ("none", "int8", "int4_per_channel", "int4_grouped"):
            with self.subTest(mode=mode):
                source = self._build_embedding_for_mode(mode)
                test_utils.randomize_serialized_variables(source)

                store = {}
                source.save_own_variables(store)
                # Variables are keyed by consecutive integer positions in the
                # mode's serialization spec -- the on-disk contract.
                n_expected = len(test_utils.serialized_variable_names(source))
                self.assertEqual(
                    set(store.keys()), {str(i) for i in range(n_expected)}
                )

                target = self._build_embedding_for_mode(mode)
                target.load_own_variables(store)
                test_utils.assert_serialized_variables_equal(
                    self, source, target
                )

    def test_load_own_variables_reports_clear_errors(self):
        # int8 spec order: embeddings ("0"), embeddings_scale ("1").
        source = self._build_embedding_for_mode("int8")
        test_utils.randomize_serialized_variables(source)
        store = test_utils.positional_store(source)

        # A missing variable trips the variable-count check with a clear error.
        missing = dict(store)
        del missing["1"]
        with self.assertRaisesRegex(ValueError, "expected 2 variables"):
            self._build_embedding_for_mode("int8").load_own_variables(missing)

        # A renumbered/corrupted key keeps the count but fails on the exact
        # missing position.
        corrupted = dict(store)
        corrupted["9"] = corrupted.pop("1")
        with self.assertRaises(KeyError):
            self._build_embedding_for_mode("int8").load_own_variables(corrupted)

    def test_embedding_int8_custom_quantizer(self):
        """
        Test custom quantizer serialization for embedding layer with
        int8 quantization.
        """
        # Setup
        weight_range = (-50, 50)
        config = Int8QuantizationConfig(
            weight_quantizer=AbsMaxQuantizer(axis=-1, value_range=weight_range),
        )

        # Build & Quantize
        layer = layers.Embedding(input_dim=100, output_dim=16)
        layer.build(None)
        layer.quantize("int8", config=config)

        # Serialize & Deserialize
        serialized = layer.get_config()
        new_layer = layers.Embedding.from_config(serialized)

        # Verify
        self.assertIsInstance(
            new_layer.quantization_config, Int8QuantizationConfig
        )
        quantizer = new_layer.quantization_config.weight_quantizer
        self.assertIsInstance(quantizer, AbsMaxQuantizer)
        self.assertEqual(quantizer.axis, (-1,))
        self.assertAllEqual(quantizer.value_range, weight_range)

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
        ("per_channel_none", None),
        ("per_channel_neg1", -1),
    )
    def test_int4_quantization_block_size(self, block_size):
        """Test int4 quantization with different block_size configurations."""

        input_dim, output_dim = 100, 256
        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer.build()

        x = np.random.randint(0, input_dim, size=(4, 8))
        y_float = layer(x)

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
            # Per-channel: one scale per vocabulary item
            expected_scale_shape = (input_dim,)
        else:
            # Sub-channel: n_groups scales per vocabulary item
            n_groups = math.ceil(output_dim / block_size)
            expected_scale_shape = (input_dim, n_groups)

        self.assertEqual(layer.embeddings_scale.shape, expected_scale_shape)

        # Verify outputs are reasonable
        y_quantized = layer(x)
        mse = ops.mean(ops.square(y_float - y_quantized))
        self.assertLess(mse, 1e-3)

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("grouped_block_128", 128),
        ("per_channel_none", None),
    )
    def test_int4_block_size_serialization(self, block_size):
        """Test that block_size is preserved through serialization."""

        input_dim, output_dim = 50, 128
        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        # Get output before serialization
        x = np.random.randint(0, input_dim, size=(2, 8))
        y_before = layer(x)

        # Save and load model to test full serialization roundtrip
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "int4_block_size_embedding_model.keras"
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

        # Verify outputs match after deserialization
        y_after = loaded_model(x)
        self.assertAllClose(y_before, y_after)

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("per_channel", None),
    )
    @pytest.mark.requires_trainable_backend
    def test_int4_block_size_with_lora(self, block_size):
        """Test int4 quantization with LoRA and different block_size."""
        input_dim, output_dim = 50, 128
        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)
        layer.enable_lora(rank=4)

        x = np.random.randint(0, input_dim, size=(4, 8))

        # Should run without error
        y = layer(x)
        self.assertEqual(y.shape, (4, 8, output_dim))

    @parameterized.named_parameters(
        ("grouped_block_64", 64),
        ("per_channel", None),
    )
    @pytest.mark.requires_trainable_backend
    def test_int4_block_size_with_dora(self, block_size):
        """Test int4 quantization with DoRA and different block_size."""
        input_dim, output_dim = 50, 128
        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)
        layer.enable_dora(rank=4)

        a_val = np.random.random(layer.dora_embeddings_a.shape).astype(
            "float32"
        )
        b_val = np.random.random(layer.dora_embeddings_b.shape).astype(
            "float32"
        )
        layer.dora_embeddings_a.assign(a_val)
        layer.dora_embeddings_b.assign(b_val)

        x = np.random.randint(0, input_dim, size=(4, 8))

        y = layer(x)
        self.assertEqual(y.shape, (4, 8, output_dim))

        layer.dora_enabled = False
        y_base = layer(x)
        layer.dora_enabled = True
        self.assertNotAllClose(y, y_base)

    def test_int4_grouped_merged_save_keeps_lora_update(self):
        # See `DenseTest.test_int4_grouped_merged_save_keeps_lora_update`.
        input_dim, output_dim, block_size = 12, 16, 4
        inputs = layers.Input((1,), dtype="int32")
        layer = layers.Embedding(input_dim, output_dim, name="target")
        model = models.Model(inputs, layer(inputs))
        layer.quantize(
            "int4", config=Int4QuantizationConfig(block_size=block_size)
        )
        ids = np.arange(input_dim)
        quantized = ops.convert_to_numpy(layer(ids))
        layer.enable_lora(2)
        rng = np.random.RandomState(0)
        layer.lora_embeddings_a.assign(
            rng.randn(input_dim, 2).astype("float32") * 0.5
        )
        layer.lora_embeddings_b.assign(
            rng.randn(2, output_dim).astype("float32") * 0.5
        )
        with_update = ops.convert_to_numpy(layer(ids))
        path = os.path.join(self.get_temp_dir(), "merged.keras")
        model.save(path)
        merged_layer = saving.load_model(path).get_layer("target")
        merged = ops.convert_to_numpy(merged_layer(ids))
        # Half a step of the re-quantized grid: groups run along the
        # output axis, one row of scales per token.
        scale = ops.convert_to_numpy(merged_layer.embeddings_scale)
        half_step = 0.5 * scale[:, np.arange(output_dim) // block_size]
        self.assertGreater(
            np.abs(with_update - quantized).max(), 4 * half_step.max()
        )
        self.assertTrue(
            np.all(np.abs(merged - with_update) <= half_step * 1.001 + 1e-6)
        )

    def test_int4_grouped_vs_perchannel_scale_shapes(self):
        """Test that grouped and per-channel have different scale shapes."""

        input_dim, output_dim = 100, 256
        block_size = 64

        # Per-channel layer
        layer_pc = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer_pc.build()
        config_pc = Int4QuantizationConfig(block_size=None)
        layer_pc.quantize("int4", config=config_pc)

        # Grouped layer
        layer_grouped = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        layer_grouped.build()
        config_grouped = Int4QuantizationConfig(block_size=block_size)
        layer_grouped.quantize("int4", config=config_grouped)

        # Verify different scale shapes
        self.assertEqual(layer_pc.embeddings_scale.shape, (input_dim,))
        n_groups = math.ceil(output_dim / block_size)
        self.assertEqual(
            layer_grouped.embeddings_scale.shape, (input_dim, n_groups)
        )

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
        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
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
        layer = layers.Embedding(input_dim=10, output_dim=16)
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

        layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        layer.build()

        config = Int4QuantizationConfig(block_size=block_size)
        layer.quantize("int4", config=config)

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        y_before = layer(x)
        g_idx_before = ops.convert_to_numpy(layer.g_idx)

        # Save and load
        model = models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "embedding_int4_g_idx_model.keras"
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

    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_string_matches_default_config(self):
        """`quantize("int4")` must resolve to the exact same scheme as the
        default `Int4QuantizationConfig()` (grouped, block_size=128): identical
        variables (names/shapes/dtypes/values), dtype policy, and outputs."""
        input_dim, output_dim = 32, 256
        embeddings = np.random.RandomState(0).randn(input_dim, output_dim)
        embeddings = embeddings.astype("float32")

        def build_quantized(config):
            layer = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
            layer.build()
            layer._embeddings.assign(embeddings)
            layer.quantize("int4", config=config)
            return layer

        layer_str = build_quantized(config=None)
        layer_cfg = build_quantized(config=Int4QuantizationConfig())

        self.assertEqual(layer_str.dtype_policy.name, "int4/128_from_float32")
        self.assertEqual(
            layer_str.dtype_policy.name, layer_cfg.dtype_policy.name
        )

        vars_str = {v.name: v for v in layer_str.weights}
        vars_cfg = {v.name: v for v in layer_cfg.weights}
        self.assertEqual(set(vars_str.keys()), set(vars_cfg.keys()))
        for name, v_str in vars_str.items():
            v_cfg = vars_cfg[name]
            self.assertEqual(tuple(v_str.shape), tuple(v_cfg.shape))
            self.assertEqual(
                standardize_dtype(v_str.dtype),
                standardize_dtype(v_cfg.dtype),
            )
            self.assertAllClose(v_str, v_cfg)

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        self.assertAllClose(layer_str(x), layer_cfg(x))

    @parameterized.named_parameters(
        ("none", None),
        ("neg1", -1),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_per_channel_escape_hatch(self, block_size):
        """`block_size=None` and `block_size=-1` both select the per-channel
        escape hatch: no zero-point / g_idx and an `int4/-1` dtype policy."""
        layer = layers.Embedding(input_dim=32, output_dim=256)
        layer.build()
        layer.quantize(
            "int4", config=Int4QuantizationConfig(block_size=block_size)
        )
        self.assertEqual(tuple(layer.embeddings_scale.shape), (32,))
        self.assertIsNone(layer.embeddings_zero)
        self.assertIsNone(layer.g_idx)
        self.assertEqual(layer.dtype_policy.name, "int4/-1_from_float32")

    @parameterized.named_parameters(
        ("block_none", "None", True),
        ("block_neg1", "-1", True),
        ("block_128", "128", False),
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu(), reason="Segfault on Tensorflow GPU"
    )
    def test_int4_policy_string_reload_builds_right_variables(
        self, block_token, per_channel
    ):
        """Old checkpoints identified only by their int4 dtype-policy string
        must deserialize and rebuild the correct variables. Covers the legacy
        `int4/None` spelling of per-channel plus `int4/-1` and `int4/128`."""
        input_dim, output_dim = 32, 256
        policy = f"int4/{block_token}_from_float32"
        layer = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim, dtype=policy
        )
        layer.build()

        self.assertEqual(layer.quantization_mode, "int4")
        if per_channel:
            self.assertEqual(tuple(layer.embeddings_scale.shape), (input_dim,))
            self.assertIsNone(layer.g_idx)
        else:
            block_size = int(block_token)
            n_groups = math.ceil(output_dim / block_size)
            self.assertEqual(
                tuple(layer.embeddings_scale.shape), (input_dim, n_groups)
            )
            self.assertIsNotNone(layer.g_idx)
            self.assertEqual(tuple(layer.g_idx.shape), (output_dim,))
        self.assertEqual(standardize_dtype(layer._embeddings.dtype), "int8")

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        self.assertEqual(tuple(layer(x).shape), (2, 3, output_dim))
