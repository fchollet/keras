import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.testing import test_utils


class AttentionTest(testing.TestCase):
    def test_attention_basics(self):
        # No scale, no concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "score_mode": "dot",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )
        # Scale and concat.
        self.run_layer_test(
            layers.Attention,
            init_kwargs={
                "use_scale": True,
                "score_mode": "concat",
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )

    def test_attention_seed_preserved_in_config(self):
        layer = layers.Attention(dropout=0.5, seed=1337)
        config = layer.get_config()
        self.assertEqual(config["seed"], 1337)
        restored = layers.Attention.from_config(config)
        self.assertEqual(restored.seed, 1337)

    def test_attention_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        # Dot.
        layer = layers.Attention(score_mode="dot")
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output,
            [[[2.462, 3.462], [1.538, 2.538]]],
            atol=1e-3,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
        self.assertAllClose(
            scores,
            [[[0.269, 0.731], [0.731, 0.269]]],
            atol=1e-3,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

        # Concat.
        layer = layers.Attention(score_mode="concat")
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output,
            [[[1.727, 2.727], [2.272, 3.272]]],
            atol=1e-3,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )
        self.assertAllClose(
            scores,
            [[[0.636, 0.363], [0.363, 0.636]]],
            atol=1e-3,
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

    def test_attention_with_mask(self):
        layer = layers.Attention()
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        query_mask = np.array([[True, False]])
        value_mask = np.array([[True, False]])
        output, scores = layer(
            [query, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        self.assertAllClose(output, [[[1.0, 1.0], [0.0, 0.0]]])
        self.assertAllClose(scores, [[[1.0, 0.0], [1.0, 0.0]]])

    @parameterized.named_parameters(
        test_utils.named_product(dtype=["float32", "float16"])
    )
    def test_attention_with_fully_masked_row(self, dtype):
        # Batch element 0 masks every value position, element 1 masks one. A
        # fully masked row attends to nothing, so its output and scores are 0.
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]] * 2)
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]] * 2)
        value_mask = np.array([[False, False], [True, False]])
        layer = layers.Attention(dtype=dtype)
        output, scores = layer(
            [query.astype(dtype), value.astype(dtype)],
            mask=[None, value_mask],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output, [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [1.0, 2.0]]]
        )
        self.assertAllClose(
            scores, [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
        )

    @parameterized.named_parameters(
        test_utils.named_product(dtype=["float32", "float16"])
    )
    @pytest.mark.requires_trainable_backend
    def test_attention_fully_masked_row_has_zero_gradients(self, dtype):
        # Every value position is masked, so the gradients are 0. Every score
        # is below -16, which overflows to -inf in float16 if the masked
        # scores are offset rather than replaced, and gives NaN gradients.
        query = np.array([[[-10.0, -10.0], [-20.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        mask = [None, np.array([[False, False]])]
        layer = layers.Attention(dtype=dtype)
        grads = ops.grad(lambda q, v: layer([q, v], mask=mask), argnums=(0, 1))(
            query.astype(dtype), value.astype(dtype)
        )
        for grad in grads:
            self.assertAllClose(grad, ops.zeros_like(grad))

    @pytest.mark.skipif(backend.backend() != "torch", reason="Torch only test.")
    def test_attention_fully_masked_row_float16_backward_has_no_nan(self):
        # A torch dtype never compares equal to the string "float16", so this
        # checks that torch also masks float16 scores with 65504 rather than
        # with 1e9, which is -inf in float16. A fully masked row of -inf gives
        # NaN in the softmax backward, which anomaly detection reports even
        # though the gradients reaching the inputs are 0.
        import torch

        query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float16)
        value = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16, requires_grad=True
        )
        value_mask = np.array([[False, False]])
        layer = layers.Attention(dtype="float16")
        with torch.autograd.detect_anomaly():
            output = layer([query, value], mask=[None, value_mask])
            output.float().sum().backward()
        self.assertAllClose(value.grad, np.zeros((1, 2, 2)))

    def test_attention_2D_mask_shape_mismatch(self):
        layer = layers.Attention()
        batch_size, Tq, Tv, dim = 2, 3, 4, 5
        query = np.random.random((batch_size, Tq, dim)).astype(np.float32)
        value = np.random.random((batch_size, Tv, dim)).astype(np.float32)
        query_mask = np.array([[True, False, True], [True, False, True]])
        value_mask = np.array(
            [[True, False, True, True], [True, False, True, True]]
        )
        output, scores = layer(
            [query, value],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        self.assertEqual(output.shape, (batch_size, Tq, dim))
        self.assertEqual(scores.shape, (batch_size, Tq, Tv))

    def test_attention_errors(self):
        layer = layers.Attention()
        tensor = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "must be called on a list"):
            layer(tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor, tensor, tensor])

        with self.assertRaisesRegex(ValueError, "layer mask must be a list"):
            layer([tensor, tensor], mask=tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor], mask=[tensor])

    def test_attention_with_dropout(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        value = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        layer_with_dropout = layers.Attention(dropout=0.2)
        layer_without_dropout = layers.Attention()

        output1, scores1 = layer_with_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        output2, scores2 = layer_without_dropout(
            [query, value], return_attention_scores=True, training=True
        )
        self.assertNotAllClose(output1, output2)
        self.assertNotAllClose(scores1, scores2)

    def test_attention_invalid_score_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument score_mode. "
            "Expected one of {'dot', 'concat'}",
        ):
            layers.Attention(score_mode="invalid_mode")

    def test_attention_calculate_scores_with_scale(self):
        query = np.random.random((2, 3, 4))
        key = np.random.random((2, 4, 4))
        layer = layers.Attention(use_scale=True, score_mode="dot")
        layer.build(input_shape=[(2, 3, 4), (2, 4, 4)])
        expected_scores = np.matmul(query, key.transpose((0, 2, 1)))
        expected_scores *= layer.scale.numpy()
        actual_scores = layer._calculate_scores(query, key)
        self.assertAllClose(
            actual_scores, expected_scores, tpu_atol=1e-2, tpu_rtol=1e-2
        )

    def test_attention_calculate_score_mask_no_causal_no_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()
        mask = layer._calculate_score_mask(
            scores, v_mask=None, use_causal_mask=False
        )
        self.assertIsNone(
            mask,
            "Mask should be None when no causal mask and no value mask "
            "are used",
        )

    def test_attention_calculate_score_mask_with_causal_no_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()

        causal_mask = layer._calculate_score_mask(
            scores, v_mask=None, use_causal_mask=True
        )
        expected_causal_mask = np.tril(
            np.ones((1, scores.shape[1], scores.shape[2])), k=0
        )
        self.assertAllClose(causal_mask, expected_causal_mask, atol=1e-6)

    def test_attention_calculate_score_mask_with_causal_and_vmask(self):
        scores = np.random.random((2, 3, 4))
        layer = layers.Attention()
        v_mask = np.array([[True, False, True, False]])

        combined_mask = layer._calculate_score_mask(
            scores, v_mask=v_mask, use_causal_mask=True
        )
        expected_causal_mask = np.tril(
            np.ones((1, scores.shape[1], scores.shape[2])), k=0
        )
        expected_combined_mask = np.logical_and(
            expected_causal_mask, v_mask[:, np.newaxis, :]
        )
        self.assertAllClose(combined_mask, expected_combined_mask, atol=1e-6)

    def test_attention_compute_mask_with_no_mask(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        self.assertIsNone(
            layer.compute_mask(inputs=dummy_inputs, mask=None),
            "compute_mask should return None when mask is None",
        )

    def test_attention_compute_mask_with_first_element_none(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        mask = [None, np.array([True, False, True])]
        self.assertIsNone(
            layer.compute_mask(inputs=dummy_inputs, mask=mask),
            "compute_mask should return None when the first element is None",
        )

    def test_attention_compute_mask_does_not_return_none_with_valid_mask(self):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        valid_mask = np.array([True, False, True])
        mask = [valid_mask, np.array([False, True, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertIsNotNone(
            computed_mask,
            "compute_mask should not return None with a valid mask",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_valid_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [
            np.random.random((2, 3, 4)),
            np.random.random((2, 4, 4)),
        ]
        valid_mask = np.array([True, False, True])
        mask = [valid_mask, np.array([False, True, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertTrue(
            np.array_equal(computed_mask, valid_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_all_true_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([True, True, True])
        mask = [valid_mask, np.array([True, True, True])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = np.array([True, True, True])
        self.assertTrue(
            np.array_equal(computed_mask, expected_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_returns_correct_tensor_with_all_false_mask(
        self,
    ):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([False, False, False])
        mask = [valid_mask, np.array([False, False, False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = np.array([False, False, False])
        self.assertTrue(
            np.array_equal(computed_mask, expected_mask),
            "compute_mask did not return the correct mask tensor",
        )

    def test_attention_compute_mask_with_tolerance_1e_3(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-3),
            "Incorrect mask tensor within tolerance 1e-3",
        )

    def test_attention_compute_mask_with_tolerance_1e_5(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-5),
            "Incorrect mask tensor within tolerance 1e-5",
        )

    def test_attention_compute_mask_with_tolerance_1e_7(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1.0, 0.0, 1.0], dtype=float)
        mask = [valid_mask, np.array([0.0, 1.0, 0.0], dtype=float)]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_mask = valid_mask
        self.assertTrue(
            np.allclose(computed_mask, expected_mask, atol=1e-7),
            "Incorrect mask tensor within tolerance 1e-7 ",
        )

    def test_attention_compute_mask_with_single_element_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([True])
        mask = [valid_mask, np.array([False])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        expected_shape = (1,)
        self.assertEqual(computed_mask.shape, expected_shape)

    def test_attention_compute_mask_with_non_boolean_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        valid_mask = np.array([1, 0, 1])
        mask = [valid_mask, np.array([0, 1, 0])]
        computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
        computed_mask = ops.convert_to_numpy(computed_mask)
        self.assertTrue(np.array_equal(computed_mask, valid_mask))

    def test_attention_compute_mask_with_edge_case_masks(self):
        layer = layers.Attention()
        dummy_inputs = [np.ones((2, 3, 4)), np.ones((2, 4, 4))]
        edge_case_masks = [
            np.array([True, True, True]),
            np.array([False, False, False]),
            np.array([True, False, True]),
        ]
        for mask in edge_case_masks:
            computed_mask = layer.compute_mask(
                inputs=dummy_inputs, mask=[mask, mask]
            )
            computed_mask = ops.convert_to_numpy(computed_mask)
            self.assertTrue(np.array_equal(computed_mask, mask))

    def test_attention_compute_mask_with_different_input_shapes(self):
        layer = layers.Attention()
        input_shapes = [(2, 3, 4), (3, 2, 5), (4, 1, 6)]
        valid_mask = np.array([True, False, True])
        for shape in input_shapes:
            dummy_inputs = [np.ones(shape), np.ones(shape)]
            mask = [valid_mask, np.array([False, True, False])]
            computed_mask = layer.compute_mask(inputs=dummy_inputs, mask=mask)
            computed_mask = ops.convert_to_numpy(computed_mask)
            self.assertTrue(np.array_equal(computed_mask, valid_mask))

    def test_attention_compute_output_shape(self):
        layer = layers.Attention()

        query = np.random.random((2, 3, 4))
        value = np.random.random((2, 3, 5))
        key = np.random.random((2, 3, 4))
        layer = layers.Attention()
        output = layer([query, value, key])
        self.assertEqual(output.shape, value.shape)
        self.assertEqual(
            layer.compute_output_shape(
                input_shape=[query.shape, value.shape, key.shape]
            ),
            output.shape,
        )

    def test_attention_nd_inputs(self):
        """Test that Attention handles N-D inputs (e.g. 4D from Conv2D)."""
        layer = layers.Attention()
        # 4D inputs: (batch, height, width, channels)
        query = np.random.random((2, 8, 6, 4)).astype(np.float32)
        value = np.random.random((2, 8, 6, 4)).astype(np.float32)
        output = layer([query, value])
        self.assertEqual(output.shape, (2, 8, 6, 4))

        # With return_attention_scores
        output, scores = layer([query, value], return_attention_scores=True)
        self.assertEqual(output.shape, (2, 8, 6, 4))
        self.assertEqual(scores.shape, (2, 8, 6, 6))

    def test_return_attention_scores_true(self):
        """Test that the layer returns attention scores along with outputs."""
        # Generate dummy input data
        query = np.random.random((2, 8, 16)).astype(np.float32)
        value = np.random.random((2, 4, 16)).astype(np.float32)

        # Initialize the Attention layer
        layer = layers.Attention()

        # Call the layer with return_attention_scores=True
        output, attention_scores = layer(
            [query, value], return_attention_scores=True
        )

        # Check the shape of the outputs
        self.assertEqual(output.shape, (2, 8, 16))  # Output shape
        self.assertEqual(
            attention_scores.shape, (2, 8, 4)
        )  # Attention scores shape

    def test_return_attention_scores_true_and_tuple(self):
        """Test that the layer outputs are a tuple when
        return_attention_scores=True."""
        # Generate dummy input data
        query = np.random.random((2, 8, 16)).astype(np.float32)
        value = np.random.random((2, 4, 16)).astype(np.float32)

        # Initialize the Attention layer
        layer = layers.Attention()

        # Call the layer with return_attention_scores=True
        outputs = layer([query, value], return_attention_scores=True)

        # Check that outputs is a tuple
        self.assertIsInstance(
            outputs, tuple, "Expected the outputs to be a tuple"
        )

    def test_return_attention_scores_true_tuple_then_unpack(self):
        """Test that outputs can be unpacked correctly."""
        # Generate dummy input data
        query = np.random.random((2, 8, 16)).astype(np.float32)
        value = np.random.random((2, 4, 16)).astype(np.float32)

        # Initialize the Attention layer
        layer = layers.Attention()

        # Call the layer with return_attention_scores=True
        outputs = layer([query, value], return_attention_scores=True)

        # Unpack the outputs
        output, attention_scores = outputs

        # Check the shape of the unpacked outputs
        self.assertEqual(output.shape, (2, 8, 16))  # Output shape
        self.assertEqual(
            attention_scores.shape, (2, 8, 4)
        )  # Attention scores shape

    def test_return_attention_scores_with_symbolic_tensors(self):
        """Test to check outputs with symbolic tensors with
        return_attention_scores = True"""
        attention = layers.Attention()
        x = layers.Input(shape=(3, 5))
        y = layers.Input(shape=(4, 5))
        output, attention_scores = attention(
            [x, y], return_attention_scores=True
        )
        self.assertEqual(output.shape, (None, 3, 5))  # Output shape
        self.assertEqual(attention_scores.shape, (None, 3, 4))

    def test_symbolic_call_does_not_leak_return_attention_scores(self):
        # `Attention` used to stash `return_attention_scores` on an instance
        # flag in `call()`, so a later symbolic call that set it to `False`
        # still returned a `(output, scores)` tuple from `compute_output_spec`
        # and `compute_output_shape`. Both paths must trust only the explicit
        # argument.
        attention = layers.Attention()
        q_eager = np.random.rand(2, 3, 5).astype("float32")
        v_eager = np.random.rand(2, 4, 5).astype("float32")
        attention([q_eager, v_eager], return_attention_scores=True)

        x = layers.Input(shape=(3, 5))
        y = layers.Input(shape=(4, 5))
        output = attention([x, y])
        self.assertNotIsInstance(output, (tuple, list))
        self.assertEqual(output.shape, (None, 3, 5))

        self.assertEqual(
            attention.compute_output_shape([x.shape, y.shape]),
            (None, 3, 5),
        )
