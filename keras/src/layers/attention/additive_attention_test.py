import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import ops
from keras.src import testing
from keras.src.testing import test_utils


class AdditiveAttentionTest(testing.TestCase):
    def test_attention_basics(self):
        # No scale
        self.run_layer_test(
            layers.AdditiveAttention,
            init_kwargs={
                "use_scale": True,
                "dropout": 0.5,
            },
            input_shape=[(2, 3, 4), (2, 4, 4)],
            expected_output_shape=(2, 3, 4),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=1,
            expected_num_losses=0,
            supports_masking=True,
            run_training_check=False,
        )
        # With scale.
        self.run_layer_test(
            layers.AdditiveAttention,
            init_kwargs={
                "use_scale": False,
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

    def test_attention_correctness(self):
        query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        key = np.array([[[0.0, 1.0], [1.0, 0.0]]])
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        layer = layers.AdditiveAttention(use_scale=False)
        output, scores = layer(
            [query, value, key],
            return_attention_scores=True,
        )
        self.assertAllClose(
            output,
            [[[1.727, 2.727], [2.272, 3.272]]],
            atol=1e-3,
            tpu_atol=1e-2,
        )
        self.assertAllClose(
            scores,
            [[[0.636, 0.363], [0.363, 0.636]]],
            atol=1e-3,
            tpu_atol=1e-2,
        )

    def test_attention_with_mask(self):
        layer = layers.AdditiveAttention(use_scale=False)
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
        layer = layers.AdditiveAttention(dtype=dtype)
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
        # Every value position is masked, so the gradients are 0. Without a
        # scale, each of the 20 dimensions adds about -1 to every score, which
        # overflows to -inf in float16 if the masked scores are offset rather
        # than replaced, and gives NaN gradients.
        query = np.full((1, 2, 20), -10.0)
        value = np.array([[[1.0] * 20, [2.0] * 20]])
        mask = [None, np.array([[False, False]])]
        layer = layers.AdditiveAttention(use_scale=False, dtype=dtype)
        grads = ops.grad(lambda q, v: layer([q, v], mask=mask), argnums=(0, 1))(
            query.astype(dtype), value.astype(dtype)
        )
        for grad in grads:
            self.assertAllClose(grad, ops.zeros_like(grad))

    def test_attention_errors(self):
        layer = layers.AdditiveAttention()
        tensor = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        with self.assertRaisesRegex(ValueError, "must be called on a list"):
            layer(tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor, tensor, tensor])

        with self.assertRaisesRegex(ValueError, "layer mask must be a list"):
            layer([tensor, tensor], mask=tensor)

        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            layer([tensor, tensor], mask=[tensor])
