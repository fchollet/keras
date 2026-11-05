import numpy as np
import pytest

from keras.src import backend
from keras.src import testing

if backend.backend() == "tensorflow":
    from keras.src.legacy.backend import binary_crossentropy


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="`keras._legacy.backend` requires the TF backend.",
)
class LegacyBackendBceTest(testing.TestCase):
    def test_binary_crossentropy_matching_shapes(self):
        target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
        output = np.array([[0.1, 0.9], [0.8, 0.2]], dtype="float32")

        for from_logits in (False, True):
            with self.subTest(from_logits=from_logits):
                result = binary_crossentropy(
                    target, output, from_logits=from_logits
                )
                self.assertEqual(tuple(result.shape.as_list()), (2, 2))

    def test_binary_crossentropy_mismatched_rank_raises(self):
        target = np.array([0.0, 1.0, 0.0], dtype="float32")
        output = np.array([[0.1, 0.9, 0.2]], dtype="float32")

        with self.assertRaisesRegex(
            ValueError,
            "Arguments `target` and `output` must have the same rank",
        ):
            binary_crossentropy(target, output)

    def test_binary_crossentropy_mismatched_dims_raise(self):
        # Broadcasting to (2, 3) is rejected, as in the current Keras BCE
        # loss / ops APIs.
        target = np.array([[0.0], [1.0]], dtype="float32")
        output = np.array(
            [[0.2, 0.8, 0.3], [0.7, 0.1, 0.4]], dtype="float32"
        )
        with self.assertRaisesRegex(
            ValueError,
            "Arguments `target` and `output` must have the same shape.",
        ):
            binary_crossentropy(target, output)

        target = np.array([0.0, 0.0, 1.0], dtype="float32")
        output = np.array([[0.2, 0.8, 0.3]], dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            "Arguments `target` and `output` must have the same rank",
        ):
            binary_crossentropy(target, output)

    def test_binary_crossentropy_unknown_rank_skips_validation(self):
        # With an unknown rank (e.g. `tf.TensorSpec(None)` during graph
        # tracing), TensorShape raises on len()/iteration; validation must
        # be skipped instead of raising inside the shape checks.
        import tensorflow as tf

        @tf.function(
            input_signature=[tf.TensorSpec(None), tf.TensorSpec(None)]
        )
        def _bce(t, o):
            return binary_crossentropy(t, o)

        target = np.zeros((2, 2), dtype="float32")
        output = np.full((2, 2), 0.5, dtype="float32")
        result = _bce(target, output)
        self.assertEqual(tuple(result.shape.as_list()), (2, 2))
