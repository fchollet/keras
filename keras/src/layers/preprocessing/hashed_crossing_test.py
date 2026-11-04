import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.common import standardize_dtype
from keras.src.testing.test_utils import named_product


class HashedCrossingTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={
                "num_bins": 3,
                "output_mode": "int",
            },
            input_data=(np.array([1, 2]), np.array([4, 5])),
            expected_output_shape=(2,),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
            # Incomplete op support on tensorflow.
            run_mixed_precision_check=False,
        )
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={"num_bins": 4, "output_mode": "one_hot"},
            input_data=(np.array([1, 2]), np.array([4, 5])),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
            # Incomplete op support on tensorflow.
            run_mixed_precision_check=False,
        )

    @parameterized.named_parameters(
        named_product(
            sparse=(
                [True, False] if backend.backend() == "tensorflow" else [False]
            )
        )
    )
    def test_correctness(self, sparse):
        layer = layers.HashedCrossing(num_bins=5)
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertAllClose(tf.constant([1, 4, 1, 1, 3]), output)

        layer = layers.HashedCrossing(
            num_bins=5, output_mode="one_hot", sparse=sparse
        )
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertSparse(output, sparse)
        self.assertAllClose(
            output,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
        )

    def test_tf_data_compatibility(self):
        layer = layers.HashedCrossing(num_bins=5)
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        ds = (
            tf.data.Dataset.from_tensor_slices((feat1, feat2))
            .batch(5)
            .map(lambda x1, x2: layer((x1, x2)))
        )
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([1, 4, 1, 1, 3]))

    def test_static_shape_preserved(self):
        layer = layers.HashedCrossing(num_bins=5)

        def call_layer(x1, x2):
            result = layer((x1, x2))
            self.assertEqual(result.shape, (5,))
            return result

        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        ds = (
            tf.data.Dataset.from_tensor_slices((feat1, feat2))
            .batch(5, drop_remainder=True)
            .map(call_layer)
        )
        next(iter(ds))

    def test_unsupported_shape_input_fails(self):
        with self.assertRaisesRegex(ValueError, "inputs should have shape"):
            layers.HashedCrossing(num_bins=10)(
                (np.array([[[1.0]]]), np.array([[[1.0]]]))
            )

    @pytest.mark.xfail
    def test_cross_output_dtype(self):
        input_1, input_2 = np.array([1]), np.array([1])

        layer = layers.HashedCrossing(num_bins=2)
        output_dtype = standardize_dtype(layer((input_1, input_2)).dtype)
        self.assertEqual(output_dtype, "int64")
        layer = layers.HashedCrossing(num_bins=2, dtype="int32")
        output_dtype = standardize_dtype(layer((input_1, input_2)).dtype)
        self.assertEqual(output_dtype, "int32")
        layer = layers.HashedCrossing(num_bins=2, output_mode="one_hot")
        output_dtype = standardize_dtype(layer((input_1, input_2)).dtype)
        self.assertEqual(output_dtype, "float32")
        layer = layers.HashedCrossing(
            num_bins=2, output_mode="one_hot", dtype="float64"
        )
        output_dtype = standardize_dtype(layer((input_1, input_2)).dtype)
        self.assertEqual(output_dtype, "float64")

    def test_non_list_input_fails(self):
        with self.assertRaisesRegex(ValueError, "should be called on a list"):
            layers.HashedCrossing(num_bins=10)(np.array(1))

    def test_single_input_fails(self):
        with self.assertRaisesRegex(ValueError, "at least two inputs"):
            layers.HashedCrossing(num_bins=10)([np.array(1)])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Need sparse tensor support.",
    )
    def test_sparse_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "inputs should be dense tensors"
        ):
            sparse_in = tf.sparse.from_dense(np.array([1]))
            layers.HashedCrossing(num_bins=10)((sparse_in, sparse_in))

    def test_float_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "should have an integer or string"
        ):
            layers.HashedCrossing(num_bins=10)(
                (np.array([1.0]), np.array([1.0]))
            )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Need string tensor support.",
    )
    def test_tf_string(self):
        layer = layers.HashedCrossing(num_bins=10)
        feat1 = tf.constant("A")
        feat2 = tf.constant(101)
        outputs = layer((feat1, feat2))
        self.assertAllClose(outputs, 1)

        layer = layers.HashedCrossing(num_bins=5, output_mode="one_hot")
        feat1 = tf.constant(["A", "B", "A", "B", "A"])
        feat2 = tf.constant([101, 101, 101, 102, 102])
        self.assertAllClose(
            tf.constant(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
            layer((feat1, feat2)),
        )

        layer = layers.HashedCrossing(num_bins=5)
        feat1 = tf.constant(["A", "B", "A", "B", "A"])
        feat2 = tf.constant([101, 101, 101, 102, 102])
        self.assertAllClose(tf.constant([1, 4, 1, 1, 3]), layer((feat1, feat2)))

        layer = layers.HashedCrossing(
            num_bins=5, output_mode="one_hot", sparse=True
        )
        cloned_layer = layers.HashedCrossing.from_config(layer.get_config())
        feat1 = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
        feat2 = tf.constant([[101], [101], [101], [102], [102]])
        original_outputs = layer((feat1, feat2))
        cloned_outputs = cloned_layer((feat1, feat2))
        self.assertAllClose(
            tf.sparse.to_dense(cloned_outputs),
            tf.sparse.to_dense(original_outputs),
        )
