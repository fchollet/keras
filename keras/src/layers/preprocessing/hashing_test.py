import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.saving import load_model
from keras.src.testing.test_utils import named_product


class ArrayLike:
    def __init__(self, values):
        self.values = values

    def __array__(self):
        return np.array(self.values)


@pytest.mark.skipif(
    backend.backend() == "numpy", reason="Broken with NumPy backend."
)
class HashingTest(testing.TestCase):
    def test_config(self):
        layer = layers.Hashing(
            num_bins=8,
            output_mode="int",
        )
        self.run_class_serialization_test(layer)

    def test_correctness(self):
        layer = layers.Hashing(num_bins=3)
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.ops.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [0], [1], [1], [2]]))

        layer = layers.Hashing(num_bins=3, mask_value="")
        inp = [["A"], ["B"], [""], ["C"], ["D"]]
        output = layer(inp)
        self.assertTrue(backend.ops.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [1], [0], [2], [2]]))

        layer = layers.Hashing(num_bins=3, salt=[133, 137])
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.ops.is_tensor(output))
        self.assertAllClose(output, np.array([[1], [2], [1], [0], [2]]))

        layer = layers.Hashing(num_bins=3, salt=133)
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        output = layer(inp)
        self.assertTrue(backend.ops.is_tensor(output))
        self.assertAllClose(output, np.array([[0], [0], [2], [1], [0]]))

    def test_tf_data_compatibility(self):
        layer = layers.Hashing(num_bins=3)
        inp = [["A"], ["B"], ["C"], ["D"], ["E"]]
        ds = tf.data.Dataset.from_tensor_slices(inp).batch(5).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([[1], [0], [1], [1], [2]]))

    @parameterized.named_parameters(
        ("list", list),
        ("tuple", tuple),
        ("numpy", np.array),
        ("array_like", ArrayLike),
    )
    def test_tensor_like_inputs(self, data_fn):
        input_data = data_fn([0, 1, 2, 3, 4])
        expected_output = [1, 0, 1, 0, 2]

        layer = layers.Hashing(num_bins=3)
        output_data = layer(input_data)
        self.assertAllClose(output_data, expected_output)

    def test_hash_single_bin(self):
        layer = layers.Hashing(num_bins=1)
        inp = np.asarray([["A"], ["B"], ["C"], ["D"], ["E"]])
        output = layer(inp)
        self.assertAllClose(output, [[0], [0], [0], [0], [0]])

    def test_hash_dense_input_farmhash(self):
        layer = layers.Hashing(num_bins=2)
        inp = np.asarray(
            [["omar"], ["stringer"], ["marlo"], ["wire"], ["skywalker"]]
        )
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        self.assertAllClose(output, [[0], [0], [1], [0], [0]])

    def test_hash_dense_input_mask_value_farmhash(self):
        empty_mask_layer = layers.Hashing(num_bins=3, mask_value="")
        omar_mask_layer = layers.Hashing(num_bins=3, mask_value="omar")
        inp = np.asarray(
            [["omar"], ["stringer"], ["marlo"], ["wire"], ["skywalker"]]
        )
        empty_mask_output = empty_mask_layer(inp)
        omar_mask_output = omar_mask_layer(inp)
        # Outputs should be one more than test_hash_dense_input_farmhash (the
        # zeroth bin is now reserved for masks).
        self.assertAllClose(empty_mask_output, [[1], [1], [2], [1], [1]])
        # 'omar' should map to 0.
        self.assertAllClose(omar_mask_output, [[0], [1], [2], [1], [1]])

    def test_hash_dense_list_input_farmhash(self):
        layer = layers.Hashing(num_bins=2)
        inp = [["omar"], ["stringer"], ["marlo"], ["wire"], ["skywalker"]]
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        self.assertAllClose(output, [[0], [0], [1], [0], [0]])

        inp = ["omar", "stringer", "marlo", "wire", "skywalker"]
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        self.assertAllClose(output, [0, 0, 1, 0, 0])

    def test_hash_dense_int_input_farmhash(self):
        layer = layers.Hashing(num_bins=3)
        inp = np.asarray([[0], [1], [2], [3], [4]])
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        self.assertAllClose(output, [[1], [0], [1], [0], [2]])

    def test_hash_dense_input_siphash(self):
        layer = layers.Hashing(num_bins=2, salt=[133, 137])
        inp = np.asarray(
            [["omar"], ["stringer"], ["marlo"], ["wire"], ["skywalker"]]
        )
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        # Note the result is different from FarmHash.
        self.assertAllClose(output, [[0], [1], [0], [1], [0]])

        layer_2 = layers.Hashing(num_bins=2, salt=[211, 137])
        output_2 = layer_2(inp)
        # Note the result is different from (133, 137).
        self.assertAllClose(output_2, [[1], [0], [1], [0], [1]])

    def test_hash_dense_int_input_siphash(self):
        layer = layers.Hashing(num_bins=3, salt=[133, 137])
        inp = np.asarray([[0], [1], [2], [3], [4]])
        output = layer(inp)
        # Assert equal for hashed output that should be true on all platforms.
        self.assertAllClose(output, [[1], [1], [2], [0], [1]])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_hash_sparse_input_farmhash(self):
        layer = layers.Hashing(num_bins=2)
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        inp = tf.SparseTensor(
            indices=indices,
            values=["omar", "stringer", "marlo", "wire", "skywalker"],
            dense_shape=[3, 2],
        )
        output = layer(inp)
        self.assertAllClose(indices, output.indices)
        self.assertAllClose(output.values, [0, 0, 1, 0, 0])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_hash_sparse_input_mask_value_farmhash(self):
        empty_mask_layer = layers.Hashing(num_bins=3, mask_value="")
        omar_mask_layer = layers.Hashing(num_bins=3, mask_value="omar")
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        inp = tf.SparseTensor(
            indices=indices,
            values=["omar", "stringer", "marlo", "wire", "skywalker"],
            dense_shape=[3, 2],
        )
        empty_mask_output = empty_mask_layer(inp)
        omar_mask_output = omar_mask_layer(inp)
        self.assertAllClose(indices, omar_mask_output.indices)
        self.assertAllClose(indices, empty_mask_output.indices)
        # Outputs should be one more than test_hash_sparse_input_farmhash (the
        # zeroth bin is now reserved for masks).
        self.assertAllClose(empty_mask_output.values, [1, 1, 2, 1, 1])
        # 'omar' should map to 0.
        self.assertAllClose(omar_mask_output.values, [0, 1, 2, 1, 1])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_hash_sparse_int_input_farmhash(self):
        layer = layers.Hashing(num_bins=3)
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        inp = tf.SparseTensor(
            indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2]
        )
        output = layer(inp)
        self.assertAllClose(indices, output.indices)
        self.assertAllClose(output.values, [1, 0, 1, 0, 2])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_hash_sparse_input_siphash(self):
        layer = layers.Hashing(num_bins=2, salt=[133, 137])
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        inp = tf.SparseTensor(
            indices=indices,
            values=["omar", "stringer", "marlo", "wire", "skywalker"],
            dense_shape=[3, 2],
        )
        output = layer(inp)
        self.assertAllClose(output.indices, indices)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose(output.values, [0, 1, 0, 1, 0])

        layer_2 = layers.Hashing(num_bins=2, salt=[211, 137])
        output = layer_2(inp)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose(output.values, [1, 0, 1, 0, 1])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_hash_sparse_int_input_siphash(self):
        layer = layers.Hashing(num_bins=3, salt=[133, 137])
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        inp = tf.SparseTensor(
            indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2]
        )
        output = layer(inp)
        self.assertAllClose(indices, output.indices)
        self.assertAllClose(output.values, [1, 1, 2, 0, 1])

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "cannot be `None`"):
            _ = layers.Hashing(num_bins=None)
        with self.assertRaisesRegex(ValueError, "cannot be `None`"):
            _ = layers.Hashing(num_bins=-1)
        with self.assertRaisesRegex(
            ValueError, "can only be a tuple of size 2"
        ):
            _ = layers.Hashing(num_bins=2, salt="string")
        with self.assertRaisesRegex(
            ValueError, "can only be a tuple of size 2"
        ):
            _ = layers.Hashing(num_bins=2, salt=[1])
        with self.assertRaisesRegex(
            ValueError, "can only be a tuple of size 2"
        ):
            _ = layers.Hashing(num_bins=1, salt=[133, 137, 177])

    @parameterized.named_parameters(
        named_product(
            mode=("int", "one_hot", "multi_hot", "count"),
            input_shape=((4,), (4, 1), (3, 4)),
        )
    )
    def test_compute_output_shape(self, mode, input_shape):
        # `compute_output_shape` must work without building the layer first
        # and agree with the actual call for every output mode.
        layer = layers.Hashing(num_bins=5, output_mode=mode)
        cos = tuple(layer.compute_output_shape((None,) + input_shape))
        actual = tuple(layer(layers.Input(shape=input_shape)).shape)
        self.assertEqual(cos, actual)

    def test_one_hot_output(self):
        input_array = np.array([0, 1, 2, 3, 4])

        expected_output = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        expected_output_shape = (None, 3)

        inputs = layers.Input(shape=(1,), dtype="int32")
        layer = layers.Hashing(num_bins=3, output_mode="one_hot")
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, expected_output_shape)

        model = models.Model(inputs, outputs)
        output_data = model(input_array)
        self.assertAllClose(output_data, expected_output)

    def test_one_hot_output_non_singleton_last_axis(self):
        inputs = layers.Input(shape=(3,), dtype="int32")
        encoded = layers.Hashing(num_bins=5, output_mode="one_hot")(inputs)
        self.assertEqual(encoded.shape, (None, 3, 5))

        flattened = layers.Flatten()(encoded)
        model = models.Model(inputs, layers.Dense(1)(flattened))
        predictions = model.predict(np.array([[1, 2, 3], [4, 5, 6]]), verbose=0)
        self.assertEqual(predictions.shape, (2, 1))

    def test_one_hot_output_preserves_singleton_axes(self):
        layer = layers.Hashing(num_bins=5, output_mode="one_hot")
        encoded = layer(np.zeros((2, 1, 1), dtype="int64"))
        self.assertEqual(encoded.shape, (2, 1, 5))

        scalar_encoded = layer(np.array(0, dtype="int64"))
        self.assertEqual(scalar_encoded.shape, (1, 5))

    def test_compute_output_spec_preserves_sparse(self):
        layer = layers.Hashing(
            num_bins=5, output_mode="one_hot", sparse=True
        )
        spec = layer.compute_output_spec(layers.Input(shape=(3,)))
        self.assertTrue(spec.sparse)
        self.assertEqual(spec.shape, (None, 3, 5))

    def test_multi_hot_output(self):
        input_array = np.array([[0, 1, 2, 3, 4]])

        expected_output = [[1.0, 1.0, 1.0]]
        expected_output_shape = (None, 3)

        inputs = layers.Input(shape=(None,), dtype="int32")
        layer = layers.Hashing(num_bins=3, output_mode="multi_hot")
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, expected_output_shape)

        model = models.Model(inputs, outputs)
        output_data = model(input_array)
        self.assertAllClose(output_data, expected_output)

    @parameterized.named_parameters(
        (
            "1d_input",
            [0, 1, 2, 3, 4],
            [2.0, 2.0, 1.0],
            (3,),
        ),
        (
            "2d_input",
            [[0, 1, 2, 3, 4]],
            [[2.0, 2.0, 1.0]],
            (None, 3),
        ),
    )
    def test_count_output(
        self, input_value, expected_output, expected_output_shape
    ):
        input_array = np.array(input_value)
        if input_array.ndim == 1:
            symbolic_sample_shape = ()
        elif input_array.ndim == 2:
            symbolic_sample_shape = (None,)
        else:
            raise TypeError("Unknown `symbolic_sample_shape`")
        inputs = layers.Input(shape=symbolic_sample_shape, dtype="int32")
        layer = layers.Hashing(num_bins=3, output_mode="count")
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, expected_output_shape)
        output_data = layer(input_array)
        self.assertAllClose(output_data, expected_output)

    @parameterized.named_parameters(
        ("int32", "int32"),
        ("int64", "int64"),
    )
    def test_int_output_dtype(self, dtype):
        input_data = layers.Input(batch_size=16, shape=(4,), dtype="string")
        layer = layers.Hashing(num_bins=3, output_mode="int", dtype=dtype)
        output = layer(input_data)
        self.assertEqual(output.dtype, dtype)

    @parameterized.named_parameters(
        ("float32", "float32"),
        ("float64", "float64"),
    )
    def test_one_hot_output_dtype(self, dtype):
        input_data = layers.Input(batch_size=16, shape=(1,), dtype="string")
        layer = layers.Hashing(num_bins=3, output_mode="one_hot", dtype=dtype)
        output = layer(input_data)
        self.assertEqual(output.dtype, dtype)

    def test_config_with_custom_name(self):
        layer = layers.Hashing(num_bins=2, name="hashing")
        config = layer.get_config()
        layer_1 = layers.Hashing.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses string dtype."
    )
    def test_saving(self):
        input_data = np.array(
            ["omar", "stringer", "marlo", "wire", "skywalker"]
        )
        inputs = layers.Input(shape=(), dtype="string")
        outputs = layers.Hashing(num_bins=100)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)

        original_output_data = model(input_data)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "keras_model.keras")
        model.save(output_path)
        loaded_model = load_model(output_path)

        # Ensure that the loaded model is unique (so that the save/load is real)
        self.assertIsNot(model, loaded_model)

        # Validate correctness of the new model.
        new_output_data = loaded_model(input_data)
        self.assertAllClose(new_output_data, original_output_data)

    @parameterized.named_parameters(
        (
            "list_input",
            [1, 2, 3],
            [1, 1, 1],
        ),
        (
            "list_input_2d",
            [[1], [2], [3]],
            [[1], [1], [1]],
        ),
        (
            "list_input_2d_multiple",
            [[1, 2], [2, 3], [3, 4]],
            [[1, 1], [1, 1], [1, 1]],
        ),
        (
            "list_input_3d",
            [[[1], [2]], [[2], [3]], [[3], [4]]],
            [[[1], [1]], [[1], [1]], [[1], [1]]],
        ),
    )
    def test_hash_list_input(self, input_data, expected):
        layer = layers.Hashing(num_bins=2)
        out_data = layer(input_data)
        self.assertAllClose(out_data, expected)

    def test_hashing_invalid_num_bins(self):
        # Test with `num_bins` set to None
        with self.assertRaisesRegex(
            ValueError,
            "The `num_bins` for `Hashing` cannot be `None` or non-positive",
        ):
            layers.Hashing(num_bins=None)

        # Test with `num_bins` set to 0
        with self.assertRaisesRegex(
            ValueError,
            "The `num_bins` for `Hashing` cannot be `None` or non-positive",
        ):
            layers.Hashing(num_bins=0)

    def test_hashing_invalid_output_mode(self):
        # Test with an unsupported `output_mode`
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument `output_mode`. Expected one of",
        ):
            layers.Hashing(num_bins=3, output_mode="unsupported_mode")

    def test_hashing_invalid_dtype_for_int_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            'When `output_mode="int"`, `dtype` should be an integer type,',
        ):
            layers.Hashing(num_bins=3, output_mode="int", dtype="float32")

    def test_hashing_sparse_with_int_mode(self):
        # Test setting `sparse=True` with `output_mode='int'`
        with self.assertRaisesRegex(
            ValueError, "`sparse` may only be true if `output_mode` is"
        ):
            layers.Hashing(num_bins=3, output_mode="int", sparse=True)


# TODO: support tf.RaggedTensor.
# def test_hash_ragged_string_input_farmhash(self):
#     layer = layers.Hashing(num_bins=2)
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[0, 0, 1, 0], [1, 0, 0]]
#     self.assertAllClose(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="string")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_input_mask_value(self):
#     empty_mask_layer = layers.Hashing(num_bins=3, mask_value="")
#     omar_mask_layer = layers.Hashing(num_bins=3, mask_value="omar")
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     empty_mask_output = empty_mask_layer(inp_data)
#     omar_mask_output = omar_mask_layer(inp_data)
#     # Outputs should be one more than test_hash_ragged_string_input_farmhash
#     # (the zeroth bin is now reserved for masks).
#     expected_output = [[1, 1, 2, 1], [2, 1, 1]]
#     self.assertAllClose(expected_output[0], empty_mask_output[1])
#     self.assertAllClose(expected_output[1], empty_mask_output[2])
#     # 'omar' should map to 0.
#     expected_output = [[0, 1, 2, 1], [2, 1, 1]]
#     self.assertAllClose(expected_output[0], omar_mask_output[0])
#     self.assertAllClose(expected_output[1], omar_mask_output[1])

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_int_input_farmhash(self):
#     layer = layers.Hashing(num_bins=3)
#     inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]], dtype="int64")
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[1, 0, 0, 2], [1, 0, 1]]
#     self.assertAllClose(expected_output[0], out_data[0])
#     self.assertAllClose(expected_output[1], out_data[1])
#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="int64")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_string_input_siphash(self):
#     layer = layers.Hashing(num_bins=2, salt=[133, 137])
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_dense_input_siphash
#     expected_output = [[0, 1, 0, 1], [0, 0, 1]]
#     self.assertAllClose(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="string")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

#     layer_2 = layers.Hashing(num_bins=2, salt=[211, 137])
#     out_data = layer_2(inp_data)
#     expected_output = [[1, 0, 1, 0], [1, 1, 0]]
#     self.assertAllClose(expected_output, out_data)

#     out_t = layer_2(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_int_input_siphash(self):
#     layer = layers.Hashing(num_bins=3, salt=[133, 137])
#     inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]], dtype="int64")
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[1, 1, 0, 1], [2, 1, 1]]
#     self.assertAllClose(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="int64")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))
