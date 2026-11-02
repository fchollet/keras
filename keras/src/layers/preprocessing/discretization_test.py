import os

import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.common import is_int_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.saving import saving_api
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset


class DiscretizationTest(testing.TestCase):
    def test_discretization_basics(self):
        self.run_layer_test(
            layers.Discretization,
            init_kwargs={
                "bin_boundaries": [0.0, 0.5, 1.0],
            },
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_adapt_flow(self):
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        output = layer(np.array([[0.0, 0.1, 0.3]]))
        self.assertTrue(output.dtype, "int32")
        self.assertLen(layer.bin_boundaries, 3)

    def test_adapt_with_generator(self):
        def data_gen():
            for _ in range(5):
                yield np.random.uniform(0, 10, size=(20,))

        layer = layers.Discretization(num_bins=4)
        layer.adapt(data_gen())
        self.assertLen(layer.bin_boundaries, 3)
        output = layer(np.array([[1.0, 5.0, 9.0]]))
        self.assertEqual(output.shape, (1, 3))

    def test_adapt_with_infinite_generator_and_steps(self):
        def data_gen():
            while True:
                yield np.random.uniform(0, 10, size=(20,))

        layer = layers.Discretization(num_bins=4)
        layer.adapt(data_gen(), steps=5)
        self.assertLen(layer.bin_boundaries, 3)

    def test_adapt_with_list_of_arrays(self):
        batches = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        layer = layers.Discretization(num_bins=3)
        layer.adapt(batches)
        self.assertLen(layer.bin_boundaries, 2)

    def test_adapt_with_grain_dataset(self):
        grain = pytest.importorskip("grain")
        raw = np.random.uniform(0, 10, size=(100,)).astype("float32")

        class Source(grain.sources.RandomAccessDataSource):
            def __getitem__(self, idx):
                return raw[idx : idx + 10]

            def __len__(self):
                return 10

        dataset = grain.MapDataset.source(Source()).batch(batch_size=5)
        layer = layers.Discretization(num_bins=4)
        layer.adapt(dataset)
        self.assertLen(layer.bin_boundaries, 3)

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "int",
                    "output_mode": "int",
                    "input_array": [[-1.0, 0.0, 0.1, 0.8, 1.2]],
                    "expected_output": [[0, 1, 1, 2, 3]],
                },
                {
                    "testcase_name": "one_hot_rank_1",
                    "output_mode": "one_hot",
                    "input_array": [0.1, 0.8],
                    "expected_output": [[0, 1, 0, 0], [0, 0, 1, 0]],
                },
                {
                    "testcase_name": "multi_hot_rank_2",
                    "output_mode": "multi_hot",
                    "input_array": [[0.1, 0.8]],
                    "expected_output": [[0, 1, 1, 0]],
                },
                {
                    "testcase_name": "one_hot_rank_3",
                    "output_mode": "one_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        ]
                    ],
                },
                {
                    "testcase_name": "multi_hot_rank_3",
                    "output_mode": "multi_hot",
                    "input_array": [[[0.15, 0.75], [0.85, 0.45]]],
                    "expected_output": [
                        [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
                    ],
                },
                {
                    "testcase_name": "count",
                    "output_mode": "count",
                    "input_array": [[0.1, 0.8, 0.9]],
                    "expected_output": [[0, 1, 2, 0]],
                },
            ],
            sparse=(
                [True, False] if backend.SUPPORTS_SPARSE_TENSORS else [False]
            ),
        )
    )
    def test_correctness(
        self, output_mode, input_array, expected_output, sparse
    ):
        if output_mode == "int" and sparse:
            pytest.skip("sparse=True cannot be combined with output_mode=int")

        input_array = np.array(input_array)
        expected_output = np.array(expected_output)

        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.5, 1.0],
            output_mode=output_mode,
            sparse=sparse,
        )
        output = layer(input_array)
        self.assertSparse(output, sparse)
        self.assertTrue(backend.ops.is_tensor(output))
        self.assertAllClose(output, expected_output)

    def test_tf_data_compatibility(self):
        # With fixed bins
        layer = layers.Discretization(
            bin_boundaries=[0.0, 0.35, 0.5, 1.0], dtype="float32"
        )
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        x = np.array([[0.0, 0.1, 0.3]])
        ds = tf_data.Dataset.from_tensor_slices(x).batch(1).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_serialization(self):
        layer = layers.Discretization(num_bins=5)

        # Serialization before `adapt` is called.
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization after `adapt` is called but `num_bins` was not reached.
        layer.adapt(np.array([0.0, 1.0, 5.0]))
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization after `adapt` is called and `num_bins` is reached.
        layer.adapt(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

        # Serialization with `bin_boundaries`.
        layer = layers.Discretization(bin_boundaries=[0.0, 0.35, 0.5, 1.0])
        config = layer.get_config()
        revived_layer = layers.Discretization.from_config(config)
        self.assertEqual(config, revived_layer.get_config())

    def test_saving(self):
        # With fixed bins
        layer = layers.Discretization(bin_boundaries=[0.0, 0.35, 0.5, 1.0])
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        x = np.array([[-1.0, 0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.2, 0.98]])
        self.assertAllClose(layer(x), np.array([[0, 1, 1, 1, 2, 3, 4, 4, 3]]))

        # With adapt flow
        layer = layers.Discretization(num_bins=4)
        layer.adapt(
            np.random.random((32, 3)),
        )
        ref_input = np.random.random((1, 2))
        ref_output = layer(ref_input)
        model = models.Sequential(
            [
                layers.Input((2,)),
                layer,
            ]
        )
        fpath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(fpath)
        model = saving_api.load_model(fpath)
        self.assertAllClose(layer(ref_input), ref_output)

    def test_init_num_bins_and_bin_boundaries_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Both `num_bins` and `bin_boundaries`"
        ):
            layers.Discretization(num_bins=3, bin_boundaries=[0.0, 1.0])

        with self.assertRaisesRegex(
            ValueError, "either `num_bins` or `bin_boundaries`"
        ):
            layers.Discretization()

    def test_call_before_adapt_raises(self):
        layer = layers.Discretization(num_bins=3)
        with self.assertRaisesRegex(ValueError, "You need .* call .*adapt"):
            layer([[0.1, 0.8, 0.9]])

    def test_model_call_vs_predict_consistency(self):
        """Test that model(input) and model.predict(input) produce consistent
        outputs."""
        # Test with int output mode
        layer = layers.Discretization(
            bin_boundaries=[-0.5, 0, 0.1, 0.2, 3],
            output_mode="int",
        )
        x = np.array([[0.0, 0.15, 0.21, 0.3], [0.0, 0.17, 0.451, 7.8]])

        # Create model
        inputs = layers.Input(shape=(4,), dtype="float32")
        outputs = layer(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)

        # Test both execution modes
        model_call_output = model(x)
        predict_output = model.predict(x)

        # Check consistency
        self.assertAllClose(model_call_output, predict_output)
        self.assertEqual(
            standardize_dtype(model_call_output.dtype),
            standardize_dtype(predict_output.dtype),
        )
        self.assertTrue(is_int_dtype(model_call_output.dtype))

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "int_mode",
                    "output_mode": "int",
                    "input_shape": (3, 4),
                    "expected_shape": (None, 3, 4),  # int mode - no change
                },
                {
                    "testcase_name": "one_hot_mode",
                    "output_mode": "one_hot",
                    "input_shape": (3, 4),
                    "expected_shape": (None, 3, 4, 5),  # one_hot - add dim
                },
                {
                    "testcase_name": "multi_hot_mode",
                    "output_mode": "multi_hot",
                    "input_shape": (3, 4),
                    "expected_shape": (None, 4, 5),  # multi_hot - replace
                },
                {
                    "testcase_name": "count_mode",
                    "output_mode": "count",
                    "input_shape": (3, 4),
                    "expected_shape": (None, 4, 5),  # count - replace
                },
            ]
        )
    )
    def test_symbolic_tensor_output_shape(
        self, output_mode, input_shape, expected_shape
    ):
        """Test symbolic tensors have correct output shape for modes."""
        # Create layer with bin_boundaries that create 5 bins (4 boundaries)
        layer = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0, 5.0], output_mode=output_mode
        )

        # Create symbolic input and get output
        symbolic_input = layers.Input(shape=input_shape)
        symbolic_output = layer(symbolic_input)

        # Verify symbolic output shape
        self.assertEqual(symbolic_output.shape, expected_shape)

        eager_input = np.random.uniform(0, 3, size=(2,) + input_shape)
        eager_output = layer(eager_input)

        # Verify batch dimension is preserved correctly
        self.assertEqual(eager_output.shape[0], 2)  # Batch size preserved
        self.assertEqual(
            symbolic_output.shape[0], None
        )  # Batch is None for symbolic

        # Verify non-batch dimensions are identical
        self.assertEqual(eager_output.shape[1:], symbolic_output.shape[1:])

        # Verify total number of dimensions is the same
        self.assertEqual(len(eager_output.shape), len(symbolic_output.shape))

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "int_mode",
                "output_mode": "int",
                "input_shape": (None, 3, 4),
                "expected_shape": (None, 3, 4),  # int mode - no change
            },
            {
                "testcase_name": "one_hot_mode",
                "output_mode": "one_hot",
                "input_shape": (None, 3, 4),
                "expected_shape": (None, 3, 4, 3),  # one_hot - add dim
            },
            {
                "testcase_name": "multi_hot_mode",
                "output_mode": "multi_hot",
                "input_shape": (None, 3, 4),
                "expected_shape": (None, 4, 3),  # multi_hot - replace
            },
            {
                "testcase_name": "count_mode",
                "output_mode": "count",
                "input_shape": (None, 3, 4),
                "expected_shape": (None, 4, 3),  # count - replace
            },
        ]
    )
    def test_compute_output_shape_modes(
        self, output_mode, input_shape, expected_shape
    ):
        """Test compute_output_shape with different output modes."""
        layer = layers.Discretization(
            bin_boundaries=[0.0, 1.0], output_mode=output_mode
        )
        result_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(result_shape, expected_shape)

    def test_compute_output_shape_edge_cases(self):
        """Test edge cases in compute_output_shape to improve coverage."""

        # Test edge case - last dimension is 1 with one_hot
        layer_one_hot = layers.Discretization(
            bin_boundaries=[0.0, 1.0], output_mode="one_hot"
        )

        # Should replace last dimension of 1 with depth
        shape = layer_one_hot.compute_output_shape((None, 5, 1))
        expected = (None, 5, 3)  # 2 boundaries = 3 bins, replace last dim
        self.assertEqual(shape, expected)

        # Test empty input shape
        shape = layer_one_hot.compute_output_shape(())
        expected = (3,)  # Just depth
        self.assertEqual(shape, expected)

    def test_compute_output_spec_method(self):
        """Test compute_output_spec method directly."""

        layer = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="one_hot"
        )

        # Create a KerasTensor input
        input_tensor = backend.KerasTensor(shape=(None, 3, 4), dtype="float32")

        # Test compute_output_spec
        output_spec = layer.compute_output_spec(input_tensor)

        # Verify shape and dtype
        expected_shape = (None, 3, 4, 4)  # 3 boundaries = 4 bins
        self.assertEqual(output_spec.shape, expected_shape)
        self.assertEqual(output_spec.dtype, layer.output_dtype)

    @parameterized.named_parameters(named_product(batch_size=[1, 3, 5, 10]))
    def test_batch_dimension_consistency(self, batch_size):
        """Test that batch dimensions are handled consistently."""

        layer = layers.Discretization(
            bin_boundaries=[0.0, 1.0, 2.0], output_mode="one_hot"
        )

        # Test different batch sizes
        input_shape = (4, 3)

        # Create eager input with specific batch size
        eager_input = np.random.uniform(0, 3, size=(batch_size,) + input_shape)
        eager_output = layer(eager_input)

        # Create symbolic input
        symbolic_input = layers.Input(shape=input_shape)
        symbolic_output = layer(symbolic_input)

        # Verify batch dimension handling
        self.assertEqual(
            eager_output.shape[0],
            batch_size,
            f"Eager batch size should be {batch_size}",
        )
        self.assertEqual(
            symbolic_output.shape[0],
            None,
            "Symbolic batch size should be None",
        )

        # Verify non-batch dimensions are identical
        self.assertEqual(
            eager_output.shape[1:],
            symbolic_output.shape[1:],
            "Non-batch dimensions should be identical",
        )

        # Verify expected output shape
        expected_shape = input_shape + (4,)  # 3 boundaries = 4 bins
        self.assertEqual(eager_output.shape[1:], expected_shape)
        self.assertEqual(symbolic_output.shape[1:], expected_shape)

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "int_mode_3_bins",
                    "num_bins": 3,
                    "output_mode": "int",
                    "input_shape": (None, 5, 4),
                    "expected_shape": (None, 5, 4),  # int mode - no change
                },
                {
                    "testcase_name": "one_hot_mode_4_bins",
                    "num_bins": 4,
                    "output_mode": "one_hot",
                    "input_shape": (None, 3, 2),
                    "expected_shape": (
                        None,
                        3,
                        2,
                        4,
                    ),  # one_hot - add depth dimension
                },
                {
                    "testcase_name": "multi_hot_mode_5_bins",
                    "num_bins": 5,
                    "output_mode": "multi_hot",
                    "input_shape": (None, 6, 3),
                    "expected_shape": (
                        None,
                        3,
                        5,
                    ),  # multi_hot - replace last dimension
                },
                {
                    "testcase_name": "count_mode_6_bins",
                    "num_bins": 6,
                    "output_mode": "count",
                    "input_shape": (None, 4, 2),
                    "expected_shape": (
                        None,
                        2,
                        6,
                    ),  # count - replace last dimension
                },
            ]
        )
    )
    def test_compute_output_shape_num_bins(
        self, num_bins, output_mode, input_shape, expected_shape
    ):
        """Test compute_output_shape with num_bins parameter."""

        layer = layers.Discretization(
            num_bins=num_bins, output_mode=output_mode
        )

        # Test compute_output_shape directly
        result_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(
            result_shape,
            expected_shape,
            f"Failed for num_bins={num_bins}, mode={output_mode}",
        )

    @parameterized.parameters([("x",), ("x_and_y",), ("x_y_and_weights",)])
    def test_adapt_pydataset_compat(self, pydataset_type):
        class CustomDataset(PyDataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                x = np.random.uniform(0, 10, size=(20,))
                y = np.random.randint(0, 2, size=(20,))
                weights = np.ones((20,))
                if pydataset_type == "x":
                    return x
                elif pydataset_type == "x_and_y":
                    return x, y
                elif pydataset_type == "x_y_and_weights":
                    return x, y, weights
                else:
                    raise NotImplementedError(pydataset_type)

        layer = layers.Discretization(num_bins=4)
        layer.adapt(CustomDataset())
        self.assertLen(layer.bin_boundaries, 3)
