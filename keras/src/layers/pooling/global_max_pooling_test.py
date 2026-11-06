import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import testing


class GlobalMaxPoolingBasicTest(testing.TestCase):
    @parameterized.parameters(
        ("channels_last", False, (3, 5, 4), (3, 4)),
        ("channels_last", True, (3, 5, 4), (3, 1, 4)),
        ("channels_first", False, (3, 5, 4), (3, 5)),
    )
    def test_global_max_pooling1d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalMaxPooling1D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
            assert_built_after_instantiation=True,
        )

    @parameterized.parameters(
        ("channels_last", False, (3, 5, 6, 4), (3, 4)),
        ("channels_last", True, (3, 5, 6, 4), (3, 1, 1, 4)),
        ("channels_first", False, (3, 5, 6, 4), (3, 5)),
    )
    def test_global_max_pooling2d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalMaxPooling2D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
            assert_built_after_instantiation=True,
        )

    @parameterized.parameters(
        ("channels_last", False, (3, 5, 6, 5, 4), (3, 4)),
        ("channels_last", True, (3, 5, 6, 5, 4), (3, 1, 1, 1, 4)),
        ("channels_first", False, (3, 5, 6, 5, 4), (3, 5)),
    )
    def test_global_max_pooling3d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalMaxPooling3D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=False,
            assert_built_after_instantiation=True,
        )


class GlobalMaxPoolingCorrectnessTest(testing.TestCase):
    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_max_pooling1d(self, data_format, keepdims):
        def np_global_max_pool1d(x, data_format, keepdims):
            steps_axis = [1] if data_format == "channels_last" else [2]
            res = np.apply_over_axes(np.max, x, steps_axis)
            if not keepdims:
                res = res.squeeze()
            return res

        inputs = np.arange(24, dtype="float32").reshape((2, 3, 4))
        layer = layers.GlobalMaxPooling1D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_global_max_pool1d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_max_pooling2d(self, data_format, keepdims):
        def np_global_max_pool2d(x, data_format, keepdims):
            steps_axis = [1, 2] if data_format == "channels_last" else [2, 3]
            res = np.apply_over_axes(np.max, x, steps_axis)
            if not keepdims:
                res = res.squeeze()
            return res

        inputs = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))
        layer = layers.GlobalMaxPooling2D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_global_max_pool2d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_max_pooling3d(self, data_format, keepdims):
        def np_global_max_pool3d(x, data_format, keepdims):
            steps_axis = (
                [1, 2, 3] if data_format == "channels_last" else [2, 3, 4]
            )
            res = np.apply_over_axes(np.max, x, steps_axis)
            if not keepdims:
                res = res.squeeze()
            return res

        inputs = np.arange(360, dtype="float32").reshape((2, 3, 3, 5, 4))
        layer = layers.GlobalMaxPooling3D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_global_max_pool3d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        (layers.GlobalMaxPooling1D, (0, 3, 4)),
        (layers.GlobalMaxPooling2D, (0, 3, 3, 4)),
        (layers.GlobalMaxPooling3D, (0, 3, 3, 3, 4)),
    )
    def test_global_max_pooling_zero_batch(self, layer_cls, input_shape):
        # The pooled axes are not empty, so an empty batch gives an empty
        # batch of channel vectors.
        outputs = layer_cls()(np.zeros(input_shape, "float32"))
        self.assertEqual(tuple(outputs.shape), (0, 4))
