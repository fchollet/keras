import numpy as np
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import testing


class GlobalAveragePoolingBasicTest(testing.TestCase):
    @parameterized.parameters(
        ("channels_last", False, (3, 5, 4), (3, 4)),
        ("channels_last", True, (3, 5, 4), (3, 1, 4)),
        ("channels_first", False, (3, 5, 4), (3, 5)),
    )
    def test_global_average_pooling1d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling1D,
            init_kwargs={
                "data_format": data_format,
                "keepdims": keepdims,
            },
            input_shape=input_shape,
            expected_output_shape=output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_losses=0,
            supports_masking=True,
            assert_built_after_instantiation=True,
        )

    @parameterized.parameters(
        ("channels_last", False, (3, 5, 6, 4), (3, 4)),
        ("channels_last", True, (3, 5, 6, 4), (3, 1, 1, 4)),
        ("channels_first", False, (3, 5, 6, 4), (3, 5)),
    )
    def test_global_average_pooling2d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling2D,
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
    def test_global_average_pooling3d(
        self,
        data_format,
        keepdims,
        input_shape,
        output_shape,
    ):
        self.run_layer_test(
            layers.GlobalAveragePooling3D,
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


class GlobalAveragePoolingCorrectnessTest(testing.TestCase):
    @parameterized.product(
        data_format=["channels_last", "channels_first"],
        keepdims=[False, True],
        dtype=["float32", "float64"],
    )
    def test_global_average_pooling1d_fully_masked(
        self, data_format, keepdims, dtype
    ):
        inputs = np.arange(18, dtype=dtype).reshape((3, 3, 2))
        mask = np.array(
            [[True, True, False], [False, False, False], [True, True, True]]
        )
        expected = np.array([[1.0, 2.0], [0.0, 0.0], [14.0, 15.0]])
        if data_format == "channels_first":
            inputs = np.transpose(inputs, (0, 2, 1))
        if keepdims:
            expected = np.expand_dims(
                expected, 1 if data_format == "channels_last" else 2
            )
        layer = layers.GlobalAveragePooling1D(
            data_format=data_format, keepdims=keepdims, dtype=dtype
        )
        outputs = layer(inputs, mask=mask)
        self.assertAllClose(outputs, expected)
        self.assertEqual(outputs.dtype, layer(inputs).dtype)

    @parameterized.parameters(False, True)
    def test_global_average_pooling1d_fully_masked_embedding(self, keepdims):
        model = models.Sequential(
            [
                layers.Embedding(
                    input_dim=4,
                    output_dim=2,
                    mask_zero=True,
                    embeddings_initializer="ones",
                ),
                layers.GlobalAveragePooling1D(keepdims=keepdims),
            ]
        )
        expected = np.array([[1.0, 1.0], [0.0, 0.0]])
        if keepdims:
            expected = np.expand_dims(expected, 1)
        self.assertAllClose(model(np.array([[1, 2, 0], [0, 0, 0]])), expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_average_pooling1d(self, data_format, keepdims):
        def np_gap1d(x, data_format, keepdims, mask=None):
            steps_axis = 1 if data_format == "channels_last" else 2
            if mask is not None:
                mask = np.expand_dims(
                    mask, 2 if data_format == "channels_last" else 1
                )
                x *= mask
                res = np.sum(x, axis=steps_axis) / np.sum(mask, axis=steps_axis)
            else:
                res = np.mean(x, axis=steps_axis)
            if keepdims:
                res = np.expand_dims(res, axis=steps_axis)
            return res

        inputs = np.arange(24, dtype="float32").reshape((2, 3, 4))
        layer = layers.GlobalAveragePooling1D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_gap1d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)

        if data_format == "channels_last":
            mask = np.array([[1, 1, 0], [0, 1, 0]], dtype="int32")
        else:
            mask = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], dtype="int32")
        outputs = layer(inputs, mask)
        expected = np_gap1d(inputs, data_format, keepdims, mask)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_average_pooling2d(self, data_format, keepdims):
        def np_gap2d(x, data_format, keepdims):
            steps_axis = [1, 2] if data_format == "channels_last" else [2, 3]
            res = np.apply_over_axes(np.mean, x, steps_axis)
            if not keepdims:
                res = res.squeeze()
            return res

        inputs = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))
        layer = layers.GlobalAveragePooling2D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_gap2d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)

    @parameterized.parameters(
        ("channels_last", False),
        ("channels_last", True),
        ("channels_first", False),
        ("channels_first", True),
    )
    def test_global_average_pooling3d(self, data_format, keepdims):
        def np_gap3d(x, data_format, keepdims):
            steps_axis = (
                [1, 2, 3] if data_format == "channels_last" else [2, 3, 4]
            )
            res = np.apply_over_axes(np.mean, x, steps_axis)
            if not keepdims:
                res = res.squeeze()
            return res

        inputs = np.arange(360, dtype="float32").reshape((2, 3, 3, 5, 4))
        layer = layers.GlobalAveragePooling3D(
            data_format=data_format,
            keepdims=keepdims,
        )
        outputs = layer(inputs)
        expected = np_gap3d(inputs, data_format, keepdims)
        self.assertAllClose(outputs, expected)
