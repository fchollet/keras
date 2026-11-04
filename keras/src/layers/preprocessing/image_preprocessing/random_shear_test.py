import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.utils import backend_utils


class RandomShearTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            layers.RandomShear,
            init_kwargs={
                "x_factor": (0.5, 1),
                "y_factor": (0.5, 1),
                "interpolation": "bilinear",
                "fill_mode": "reflect",
                "data_format": "channels_last",
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_config_with_float_factors(self):
        # A float factor `f` is used as the range `(-f, f)`. `get_config()`
        # must return a factor that the constructor accepts again.
        layer = layers.RandomShear(x_factor=0.1, y_factor=0.2, seed=1)
        self.run_class_serialization_test(layer)

    def test_random_posterization_inference(self):
        seed = 3481
        layer = layers.RandomShear(1, 1)
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_shear_pixel_level(self):
        image = np.zeros((1, 5, 5, 3))
        image[0, 1:4, 1:4, :] = 1.0
        image[0, 2, 2, :] = [0.0, 1.0, 0.0]
        image = keras.ops.convert_to_tensor(image, dtype="float32")

        data_format = backend.config.image_data_format()
        if data_format == "channels_first":
            image = keras.ops.transpose(image, (0, 3, 1, 2))

        shear_layer = layers.RandomShear(
            x_factor=(0.2, 0.3),
            y_factor=(0.2, 0.3),
            interpolation="bilinear",
            fill_mode="constant",
            fill_value=0.0,
            seed=42,
            data_format=data_format,
        )

        sheared_image = shear_layer(image)

        if data_format == "channels_first":
            sheared_image = keras.ops.transpose(sheared_image, (0, 2, 3, 1))

        original_pixel = image[0, 2, 2, :]
        sheared_pixel = sheared_image[0, 2, 2, :]
        self.assertNotAllClose(original_pixel, sheared_pixel)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomShear(1, 1)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    @parameterized.named_parameters(
        (
            "with_x_shift",
            [[1.0, 0.0]],
            [[[0.0, 1.0, 3.2, 3.0], [1.2, 4.0, 4.8, 6.0]]],
        ),
        (
            "with_y_shift",
            [[0.0, 1.0]],
            [[[2.0, 0.0, 4.0, 0.5], [6.0, 0.0, 8.0, 0.0]]],
        ),
        (
            "with_xy_shift",
            [[1.0, 1.0]],
            [[[0.0, 0.0, 3.2, 3.5], [1.2, 0.0, 4.8, 4.5]]],
        ),
    )
    def test_random_shear_bounding_boxes(self, translation, expected_boxes):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (10, 8, 3)
        else:
            image_shape = (3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [2, 1, 4, 3],
                    [6, 4, 8, 6],
                ]
            ),
            "labels": np.array([[1, 2]]),
        }
        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        layer = layers.RandomShear(
            x_factor=0.5,
            y_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "shear_factor": backend_utils.convert_tf_tensor(
                np.array(translation)
            ),
            "input_shape": image_shape,
        }
        output = layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )

        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        (
            "with_x_shift",
            [[1.0, 0.0]],
            [[[0.0, 1.0, 3.2, 3.0], [1.2, 4.0, 4.8, 6.0]]],
        ),
        (
            "with_y_shift",
            [[0.0, 1.0]],
            [[[2.0, 0.0, 4.0, 0.5], [6.0, 0.0, 8.0, 0.0]]],
        ),
        (
            "with_xy_shift",
            [[1.0, 1.0]],
            [[[0.0, 0.0, 3.2, 3.5], [1.2, 0.0, 4.8, 4.5]]],
        ),
    )
    def test_random_shear_tf_data_bounding_boxes(
        self, translation, expected_boxes
    ):
        data_format = backend.config.image_data_format()
        if backend.config.image_data_format() == "channels_last":
            image_shape = (1, 10, 8, 3)
        else:
            image_shape = (1, 3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        layer = layers.RandomShear(
            x_factor=0.5,
            y_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "shear_factor": np.array(translation),
            "input_shape": image_shape,
        }

        ds = ds.map(
            lambda x: layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["boxes"], expected_boxes)
