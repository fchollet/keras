import grain
import numpy as np
import pytest
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.testing.test_utils import named_product


class ResizingTest(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            interpolation=["nearest", "bilinear", "bicubic", "lanczos5"],
            crop_pad=[(False, False), (True, False), (False, True)],
            antialias=[False, True],
            data_format=["channels_last", "channels_first"],
        )
    )
    def test_resizing_basics(
        self,
        interpolation,
        crop_pad,
        antialias,
        data_format,
    ):
        crop_to_aspect_ratio, pad_to_aspect_ratio = crop_pad
        if data_format == "channels_last":
            input_shape = (2, 12, 12, 3)
            expected_output_shape = (2, 6, 6, 3)
        else:
            input_shape = (2, 3, 12, 12)
            expected_output_shape = (2, 3, 6, 6)

        self.run_layer_test(
            layers.Resizing,
            init_kwargs={
                "height": 6,
                "width": 6,
                "interpolation": interpolation,
                "crop_to_aspect_ratio": crop_to_aspect_ratio,
                "pad_to_aspect_ratio": pad_to_aspect_ratio,
                "antialias": antialias,
                "data_format": data_format,
            },
            input_shape=input_shape,
            expected_output_shape=expected_output_shape,
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_down_sampling_numeric(self, data_format):
        img = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype(np.float32)
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=2, width=2, interpolation="nearest", data_format=data_format
        )(img)
        ref_out = (
            np.asarray([[5, 7], [13, 15]])
            .astype(np.float32)
            .reshape((1, 2, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(out, ref_out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_up_sampling_numeric(self, data_format):
        img = np.reshape(np.arange(0, 4), (1, 2, 2, 1)).astype(np.float32)
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=4,
            width=4,
            interpolation="nearest",
            data_format=data_format,
        )(img)
        ref_out = (
            np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
            .astype(np.float32)
            .reshape((1, 4, 4, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(out, ref_out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_crop_to_aspect_ratio(self, data_format):
        img = np.reshape(np.arange(0, 16), (1, 4, 4, 1)).astype("float32")
        if data_format == "channels_first":
            img = img.transpose(0, 3, 1, 2)
        out = layers.Resizing(
            height=4,
            width=2,
            interpolation="nearest",
            data_format=data_format,
            crop_to_aspect_ratio=True,
        )(img)
        ref_out = (
            np.asarray(
                [
                    [1, 2],
                    [5, 6],
                    [9, 10],
                    [13, 14],
                ]
            )
            .astype("float32")
            .reshape((1, 4, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(0, 3, 1, 2)
        self.assertAllClose(out, ref_out)

    @parameterized.parameters([("channels_first",), ("channels_last",)])
    def test_unbatched_image(self, data_format):
        img = np.reshape(np.arange(0, 16), (4, 4, 1)).astype("float32")
        if data_format == "channels_first":
            img = img.transpose(2, 0, 1)
        out = layers.Resizing(
            2, 2, interpolation="nearest", data_format=data_format
        )(img)
        ref_out = (
            np.asarray(
                [
                    [5, 7],
                    [13, 15],
                ]
            )
            .astype("float32")
            .reshape((2, 2, 1))
        )
        if data_format == "channels_first":
            ref_out = ref_out.transpose(2, 0, 1)
        self.assertAllClose(out, ref_out)

    def test_tf_data_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Resizing(8, 9)
        input_data = np.random.random(input_shape)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        output = next(iter(ds)).numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    def test_grain_compatibility(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Resizing(8, 9)
        input_data = np.random.random(input_shape)
        ds = (
            grain.MapDataset.source(input_data)
            .to_iter_dataset()
            .batch(2)
            .map(layer)
        )
        output = next(iter(ds))
        output_np = backend.ops.convert_to_numpy(output)

        self.assertEqual(tuple(output_np.shape), output_shape)
        self.assertTrue(backend.ops.is_tensor(output))
        # Ensure the device of the data is on CPU.
        if backend.backend() == "tensorflow":
            self.assertIn("CPU", str(output.device).upper())
        elif backend.backend() == "jax":
            self.assertIn("CPU", str(output.device).upper())
        elif backend.backend() == "torch":
            self.assertEqual("cpu", str(output.device))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Sequential + tf.data only works with TF backend",
    )
    def test_tf_data_compatibility_sequential(self):
        # Test compatibility when wrapping in a Sequential
        # https://github.com/keras-team/keras/issues/347
        if backend.config.image_data_format() == "channels_last":
            input_shape = (2, 10, 12, 3)
            output_shape = (2, 8, 9, 3)
        else:
            input_shape = (2, 3, 10, 12)
            output_shape = (2, 3, 8, 9)
        layer = layers.Resizing(8, 9)
        input_data = np.random.random(input_shape)
        ds = (
            tf_data.Dataset.from_tensor_slices(input_data)
            .batch(2)
            .map(Sequential([layer]))
        )
        output = next(iter(ds)).numpy()
        self.assertEqual(tuple(output.shape), output_shape)

    @parameterized.parameters(
        [((15, 10), "channels_last"), ((15, 100), "channels_last")]
    )
    def test_data_stretch(self, size, data_format):
        img = np.random.rand(1, 1, 4, 4)
        output = layers.Resizing(
            size[0], size[1], data_format=data_format, crop_to_aspect_ratio=True
        )(img)
        self.assertEqual(output.shape, (1, *size, 4))

    @parameterized.named_parameters(
        (
            "with_pad_to_aspect_ratio",
            True,
            False,
            [[6.0, 2.0, 10.0, 6.0], [14.0, 8.0, 18.0, 12.0]],
        ),
        (
            "with_crop_to_aspect_ratio",
            False,
            True,
            [[5.0, 0.0, 10.0, 5.0], [15.0, 7.5, 20.0, 12.5]],
        ),
        (
            "boxes_stretch",
            False,
            False,
            [[5.0, 2.0, 10.0, 6.0], [15.0, 8.0, 20.0, 12.0]],
        ),
    )
    def test_resize_bounding_boxes(
        self, pad_to_aspect_ratio, crop_to_aspect_ratio, expected_boxes
    ):
        if backend.config.image_data_format() == "channels_last":
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
            ),  # Example boxes (normalized)
            "labels": np.array([[1, 2]]),  # Dummy labels
        }
        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        resizing_layer = layers.Resizing(
            height=20,
            width=20,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            bounding_box_format="xyxy",
        )
        output = resizing_layer(input_data)
        self.assertAllClose(output["bounding_boxes"]["boxes"], expected_boxes)

    @parameterized.named_parameters(
        # The image path crops the 65x97 input to 65x96 at offset (0, 0),
        # then resizes it to 53x79.
        (
            "crop",
            False,
            True,
            (65, 97),
            (53, 79),
            [15.964583, 10.6, 63.858333, 42.4],
        ),
        # The image path pads the 31x47 input to 31x47 (no padding), then
        # resizes it to 19x29.
        (
            "pad",
            True,
            False,
            (31, 47),
            (19, 29),
            [5.8, 3.8, 23.2, 15.2],
        ),
        # Padding with an odd remainder: the image path pads the 10x5 input
        # by 2 columns on each side (9 columns, one less than
        # `pad_width=10`), then resizes it to 20x20.
        (
            "pad_odd_remainder",
            True,
            False,
            (10, 5),
            (20, 20),
            [6.666667, 4.0, 13.333333, 16.0],
        ),
    )
    def test_resize_bounding_boxes_follow_image_geometry(
        self,
        pad_to_aspect_ratio,
        crop_to_aspect_ratio,
        input_size,
        target_size,
        expected_box,
    ):
        # Regression test for
        # https://github.com/keras-team/keras/issues/23772: boxes must follow
        # the integer crop/pad geometry that is applied to the images.
        height, width = input_size
        if backend.config.image_data_format() == "channels_last":
            image_shape = (1, height, width, 1)
        else:
            image_shape = (1, 1, height, width)
        box = [0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height]
        input_data = {
            "images": np.zeros(image_shape, "float32"),
            "bounding_boxes": {
                "boxes": np.array([[box]], "float32"),
                "labels": np.array([[1]]),
            },
        }
        resizing_layer = layers.Resizing(
            height=target_size[0],
            width=target_size[1],
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            bounding_box_format="xyxy",
        )
        output = resizing_layer(input_data)
        self.assertAllClose(
            output["bounding_boxes"]["boxes"], [[expected_box]], atol=1e-4
        )

    @parameterized.named_parameters(
        (
            "with_pad_to_aspect_ratio",
            True,
            False,
            [[6.0, 2.0, 10.0, 6.0], [14.0, 8.0, 18.0, 12.0]],
        ),
        (
            "with_crop_to_aspect_ratio",
            False,
            True,
            [[5.0, 0.0, 10.0, 5.0], [15.0, 7.5, 20.0, 12.5]],
        ),
        (
            "boxes_stretch",
            False,
            False,
            [[5.0, 2.0, 10.0, 6.0], [15.0, 8.0, 20.0, 12.0]],
        ),
    )
    def test_resize_tf_data_bounding_boxes(
        self, pad_to_aspect_ratio, crop_to_aspect_ratio, expected_boxes
    ):
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
            ),  # Example boxes (normalized)
            "labels": np.array([[1, 2]]),  # Dummy labels
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        resizing_layer = layers.Resizing(
            height=20,
            width=20,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            bounding_box_format="xyxy",
        )
        ds = ds.map(resizing_layer)
        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["bounding_boxes"]["boxes"], expected_boxes)
