import math

import jax
import numpy as np
import pytest
import scipy.ndimage
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import random as krandom
from keras.src import testing
from keras.src.backend.common import dtypes
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import image as kimage
from keras.src.ops import numpy as knp
from keras.src.testing.test_utils import named_product


def _bf16_exact(shape):
    """Random float32 values exactly representable in bfloat16.

    The reconstruct_patches round-trip tests assert
    reconstruct(extract(x)) == x with a tight tolerance. extract_patches
    is conv-based, and TPUs compute float32 convolutions in bfloat16 by
    default, so arbitrary float32 values are rounded during extraction.
    Multiples of 1/256 in [0, 1) carry at most 8 significand bits, so
    the extraction stays lossless on every device.
    """
    return (np.round(np.random.random(shape) * 256.0) / 256.0).astype("float32")


class ImageOpsDynamicShapeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (None, 20, 20, 1))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (None, 1, 20, 20))

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_resize(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 15, 15, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (None, 3, 15, 15))

        x = KerasTensor([3, None, None])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (3, 15, 15))

    def test_resize_with_crop_and_pad_dynamic_shape(self):
        # Test channels_last
        x = KerasTensor([None, None, None, 3])
        out = kimage.resize(x, size=(15, 10), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (None, 15, 10, 3))
        out = kimage.resize(x, size=(15, 10), pad_to_aspect_ratio=True)
        self.assertEqual(out.shape, (None, 15, 10, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.resize(x, size=(15, 10), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (15, 10, 3))
        out = kimage.resize(x, size=(15, 10), pad_to_aspect_ratio=True)
        self.assertEqual(out.shape, (15, 10, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, None, None])
        out = kimage.resize(x, size=(15, 10), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (None, 3, 15, 10))
        out = kimage.resize(x, size=(15, 10), pad_to_aspect_ratio=True)
        self.assertEqual(out.shape, (None, 3, 15, 10))

        x = KerasTensor([3, None, None])
        out = kimage.resize(x, size=(15, 10), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (3, 15, 10))
        out = kimage.resize(x, size=(15, 10), pad_to_aspect_ratio=True)
        self.assertEqual(out.shape, (3, 15, 10))

    def test_affine_transform(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        transform = KerasTensor([None, 8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        transform = KerasTensor([None, 8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_extract_patches(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (None, 4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (None, 4, 4, 75))
        out = kimage.extract_patches(x, 5, strides=1)
        self.assertEqual(out.shape, (None, 16, 16, 75))
        out = kimage.extract_patches(x, 5, strides=(2, 3))
        self.assertEqual(out.shape, (None, 8, 6, 75))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (None, 75, 4, 4))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (None, 75, 4, 4))
        out = kimage.extract_patches(x, 5, strides=1)
        self.assertEqual(out.shape, (None, 75, 16, 16))
        out = kimage.extract_patches(x, 5, strides=(2, 3))
        self.assertEqual(out.shape, (None, 75, 8, 6))

    def test_extract_patches_3d(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 20, 3])
        p_d, p_h, p_w = 5, 5, 5
        out = kimage.extract_patches_3d(x, (p_d, p_h, p_w))
        self.assertEqual(out.shape, (None, 4, 4, 4, 375))
        out = kimage.extract_patches_3d(x, 5)
        self.assertEqual(out.shape, (None, 4, 4, 4, 375))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20, 20])
        p_d, p_h, p_w = 5, 5, 5
        out = kimage.extract_patches_3d(x, (p_d, p_h, p_w))
        self.assertEqual(out.shape, (None, 375, 4, 4, 4))
        out = kimage.extract_patches_3d(x, 5)
        self.assertEqual(out.shape, (None, 375, 4, 4, 4))

    def test_reconstruct_patches(self):
        # 2D: patches (None, 4, 4, 75) -> image (None, 20, 20, 3)
        patches = KerasTensor([None, 4, 4, 75])
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5),
            output_size=(20, 20),
            padding="valid",
        )
        self.assertEqual(out.shape, (None, 20, 20, 3))
        # Dynamic grid dims still resolve channels from flat dim.
        patches_dyn = KerasTensor([None, None, None, 75])
        out = kimage.reconstruct_patches(
            patches_dyn,
            size=(5, 5),
            output_size=(20, 20),
            padding="same",
        )
        self.assertEqual(out.shape, (None, 20, 20, 3))

    def test_reconstruct_patches_3d(self):
        # 3D: patches (None, 4, 4, 4, 375) -> volume (None, 20, 20, 20, 3)
        patches = KerasTensor([None, 4, 4, 4, 375])
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5, 5),
            output_size=(20, 20, 20),
            padding="valid",
        )
        self.assertEqual(out.shape, (None, 20, 20, 20, 3))

    def test_map_coordinates(self):
        input = KerasTensor([20, 20, None])
        coordinates = KerasTensor([3, 15, 15, None])
        out = kimage.map_coordinates(input, coordinates, 0)
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_pad_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (None, 20, 30, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (20, 30, 3))

        # Test unknown shape
        x = KerasTensor([None, None, 3])
        out = kimage.pad_images(x, 2, 3, 2, 3)
        self.assertEqual(out.shape, (None, None, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (None, 3, 20, 30))

        x = KerasTensor([3, None, None])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (3, 20, 30))

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 10, 20, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 3, 10, 20))

        x = KerasTensor([3, None, None])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))

    def test_pad_images_invalid_rank_keras_input(self):
        x = KerasTensor([None, 16])
        with self.assertRaises(ValueError):
            kimage.pad_images(x, 0, 0, target_height=16, target_width=16)

    def test_crop_images_invalid_rank_keras_input(self):
        x = KerasTensor([None, 16])
        with self.assertRaises(ValueError):
            kimage.crop_images(x, 0, 0, target_height=10, target_width=16)

    def test_pad_images_invalid_padding_params_keras_input(self):
        x = KerasTensor([None, 16, 16, 3])
        with self.assertRaisesRegex(
            ValueError,
            "Must specify exactly two of top_padding, bottom_padding, "
            "target_height",
        ):
            kimage.pad_images(x, 0, 0, 0, target_height=16, target_width=16)

    def test_crop_images_invalid_cropping_params_keras_input(self):
        x = KerasTensor([None, 16, 16, 3])
        with self.assertRaisesRegex(
            ValueError,
            "Must specify exactly two of top_cropping, bottom_cropping, "
            "target_height",
        ):
            kimage.crop_images(x, 0, 0, 0, target_height=10, target_width=16)

    def test_perspective_transform(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        start_points = KerasTensor([None, 4, 2])
        end_points = KerasTensor([None, 4, 2])
        out = kimage.perspective_transform(x, start_points, end_points)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        start_points = KerasTensor([None, 4, 2])
        end_points = KerasTensor([None, 4, 2])
        out = kimage.perspective_transform(x, start_points, end_points)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_gaussian_blur(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.gaussian_blur(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.gaussian_blur(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_elastic_transform(self):
        # Test channels_last
        x = KerasTensor([None, 20, 20, 3])
        out = kimage.elastic_transform(x)
        self.assertEqual(out.shape, (None, 20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 20, 20])
        out = kimage.elastic_transform(x)
        self.assertEqual(out.shape, (None, 3, 20, 20))

    def test_scale_and_translate(self):
        images = KerasTensor([None, 20, 20, 3])
        output_shape = (None, 25, 25, 3)
        scale = KerasTensor([2])
        translation = KerasTensor([2])
        out = kimage.scale_and_translate(
            images,
            output_shape=output_shape,
            scale=scale,
            translation=translation,
            spatial_dims=(1, 2),
            method="linear",
        )
        self.assertEqual(out.shape, output_shape)


class ImageOpsStaticShapeTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (20, 20, 1))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.rgb_to_grayscale(x)
        self.assertEqual(out.shape, (1, 20, 20))

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.rgb_to_hsv(x)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.hsv_to_rgb(x)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_resize(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (15, 15, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.resize(x, size=(15, 15))
        self.assertEqual(out.shape, (3, 15, 15))

    def test_affine_transform(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        transform = KerasTensor([8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        transform = KerasTensor([8])
        out = kimage.affine_transform(x, transform)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_extract_patches(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (4, 4, 75))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (4, 4, 75))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        p_h, p_w = 5, 5
        out = kimage.extract_patches(x, (p_h, p_w))
        self.assertEqual(out.shape, (75, 4, 4))
        out = kimage.extract_patches(x, 5)
        self.assertEqual(out.shape, (75, 4, 4))

    def test_extract_patches_3d(self):
        # Test channels_last
        x = KerasTensor([20, 20, 20, 3])
        p_d, p_h, p_w = 5, 5, 5
        out = kimage.extract_patches_3d(x, (p_d, p_h, p_w))
        self.assertEqual(out.shape, (4, 4, 4, 375))
        out = kimage.extract_patches_3d(x, 5)
        self.assertEqual(out.shape, (4, 4, 4, 375))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20, 20])
        p_d, p_h, p_w = 5, 5, 5
        out = kimage.extract_patches_3d(x, (p_d, p_h, p_w))
        self.assertEqual(out.shape, (375, 4, 4, 4))
        out = kimage.extract_patches_3d(x, 5)
        self.assertEqual(out.shape, (375, 4, 4, 4))

    def test_reconstruct_patches(self):
        # 2D unbatched: patches (4, 4, 75) -> image (20, 20, 3)
        patches = KerasTensor([4, 4, 75])
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5),
            output_size=(20, 20),
            padding="valid",
        )
        self.assertEqual(out.shape, (20, 20, 3))

    def test_reconstruct_patches_3d(self):
        # 3D unbatched: patches (4, 4, 4, 375) -> volume (20, 20, 20, 3)
        patches = KerasTensor([4, 4, 4, 375])
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5, 5),
            output_size=(20, 20, 20),
            padding="valid",
        )
        self.assertEqual(out.shape, (20, 20, 20, 3))

    def test_reconstruct_patches_output_size_not_tuple(self):
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(TypeError, "tuple or list"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=20,
                padding="valid",
            )
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(TypeError, "tuple or list"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=20,
                padding="valid",
            )

    def test_reconstruct_patches_output_size_out_of_range_same(self):
        # 2D: gH=4, pH=5 → valid H range is (15, 20]. 30 is too large.
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "padding='same'.*height"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(30, 20),
                padding="same",
            )
        with self.assertRaisesRegex(ValueError, "padding='same'.*width"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 30),
                padding="same",
            )
        # 3D: gD=4, pD=5 → valid D range is (15, 20]. 10 is too small.
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(ValueError, "padding='same'.*depth"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(10, 20, 20),
                padding="same",
            )
        with self.assertRaisesRegex(ValueError, "padding='same'.*height"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 30, 20),
                padding="same",
            )
        with self.assertRaisesRegex(ValueError, "padding='same'.*width"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 20, 30),
                padding="same",
            )

    def test_reconstruct_patches_invalid_size_length(self):
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "length 2 or 3"):
            kimage.reconstruct_patches(
                patches,
                size=(5,),
                output_size=(20, 20),
                padding="valid",
            )

    def test_reconstruct_patches_invalid_size_type(self):
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(TypeError, "int or a tuple"):
            kimage.reconstruct_patches(
                patches,
                size="not_a_size",
                output_size=(20, 20),
                padding="valid",
            )

    def test_reconstruct_patches_invalid_padding(self):
        # _reconstruct_patches_2d raises for bad padding even though the
        # public `reconstruct_patches` op doesn't pre-validate it.
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 20),
                padding="reflect",
            )
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(ValueError, "'same' or 'valid'"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 20, 20),
                padding="reflect",
            )

    def test_reconstruct_patches_invalid_output_size_length(self):
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "length 2"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 20, 20),
                padding="valid",
            )
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(ValueError, "length 3"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 20),
                padding="valid",
            )

    def test_reconstruct_patches_valid_size_mismatch(self):
        # valid padding requires (grid - 1) * stride + size == output_size.
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "requires output_size"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(19, 20),
                padding="valid",
            )
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(ValueError, "requires output_size"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 19, 20),
                padding="valid",
            )

    def test_reconstruct_patches_unbatched_and_int_size(self):
        # Unbatched input (3D for 2D, 4D for 3D) exercises the squeeze path.
        # An int `size` exercises the int->tuple normalization (2D only;
        # the 3D case requires an explicit length-3 tuple).
        patches = np.zeros((4, 4, 75), dtype="float32")
        out = kimage.reconstruct_patches(
            patches,
            size=5,
            output_size=(20, 20),
            padding="valid",
        )
        self.assertEqual(tuple(out.shape), (20, 20, 3))
        patches_3d = np.zeros((4, 4, 4, 375), dtype="float32")
        out = kimage.reconstruct_patches(
            patches_3d,
            size=(5, 5, 5),
            output_size=(20, 20, 20),
            padding="valid",
        )
        self.assertEqual(tuple(out.shape), (20, 20, 20, 3))

    def test_reconstruct_patches_dispatches_to_3d(self):
        # `reconstruct_patches` dispatches to the 3D path when size is a
        # length-3 tuple.
        patches = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5, 5),
            output_size=(20, 20, 20),
            padding="valid",
        )
        self.assertEqual(tuple(out.shape), (1, 20, 20, 20, 3))

    def test_reconstruct_patches_int_strides(self):
        # int strides argument is broadcast to a tuple inside the validator.
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5),
            output_size=(20, 20),
            strides=5,
            padding="valid",
        )
        self.assertEqual(tuple(out.shape), (1, 20, 20, 3))

    def test_reconstruct_patches_flat_not_divisible(self):
        # Last dim 80 is not divisible by prod(size)=25.
        patches = np.zeros((1, 4, 4, 80), dtype="float32")
        with self.assertRaisesRegex(ValueError, "not divisible"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 20),
                padding="valid",
            )
        patches_3d = np.zeros((1, 4, 4, 4, 400), dtype="float32")
        with self.assertRaisesRegex(ValueError, "not divisible"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 20, 20),
                padding="valid",
            )

    def test_reconstruct_patches_symbolic_validation(self):
        # The symbolic path mirrors the eager checks when dims are static,
        # so invalid graphs fail at build time instead of at run time.
        patches = KerasTensor([None, 4, 4, 80])  # 80 not divisible by 25
        with self.assertRaisesRegex(ValueError, "not divisible"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 20),
                padding="valid",
            )
        patches = KerasTensor([None, 4, 4, 75])
        with self.assertRaisesRegex(ValueError, "requires output_size"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(19, 20),
                padding="valid",
            )
        with self.assertRaisesRegex(ValueError, "padding='same'.*height"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(30, 20),
                padding="same",
            )
        patches_3d = KerasTensor([None, 4, 4, 4, 375])
        with self.assertRaisesRegex(ValueError, "requires output_size"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 19, 20),
                padding="valid",
            )
        with self.assertRaisesRegex(ValueError, "padding='same'.*depth"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(10, 20, 20),
                padding="same",
            )

    def test_reconstruct_patches_channels_first(self):
        # 2D channels_first batched: (B, pH*pW*C, gH, gW) -> (B, C, H, W)
        patches = KerasTensor([2, 75, 4, 4])
        out = kimage.reconstruct_patches(
            patches,
            size=(5, 5),
            output_size=(20, 20),
            padding="valid",
            data_format="channels_first",
        )
        self.assertEqual(out.shape, (2, 3, 20, 20))
        # 3D channels_first batched: (B, pD*pH*pW*C, gD, gH, gW)
        patches_3d = KerasTensor([2, 375, 4, 4, 4])
        out = kimage.reconstruct_patches(
            patches_3d,
            size=(5, 5, 5),
            output_size=(20, 20, 20),
            padding="valid",
            data_format="channels_first",
        )
        self.assertEqual(out.shape, (2, 3, 20, 20, 20))
        # The symbolic validation is data_format aware: the grid dims sit
        # at the end in channels_first, and a mismatch still raises.
        with self.assertRaisesRegex(ValueError, "requires output_size"):
            kimage.reconstruct_patches(
                KerasTensor([2, 75, 4, 4]),
                size=(5, 5),
                output_size=(19, 20),
                padding="valid",
                data_format="channels_first",
            )

    def test_reconstruct_patches_channels_first_bad_rank(self):
        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            kimage.reconstruct_patches(
                np.zeros((75, 4), dtype="float32"),
                size=(5, 5),
                output_size=(20, 20),
                padding="valid",
                data_format="channels_first",
            )
        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            kimage.reconstruct_patches(
                np.zeros((375, 4, 4), dtype="float32"),
                size=(5, 5, 5),
                output_size=(20, 20, 20),
                padding="valid",
                data_format="channels_first",
            )

    def test_reconstruct_patches_autoinfer_valid(self):
        # output_size omitted -> inferred from the (static) patch grid.
        patches = KerasTensor([2, 4, 4, 75])
        out = kimage.reconstruct_patches(patches, size=(5, 5), padding="valid")
        self.assertEqual(out.shape, (2, 20, 20, 3))
        patches_3d = KerasTensor([2, 4, 4, 4, 375])
        out = kimage.reconstruct_patches(
            patches_3d, size=(5, 5, 5), padding="valid"
        )
        self.assertEqual(out.shape, (2, 20, 20, 20, 3))
        # channels_first: grid dims are the trailing axes.
        patches_cf = KerasTensor([2, 75, 4, 4])
        out = kimage.reconstruct_patches(
            patches_cf,
            size=(5, 5),
            padding="valid",
            data_format="channels_first",
        )
        self.assertEqual(out.shape, (2, 3, 20, 20))

    def test_reconstruct_patches_autoinfer_dynamic_grid_symbolic(self):
        # Dynamic grid + auto-infer -> spatial dims stay unknown (no raise).
        patches = KerasTensor([2, None, None, 75])
        out = kimage.reconstruct_patches(patches, size=(5, 5), padding="valid")
        self.assertEqual(out.shape, (2, None, None, 3))
        # Mixed static/dynamic grid -> static dims are still resolved.
        patches = KerasTensor([2, None, 4, 75])
        out = kimage.reconstruct_patches(patches, size=(5, 5), padding="valid")
        self.assertEqual(out.shape, (2, None, 20, 3))

    def test_reconstruct_patches_autoinfer_same_raises(self):
        # Auto-infer is valid-only; same requires explicit output_size.
        patches = np.zeros((2, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(ValueError, "only supported for"):
            kimage.reconstruct_patches(patches, size=(5, 5), padding="same")
        patches_3d = np.zeros((2, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(ValueError, "only supported for"):
            kimage.reconstruct_patches(
                patches_3d, size=(5, 5, 5), padding="same"
            )
        # The symbolic path raises at graph-construction time too, so a
        # functional model can't be built that only fails on first call.
        with self.assertRaisesRegex(ValueError, "only supported for"):
            kimage.reconstruct_patches(
                KerasTensor([2, 4, 4, 75]), size=(5, 5), padding="same"
            )

    def test_reconstruct_patches_strides_overlap_not_implemented(self):
        # Overlapping strides (strides != size) are rejected at the op level.
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        with self.assertRaisesRegex(NotImplementedError, "non-overlapping"):
            kimage.reconstruct_patches(
                patches,
                size=(5, 5),
                output_size=(20, 20),
                strides=(2, 2),
            )
        patches_3d = np.zeros((1, 4, 4, 4, 375), dtype="float32")
        with self.assertRaisesRegex(NotImplementedError, "non-overlapping"):
            kimage.reconstruct_patches(
                patches_3d,
                size=(5, 5, 5),
                output_size=(20, 20, 20),
                strides=(2, 2, 2),
            )

    def test_reconstruct_patches_wrong_rank(self):
        # Eager and symbolic wrong-rank inputs get a clear ValueError.
        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            kimage.reconstruct_patches(
                np.zeros((1, 2, 4, 4, 75), dtype="float32"),
                size=(5, 5),
                output_size=(20, 20),
            )
        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            kimage.reconstruct_patches(
                np.zeros((4, 4, 375), dtype="float32"),
                size=(5, 5, 5),
                output_size=(20, 20, 20),
            )
        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            kimage.reconstruct_patches(
                KerasTensor([1, 2, 4, 4, 75]),
                size=(5, 5),
                output_size=(20, 20),
            )

    def test_reconstruct_patches_op_class(self):
        # Direct use of the Operation class (eager `call` + `get_config`).
        op = kimage.ReconstructPatches(size=(5, 5), output_size=(20, 20))
        patches = np.zeros((1, 4, 4, 75), dtype="float32")
        out = op(patches)
        self.assertEqual(tuple(out.shape), (1, 20, 20, 3))
        config = op.get_config()
        self.assertEqual(config["size"], (5, 5))
        self.assertEqual(config["output_size"], (20, 20))
        self.assertEqual(config["strides"], (5, 5))
        self.assertEqual(config["padding"], "valid")

    def test_map_coordinates(self):
        input = KerasTensor([20, 20, 3])
        coordinates = KerasTensor([3, 15, 15, 3])
        out = kimage.map_coordinates(input, coordinates, 0)
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_map_coordinates_uint8(self):
        image_uint8 = tf.ones((1, 1, 3), dtype=tf.uint8)
        coordinates = tf.convert_to_tensor([-1.0, 0.0, 0.0])[..., None, None]

        if backend.backend() != "tensorflow":
            pytest.skip("Skipping test because the backend is not TensorFlow.")

        out = kimage.map_coordinates(
            image_uint8, coordinates, order=1, fill_mode="constant"
        )
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_map_coordinates_float32(self):
        image_float32 = tf.ones((1, 1, 3), dtype=tf.float32)
        coordinates = tf.convert_to_tensor([-1.0, 0.0, 0.0])[..., None, None]

        if backend.backend() != "tensorflow":
            pytest.skip("Skipping test because the backend is not TensorFlow.")

        out = kimage.map_coordinates(
            image_float32, coordinates, order=1, fill_mode="constant"
        )
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_map_coordinates_nearest(self):
        image_uint8 = tf.ones((1, 1, 3), dtype=tf.uint8)
        coordinates = tf.convert_to_tensor([-1.0, 0.0, 0.0])[..., None, None]

        if backend.backend() != "tensorflow":
            pytest.skip("Skipping test because the backend is not TensorFlow.")

        out = kimage.map_coordinates(
            image_uint8, coordinates, order=1, fill_mode="nearest"
        )
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_map_coordinates_manual_cast(self):
        image_uint8 = tf.ones((1, 1, 3), dtype=tf.uint8)
        coordinates = tf.convert_to_tensor([-1.0, 0.0, 0.0])[..., None, None]
        image_uint8_casted = tf.cast(image_uint8, dtype=tf.float32)

        if backend.backend() != "tensorflow":
            pytest.skip("Skipping test because the backend is not TensorFlow.")

        out = tf.cast(
            kimage.map_coordinates(
                image_uint8_casted, coordinates, order=1, fill_mode="constant"
            ),
            dtype=tf.uint8,
        )
        self.assertEqual(out.shape, coordinates.shape[1:])

    def test_pad_images(self):
        # Test channels_last
        x = KerasTensor([15, 25, 3])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (20, 30, 3))

        x_batch = KerasTensor([2, 15, 25, 3])
        out_batch = kimage.pad_images(
            x_batch, 2, 3, target_height=20, target_width=30
        )
        self.assertEqual(out_batch.shape, (2, 20, 30, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 15, 25])
        out = kimage.pad_images(x, 2, 3, target_height=20, target_width=30)
        self.assertEqual(out.shape, (3, 20, 30))

        x_batch = KerasTensor([2, 3, 15, 25])
        out_batch = kimage.pad_images(
            x_batch, 2, 3, target_height=20, target_width=30
        )
        self.assertEqual(out_batch.shape, (2, 3, 20, 30))

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        x_batch = KerasTensor([2, 15, 25, 3])
        out_batch = kimage.crop_images(
            x_batch, 2, 3, target_height=10, target_width=20
        )
        self.assertEqual(out_batch.shape, (2, 10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))

        # Test channels_first and batched
        x_batch = KerasTensor([2, 3, 15, 25])
        out_batch = kimage.crop_images(
            x_batch, 2, 3, target_height=10, target_width=20
        )
        self.assertEqual(out_batch.shape, (2, 3, 10, 20))

    def test_perspective_transform(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        start_points = KerasTensor([4, 2])
        end_points = KerasTensor([4, 2])
        out = kimage.perspective_transform(x, start_points, end_points)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        start_points = KerasTensor([4, 2])
        end_points = KerasTensor([4, 2])
        out = kimage.perspective_transform(x, start_points, end_points)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_gaussian_blur(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        kernel_size = KerasTensor(
            [
                2,
            ]
        )
        sigma = KerasTensor(
            [
                2,
            ]
        )
        out = kimage.gaussian_blur(x, kernel_size, sigma)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        kernel_size = KerasTensor(
            [
                2,
            ]
        )
        sigma = KerasTensor(
            [
                2,
            ]
        )
        out = kimage.gaussian_blur(x, kernel_size, sigma)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_elastic_transform(self):
        # Test channels_last
        x = KerasTensor([20, 20, 3])
        out = kimage.elastic_transform(x)
        self.assertEqual(out.shape, (20, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([3, 20, 20])
        out = kimage.elastic_transform(x)
        self.assertEqual(out.shape, (3, 20, 20))

    def test_scale_and_translate(self):
        images = KerasTensor([20, 20, 3])
        output_shape = (25, 25, 3)
        scale = KerasTensor([2])
        translation = KerasTensor([2])
        out = kimage.scale_and_translate(
            images,
            output_shape=output_shape,
            scale=scale,
            translation=translation,
            spatial_dims=(0, 1),
            method="linear",
        )
        self.assertEqual(out.shape, output_shape)

    def test_ssim(self):
        # Test unbatched channels_last
        x1 = KerasTensor([32, 32, 3])
        x2 = KerasTensor([32, 32, 3])
        out = kimage.ssim(x1, x2, max_val=1.0)
        self.assertEqual(out.shape, ())

        # Test batched channels_last
        x1 = KerasTensor([None, 32, 32, 3])
        x2 = KerasTensor([None, 32, 32, 3])
        out = kimage.ssim(x1, x2, max_val=1.0)
        self.assertEqual(out.shape, (None,))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x1 = KerasTensor([None, 3, 32, 32])
        x2 = KerasTensor([None, 3, 32, 32])
        out = kimage.ssim(x1, x2, max_val=1.0)
        self.assertEqual(out.shape, (None,))


AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}


def _compute_affine_transform_coordinates(image, transform):
    image = image.copy()
    transform = transform.copy()
    need_squeeze = False
    if len(image.shape) == 3:  # unbatched
        need_squeeze = True
        image = np.expand_dims(image, axis=0)
        transform = np.expand_dims(transform, axis=0)
    batch_size = image.shape[0]
    # get indices
    meshgrid = np.meshgrid(
        *[np.arange(size) for size in image.shape[1:]], indexing="ij"
    )
    indices = np.concatenate(
        [np.expand_dims(x, axis=-1) for x in meshgrid], axis=-1
    )
    indices = np.tile(indices, (batch_size, 1, 1, 1, 1))
    # swap the values
    transform[:, 4], transform[:, 0] = (
        transform[:, 0].copy(),
        transform[:, 4].copy(),
    )
    transform[:, 5], transform[:, 2] = (
        transform[:, 2].copy(),
        transform[:, 5].copy(),
    )
    # deal with transform
    transform = np.pad(transform, pad_width=[[0, 0], [0, 1]], constant_values=1)
    transform = np.reshape(transform, (batch_size, 3, 3))
    offset = np.pad(transform[:, 0:2, 2], pad_width=[[0, 0], [0, 1]])
    transform[:, 0:2, 2] = 0
    # transform the indices
    coordinates = np.einsum("Bhwij, Bjk -> Bhwik", indices, transform)
    coordinates = np.moveaxis(coordinates, source=-1, destination=1)
    coordinates += np.reshape(offset, (*offset.shape, 1, 1, 1))
    if need_squeeze:
        coordinates = np.squeeze(coordinates, axis=0)
    return coordinates


def _fixed_map_coordinates(
    input, coordinates, order, fill_mode="constant", fill_value=0.0
):
    # SciPy's implementation of map_coordinates handles boundaries incorrectly,
    # unless mode='reflect'. For order=1, this only affects interpolation
    # outside the bounds of the original array.
    # https://github.com/scipy/scipy/issues/2640
    padding = [
        (
            max(-np.floor(c.min()).astype(int) + 1, 0),
            max(np.ceil(c.max()).astype(int) + 1 - size, 0),
        )
        for c, size in zip(coordinates, input.shape)
    ]
    shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
    pad_mode = {
        "nearest": "edge",
        "mirror": "reflect",
        "reflect": "symmetric",
    }.get(fill_mode, fill_mode)
    if fill_mode == "constant":
        padded = np.pad(
            input, padding, mode=pad_mode, constant_values=fill_value
        )
    else:
        padded = np.pad(input, padding, mode=pad_mode)
    result = scipy.ndimage.map_coordinates(
        padded, shifted_coords, order=order, mode=fill_mode, cval=fill_value
    )
    return result


def _perspective_transform_numpy(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = np.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = np.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, channels = images.shape

    transforms = _compute_homography_matrix(start_points, end_points)

    if len(transforms.shape) == 1:
        transforms = np.expand_dims(transforms, axis=0)
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = np.tile(transforms, (batch_size, 1))

    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )

    output = np.empty((batch_size, height, width, channels))

    for i in range(batch_size):
        a0, a1, a2, a3, a4, a5, a6, a7 = transforms[i]
        denom = a6 * x + a7 * y + 1.0
        x_in = (a0 * x + a1 * y + a2) / denom
        y_in = (a3 * x + a4 * y + a5) / denom

        coords = np.stack([y_in.ravel(), x_in.ravel()], axis=0)

        mapped_channels = []
        for channel in range(channels):
            channel_img = images[i, :, :, channel]

            mapped_channel = _fixed_map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mapped_channel.reshape(height, width))

        output[i] = np.stack(mapped_channels, axis=-1)

    if data_format == "channels_first":
        output = np.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = np.squeeze(output, axis=0)

    return output


def gaussian_blur_np(
    images,
    kernel_size,
    sigma,
    data_format=None,
):
    def _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = np.arange(size, dtype=dtype) - (size - 1) / 2
            kernel1d = np.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / np.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            kernel1d_height = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_width = _get_gaussian_kernel1d(size[1], sigma[1])
            return np.outer(kernel1d_height, kernel1d_width)

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel[:, :, np.newaxis]
        kernel = np.tile(kernel, (1, 1, num_channels))
        return kernel.astype(dtype)

    images = np.asarray(images)
    input_dtype = images.dtype
    kernel_size = np.asarray(kernel_size)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_first":
        images = np.transpose(images, (0, 2, 3, 1))

    num_channels = images.shape[-1]
    kernel = _create_gaussian_kernel(
        kernel_size, sigma, num_channels, input_dtype
    )
    batch_size, height, width, _ = images.shape

    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    pad_h = (kernel_h - 1) // 2
    pad_h_after = kernel_h - 1 - pad_h
    pad_w = (kernel_w - 1) // 2
    pad_w_after = kernel_w - 1 - pad_w

    padded_images = np.pad(
        images,
        (
            (0, 0),
            (pad_h, pad_h_after),
            (pad_w, pad_w_after),
            (0, 0),
        ),
        mode="constant",
    )

    blurred_images = np.zeros_like(images)
    kernel_reshaped = kernel.reshape((1, kernel_h, kernel_w, num_channels))

    for b in range(batch_size):
        image_patch = padded_images[b : b + 1, :, :, :]

    for i in range(height):
        for j in range(width):
            patch = image_patch[:, i : i + kernel_h, j : j + kernel_w, :]
            blurred_images[b, i, j, :] = np.sum(
                patch * kernel_reshaped, axis=(1, 2)
            )

    if data_format == "channels_first":
        blurred_images = np.transpose(blurred_images, (0, 3, 1, 2))
    if need_squeeze:
        blurred_images = np.squeeze(blurred_images, axis=0)

    return blurred_images


def elastic_transform_np(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)

    images = np.asarray(images)
    input_dtype = images.dtype

    alpha = np.asarray(alpha, dtype=input_dtype)
    sigma = np.asarray(sigma, dtype=input_dtype)

    kernel_size = (int(6 * sigma) | 1, int(6 * sigma) | 1)

    need_squeeze = False
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    rng = np.random.default_rng([seed, 0])
    dx = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0).astype(
            input_dtype
        )
        * sigma
    )
    dy = (
        rng.normal(size=(batch_size, height, width), loc=0.0, scale=1.0).astype(
            input_dtype
        )
        * sigma
    )

    dx = gaussian_blur_np(
        np.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )
    dy = gaussian_blur_np(
        np.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma, sigma),
        data_format=data_format,
    )

    dx = np.squeeze(dx)
    dy = np.squeeze(dy)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x[None, :, :], y[None, :, :]

    distorted_x = x + alpha * dx
    distorted_y = y + alpha * dy

    transformed_images = np.zeros_like(images)

    if data_format == "channels_last":
        for i in range(channels):
            transformed_images[..., i] = np.stack(
                [
                    _fixed_map_coordinates(
                        images[b, ..., i],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                    for b in range(batch_size)
                ]
            )
    else:
        for i in range(channels):
            transformed_images[:, i, :, :] = np.stack(
                [
                    _fixed_map_coordinates(
                        images[b, i, ...],
                        [distorted_y[b], distorted_x[b]],
                        order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                        fill_mode=fill_mode,
                        fill_value=fill_value,
                    )
                    for b in range(batch_size)
                ]
            )

    if need_squeeze:
        transformed_images = np.squeeze(transformed_images, axis=0)
    transformed_images = transformed_images.astype(input_dtype)

    return transformed_images


def _compute_homography_matrix(start_points, end_points):
    start_x1, start_y1 = start_points[:, 0, 0], start_points[:, 0, 1]
    start_x2, start_y2 = start_points[:, 1, 0], start_points[:, 1, 1]
    start_x3, start_y3 = start_points[:, 2, 0], start_points[:, 2, 1]
    start_x4, start_y4 = start_points[:, 3, 0], start_points[:, 3, 1]

    end_x1, end_y1 = end_points[:, 0, 0], end_points[:, 0, 1]
    end_x2, end_y2 = end_points[:, 1, 0], end_points[:, 1, 1]
    end_x3, end_y3 = end_points[:, 2, 0], end_points[:, 2, 1]
    end_x4, end_y4 = end_points[:, 3, 0], end_points[:, 3, 1]

    coefficient_matrix = np.stack(
        [
            np.stack(
                [
                    end_x1,
                    end_y1,
                    np.ones_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    -start_x1 * end_x1,
                    -start_x1 * end_y1,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    np.zeros_like(end_x1),
                    end_x1,
                    end_y1,
                    np.ones_like(end_x1),
                    -start_y1 * end_x1,
                    -start_y1 * end_y1,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x2,
                    end_y2,
                    np.ones_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    -start_x2 * end_x2,
                    -start_x2 * end_y2,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    np.zeros_like(end_x2),
                    end_x2,
                    end_y2,
                    np.ones_like(end_x2),
                    -start_y2 * end_x2,
                    -start_y2 * end_y2,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x3,
                    end_y3,
                    np.ones_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    -start_x3 * end_x3,
                    -start_x3 * end_y3,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    np.zeros_like(end_x3),
                    end_x3,
                    end_y3,
                    np.ones_like(end_x3),
                    -start_y3 * end_x3,
                    -start_y3 * end_y3,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    end_x4,
                    end_y4,
                    np.ones_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    -start_x4 * end_x4,
                    -start_x4 * end_y4,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    np.zeros_like(end_x4),
                    end_x4,
                    end_y4,
                    np.ones_like(end_x4),
                    -start_y4 * end_x4,
                    -start_y4 * end_y4,
                ],
                axis=-1,
            ),
        ],
        axis=1,
    )

    target_vector = np.stack(
        [
            start_x1,
            start_y1,
            start_x2,
            start_y2,
            start_x3,
            start_y3,
            start_x4,
            start_y4,
        ],
        axis=-1,
    )
    target_vector = np.expand_dims(target_vector, axis=-1)

    homography_matrix = np.linalg.solve(coefficient_matrix, target_vector)
    homography_matrix = np.reshape(homography_matrix, [-1, 8])

    return homography_matrix


class ImageOpsCorrectnessTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    def test_rgb_to_grayscale(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 50, 50, 3)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 3, 50, 50)).astype("float32") * 255
        out = kimage.rgb_to_grayscale(x)
        ref_out = tf.image.rgb_to_grayscale(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test class
        out = kimage.RGBToGrayscale()(x)
        self.assertAllClose(out, ref_out.numpy())

    def test_rgb_to_hsv(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 50, 50, 3)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 3, 50, 50)).astype("float32")
        out = kimage.rgb_to_hsv(x)
        ref_out = tf.image.rgb_to_hsv(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test class
        out = kimage.RGBToHSV()(x)
        self.assertAllClose(out, ref_out.numpy())

    def test_hsv_to_rgb(self):
        # Test channels_last
        x = np.random.random((50, 50, 3)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 50, 50, 3)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(x)
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 50, 50)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(np.transpose(x, [1, 2, 0]))
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        x = np.random.random((2, 3, 50, 50)).astype("float32")
        out = kimage.hsv_to_rgb(x)
        ref_out = tf.image.hsv_to_rgb(np.transpose(x, [0, 2, 3, 1]))
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out.numpy())

        # Test class
        out = kimage.HSVToRGB()(x)
        self.assertAllClose(out, ref_out.numpy())

    def _skip_resize_if_unsupported(self, interpolation, antialias):
        if backend.backend() == "torch":
            if interpolation == "bicubic" and antialias is False:
                self.skipTest(
                    "Resizing with Bicubic interpolation in "
                    "PyTorch backend produces noise. Please "
                    "turn on anti-aliasing. "
                    f"Received: interpolation={interpolation}, "
                    f"antialias={antialias}."
                )

    @parameterized.named_parameters(
        named_product(
            interpolation=[
                "bilinear",
                "nearest",
                "lanczos3",
                "lanczos5",
                "bicubic",
            ],
            antialias=[True, False],
            crop_pad_mode=["none", "crop", "pad"],
        )
    )
    def test_resize(self, interpolation, antialias, crop_pad_mode):
        self._skip_resize_if_unsupported(interpolation, antialias)

        crop_to_aspect_ratio = crop_pad_mode == "crop"
        pad_to_aspect_ratio = crop_pad_mode == "pad"
        kwargs = {
            "size": (15, 15),
            "interpolation": interpolation,
            "antialias": antialias,
            "crop_to_aspect_ratio": crop_to_aspect_ratio,
            "pad_to_aspect_ratio": pad_to_aspect_ratio,
        }

        def get_ref_x(x):
            # Assumes channels_last
            if crop_pad_mode == "crop":
                if x.ndim == 3:
                    return x[15:45, :, :]
                return x[:, 15:45, :, :]
            elif crop_pad_mode == "pad":
                pad_width = ((0, 0), (15, 15), (0, 0))
                if x.ndim == 4:
                    pad_width = ((0, 0),) + pad_width
                return np.pad(x, pad_width, mode="constant")
            return x

        # Test channels_last
        x = np.random.random((60, 30, 3)).astype("float32") * 255
        out = kimage.resize(x, **kwargs)
        ref_x = get_ref_x(x)
        ref_out = tf.image.resize(
            ref_x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-4)

        x = np.random.random((2, 60, 30, 3)).astype("float32") * 255
        out = kimage.resize(x, **kwargs)
        ref_x = get_ref_x(x)
        ref_out = tf.image.resize(
            ref_x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-4)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 60, 30)).astype("float32") * 255
        out = kimage.resize(x, **kwargs)
        x_cl = np.transpose(x, [1, 2, 0])
        ref_x = get_ref_x(x_cl)
        ref_out = tf.image.resize(
            ref_x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        ref_out = tf.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-4)

        x = np.random.random((2, 3, 60, 30)).astype("float32") * 255
        out = kimage.resize(x, **kwargs)
        x_cl = np.transpose(x, [0, 2, 3, 1])
        ref_x = get_ref_x(x_cl)
        ref_out = tf.image.resize(
            ref_x,
            size=(15, 15),
            method=interpolation,
            antialias=antialias,
        )
        ref_out = tf.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-4)

        # Test class
        out = kimage.Resize(**kwargs)(x)
        self.assertAllClose(out, ref_out, atol=1e-4)

    @parameterized.named_parameters(
        named_product(
            interpolation=[
                "bilinear",
                "nearest",
                "lanczos3",
                "lanczos5",
                "bicubic",
            ],
            antialias=[True, False],
            dtype=["float64", "uint8", "int8", "int16"],
        )
    )
    def test_resize_dtype(self, interpolation, antialias, dtype):
        self._skip_resize_if_unsupported(interpolation, antialias)

        if dtype == "float64":
            x = np.random.random((2, 30, 30, 3)).astype("float64") * 255
            out = kimage.resize(
                x,
                size=(15, 15),
                interpolation=interpolation,
                antialias=antialias,
            )
            expected = "float64"
            if backend.backend() == "jax" and not jax.config.jax_enable_x64:
                expected = "float32"
            self.assertEqual(standardize_dtype(out.dtype), expected)

            ref_out = tf.image.resize(
                x,
                size=(15, 15),
                method=interpolation,
                antialias=antialias,
            )
            self.assertAllClose(out, ref_out, atol=1e-4)
        else:
            dtype_info = np.iinfo(dtype)
            x = np.full((1, 30, 30, 3), dtype_info.min, dtype=dtype)
            x[:, 15:, 15:, :] = dtype_info.max
            out = kimage.resize(
                x,
                size=(45, 45),
                interpolation=interpolation,
                antialias=antialias,
            )
            self.assertEqual(standardize_dtype(out.dtype), dtype)

            ref_x = x.astype("float32")
            ref_out = tf.image.resize(
                ref_x,
                size=(45, 45),
                method=interpolation,
                antialias=antialias,
            )
            ref_out = np.clip(
                np.round(ref_out), dtype_info.min, dtype_info.max
            ).astype(dtype)
            self.assertAllClose(out, ref_out, atol=2)

    def test_resize_uint8_round(self):
        x = np.array([0, 1, 254, 255], dtype="uint8").reshape(1, 2, 2, 1)
        expected = np.array(
            # OpenCV as gold standard.
            # [
            #     [0, 0, 1, 1],
            #     [64, 64, 64, 65],
            #     [191, 191, 191, 192],
            #     [254, 254, 255, 255],
            # ]
            #
            # Resize without `round` - differences in 8 points
            # [
            #     [0, 0, 0, 1],
            #     [63, 63, 64, 64],
            #     [190, 190, 191, 191],
            #     [254, 254, 254, 255],
            # ]
            #
            # Resize with `round` - differences in 2 points
            [
                [0, 0, 1, 1],
                [64, 64, 64, 64],
                [190, 191, 191, 192],
                [254, 254, 255, 255],
            ],
            dtype="uint8",
        ).reshape(1, 4, 4, 1)
        out = kimage.resize(
            x,
            size=(4, 4),
            interpolation="bilinear",
            antialias=False,
        )
        self.assertEqual(tuple(out.shape), tuple(expected.shape))
        self.assertEqual(standardize_dtype(out.dtype), "uint8")
        self.assertAllClose(out, expected, atol=1e-4)

    def test_resize_uint8_round_saturate(self):
        x = np.array([0, 1, 254, 255], dtype="uint8").reshape(1, 2, 2, 1)
        expected = np.array(
            # OpenCV as gold standard. Same for `torch` backend.
            (
                [
                    [0, 0, 0, 0],
                    [57, 58, 58, 59],
                    [196, 197, 197, 198],
                    [255, 255, 255, 255],
                ]
                if "torch" == backend.backend()
                # Resize without `round` and `saturate_cast` - differences in
                # 16 points
                # [
                #     [234, 234, 235, 235],
                #     [-5, -6, -5, -6],
                #     [5, 4, 5, 4],
                #     [-235, -235, -234, -234],
                # ]
                #
                # Resize with `round` and `saturate_cast` - differences in
                # 8 points
                else [
                    [0, 0, 0, 0],
                    [53, 53, 53, 54],
                    [201, 202, 202, 202],
                    [255, 255, 255, 255],
                ]
            ),
            dtype="uint8",
        ).reshape(1, 4, 4, 1)
        out = kimage.resize(
            x,
            size=(4, 4),
            interpolation="bicubic",
            antialias=False,
        )
        self.assertEqual(tuple(out.shape), tuple(expected.shape))
        self.assertEqual(standardize_dtype(out.dtype), "uint8")
        self.assertAllClose(out, expected, atol=1e-4)

    def test_resize_with_crop(self):
        # Test channels_last
        x = np.random.random((60, 50, 3)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (25, 25, 3))

        x = np.random.random((2, 50, 60, 3)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (2, 25, 25, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 60, 50)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (3, 25, 25))

        x = np.random.random((2, 3, 50, 60)).astype("float32") * 255
        out = kimage.resize(x, size=(25, 25), crop_to_aspect_ratio=True)
        self.assertEqual(out.shape, (2, 3, 25, 25))

    @parameterized.named_parameters(named_product(fill_value=[1.0, 2.0]))
    def test_resize_with_pad(self, fill_value):
        # Test channels_last
        x = np.random.random((60, 50, 3)).astype("float32") * 255
        out = kimage.resize(
            x,
            size=(25, 25),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        self.assertEqual(out.shape, (25, 25, 3))

        x = np.random.random((2, 50, 60, 3)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 25, 25, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.random((3, 60, 50)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (3, 25, 25))

        x = np.random.random((2, 3, 50, 60)).astype("float32") * 255
        out = kimage.resize(
            x, size=(25, 25), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 3, 25, 25))

        x = np.ones((2, 3, 10, 10)) * 128
        out = kimage.resize(
            x, size=(4, 4), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 3, 4, 4))
        self.assertAllClose(out[:, 0, :, :], np.ones((2, 4, 4)) * 128)

        x = np.ones((2, 3, 10, 8)) * 128
        out = kimage.resize(
            x, size=(4, 4), pad_to_aspect_ratio=True, fill_value=fill_value
        )
        self.assertEqual(out.shape, (2, 3, 4, 4))
        self.assertAllClose(
            out,
            np.concatenate(
                [
                    np.ones((2, 3, 4, 1)) * 96.25,
                    np.ones((2, 3, 4, 2)) * 128.0,
                    np.ones((2, 3, 4, 1)) * 96.25,
                ],
                axis=3,
            ),
            atol=1.0,
        )

    def test_resize_pad_to_aspect_ratio_fill_value_correctness(self):
        fill_value = 0.5
        x = np.zeros((1, 10, 10, 3), dtype="float32")
        out = kimage.resize(
            x,
            size=(10, 20),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        out_np = backend.ops.convert_to_numpy(out)
        self.assertAllClose(
            out_np[0, :, :5, :], np.ones((10, 5, 3)) * fill_value
        )
        self.assertAllClose(
            out_np[0, :, 15:, :], np.ones((10, 5, 3)) * fill_value
        )

        x = np.zeros((1, 10, 10, 3), dtype="float32")
        out = kimage.resize(
            x,
            size=(20, 10),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        out_np = backend.ops.convert_to_numpy(out)
        self.assertAllClose(
            out_np[0, :5, :, :], np.ones((5, 10, 3)) * fill_value
        )
        self.assertAllClose(
            out_np[0, 15:, :, :], np.ones((5, 10, 3)) * fill_value
        )

    @parameterized.named_parameters(
        named_product(
            dtype=["uint8", "int32"],
            fill_value=[0.0, 10.0],
        )
    )
    def test_resize_pad_to_aspect_ratio_integer_dtype(self, dtype, fill_value):
        x = np.random.randint(0, 255, size=(2, 60, 30, 3), dtype=dtype)
        out = kimage.resize(
            x,
            size=(15, 15),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        self.assertEqual(out.shape, (2, 15, 15, 3))
        self.assertEqual(standardize_dtype(out.dtype), dtype)

        # 3D unbatched input
        x_3d = np.random.randint(0, 255, size=(60, 30, 3), dtype=dtype)
        out_3d = kimage.resize(
            x_3d,
            size=(15, 15),
            pad_to_aspect_ratio=True,
            fill_value=fill_value,
        )
        self.assertEqual(out_3d.shape, (15, 15, 3))
        self.assertEqual(standardize_dtype(out_3d.dtype), dtype)

    @parameterized.named_parameters(
        ("zero_height", (0, 10)),
        ("zero_width", (10, 0)),
        ("zero_both", (0, 0)),
        ("negative_height", (-1, 10)),
        ("negative_width", (10, -1)),
    )
    def test_resize_invalid_size_zero_or_negative(self, invalid_size):
        """Resize rejects zero or negative height/width."""
        x = np.random.random((10, 10, 3)).astype("float32")
        with self.assertRaisesRegex(
            ValueError,
            "`size` must have positive height and width",
        ):
            kimage.resize(x, size=invalid_size)

    @parameterized.named_parameters(
        named_product(
            interpolation=["bilinear", "nearest"],
            fill_mode=["constant", "nearest", "wrap", "mirror", "reflect"],
        )
    )
    def test_affine_transform(self, interpolation, fill_mode):
        if backend.backend() == "tensorflow" and fill_mode == "mirror":
            self.skipTest(
                "In tensorflow backend, applying affine_transform with "
                "fill_mode=mirror is not supported"
            )
        if backend.backend() == "tensorflow" and fill_mode == "wrap":
            self.skipTest(
                "In tensorflow backend, the numerical results of applying "
                "affine_transform with fill_mode=wrap is inconsistent with"
                "scipy"
            )
        if (
            testing.jax_uses_tpu()
            and interpolation == "bilinear"
            and fill_mode == "constant"
        ):
            self.skipTest(
                "JAX on TPU interpolation='bilinear' and fill_mode='constant' "
                "Produces one incorrect pixel in the corner"
            )
        # TODO: `nearest` interpolation in jax and torch causes random index
        # shifting, resulting in significant differences in output which leads
        # to failure
        if backend.backend() in ("jax", "torch") and interpolation == "nearest":
            self.skipTest(
                f"In {backend.backend()} backend, "
                f"interpolation={interpolation} causes index shifting and "
                "leads test failure"
            )

        # Test channels_last
        np.random.seed(42)
        x = np.random.uniform(size=(50, 50, 3)).astype("float32") * 255
        transform = np.random.uniform(size=(6)).astype("float32")
        transform = np.pad(transform, (0, 2))  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x, transform, interpolation=interpolation, fill_mode=fill_mode
        )
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = _fixed_map_coordinates(
            x,
            coordinates,
            order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
            fill_mode=fill_mode,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, tpu_atol=10, tpu_rtol=10)

        x = np.random.uniform(size=(2, 50, 50, 3)).astype("float32") * 255
        transform = np.random.uniform(size=(2, 6)).astype("float32")
        transform = np.pad(transform, [(0, 0), (0, 2)])  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        coordinates = _compute_affine_transform_coordinates(x, transform)
        ref_out = np.stack(
            [
                _fixed_map_coordinates(
                    x[i],
                    coordinates[i],
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                )
                for i in range(x.shape[0])
            ],
            axis=0,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, tpu_atol=10, tpu_rtol=10)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 50, 50)).astype("float32") * 255
        transform = np.random.uniform(size=(6)).astype("float32")
        transform = np.pad(transform, (0, 2))  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x, transform, interpolation=interpolation, fill_mode=fill_mode
        )
        coordinates = _compute_affine_transform_coordinates(
            np.transpose(x, [1, 2, 0]), transform
        )
        ref_out = _fixed_map_coordinates(
            np.transpose(x, [1, 2, 0]),
            coordinates,
            order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
            fill_mode=fill_mode,
        )
        ref_out = np.transpose(ref_out, [2, 0, 1])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, tpu_atol=1, tpu_rtol=1)

        x = np.random.uniform(size=(2, 3, 50, 50)).astype("float32") * 255
        transform = np.random.uniform(size=(2, 6)).astype("float32")
        transform = np.pad(transform, [(0, 0), (0, 2)])  # makes c0, c1 always 0
        out = kimage.affine_transform(
            x,
            transform,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        coordinates = _compute_affine_transform_coordinates(
            np.transpose(x, [0, 2, 3, 1]), transform
        )
        ref_out = np.stack(
            [
                _fixed_map_coordinates(
                    np.transpose(x[i], [1, 2, 0]),
                    coordinates[i],
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                )
                for i in range(x.shape[0])
            ],
            axis=0,
        )
        ref_out = np.transpose(ref_out, [0, 3, 1, 2])
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, tpu_atol=10, tpu_rtol=10)

        # Test class
        out = kimage.AffineTransform(
            interpolation=interpolation, fill_mode=fill_mode
        )(x, transform)
        self.assertAllClose(out, ref_out, atol=1e-2, tpu_atol=10, tpu_rtol=10)

    @parameterized.named_parameters(
        named_product(
            size=[(3, 3), (5, 5)],
            strides=[None, (1, 1), (2, 2)],
            dilation_rate=[1, 3],
            padding=["valid", "same"],
        )
    )
    def test_extract_patches(self, size, strides, dilation_rate, padding):
        patch_h, patch_w = size[0], size[1]
        if strides is None:
            strides_h, strides_w = patch_h, patch_w
        else:
            strides_h, strides_w = strides[0], strides[1]

        # Test channels_last
        image = np.random.uniform(size=(1, 20, 20, 3)).astype("float32")
        patches_out = kimage.extract_patches(
            image,
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )
        patches_ref = tf.image.extract_patches(
            image,
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, strides_h, strides_w, 1),
            rates=(1, dilation_rate, dilation_rate, 1),
            padding=padding.upper(),
        )
        self.assertEqual(tuple(patches_out.shape), tuple(patches_ref.shape))
        self.assertAllClose(patches_out, patches_ref, atol=1e-2)

        # Test channels_first
        if backend.backend() == "tensorflow":
            # tensorflow doesn't support channels_first in
            # `kimage.extract_patches`
            return
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 3, 20, 20)).astype("float32")
        patches_out = kimage.extract_patches(
            image,
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )
        patches_ref = tf.image.extract_patches(
            np.transpose(image, [0, 2, 3, 1]),
            sizes=(1, patch_h, patch_w, 1),
            strides=(1, strides_h, strides_w, 1),
            rates=(1, dilation_rate, dilation_rate, 1),
            padding=padding.upper(),
        )
        patches_ref = tf.transpose(patches_ref, [0, 3, 1, 2])
        self.assertEqual(tuple(patches_out.shape), tuple(patches_ref.shape))
        self.assertAllClose(patches_out, patches_ref, atol=1e-2)

        # Test class
        patches_out = kimage.ExtractPatches(
            size=size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
        )(image)
        self.assertAllClose(patches_out, patches_ref, atol=1e-2)

    @parameterized.named_parameters(
        named_product(
            # (input_shape, coordinates_shape)
            shape=[((5,), (7,)), ((3, 4, 5), (2, 3, 4))],
            # TODO: scipy.ndimage.map_coordinates does not support float16
            # TODO: torch cpu does not support round & floor for float16
            dtype=["uint8", "int32", "float32"],
            order=[0, 1],
            fill_mode=["constant", "nearest", "wrap", "mirror", "reflect"],
        )
    )
    def test_map_coordinates(self, shape, dtype, order, fill_mode):
        input_shape, coordinates_shape = shape
        input = np.arange(math.prod(input_shape), dtype=dtype).reshape(
            input_shape
        )
        coordinates_dtype = "float32" if "int" in dtype else dtype
        coordinates = [
            (size - 1)
            * np.random.uniform(size=coordinates_shape).astype(
                coordinates_dtype
            )
            for size in input_shape
        ]
        output = kimage.map_coordinates(input, coordinates, order, fill_mode)
        expected = _fixed_map_coordinates(input, coordinates, order, fill_mode)
        self.assertAllClose(output, expected)

        # Test class
        output = kimage.MapCoordinates(order, fill_mode)(input, coordinates)
        self.assertAllClose(output, expected)

    @parameterized.parameters(
        [
            (0, 0, 3, 3, None, None),
            (1, 0, 4, 3, None, None),
            (0, 1, 3, 4, None, None),
            (0, 0, 4, 3, None, None),
            (0, 0, 3, 4, None, None),
            (0, 0, None, None, 0, 1),
            (0, 0, None, None, 1, 0),
            (1, 2, None, None, 3, 4),
        ]
    )
    def test_pad_images(
        self,
        top_padding,
        left_padding,
        target_height,
        target_width,
        bottom_padding,
        right_padding,
    ):
        # Test channels_last
        image = np.random.uniform(size=(3, 3, 1)).astype("float32")
        _target_height = target_height  # For `tf.image.pad_to_bounding_box`
        _target_width = target_width  # For `tf.image.pad_to_bounding_box`
        if _target_height is None:
            _target_height = image.shape[0] + top_padding + bottom_padding
        if _target_width is None:
            _target_width = image.shape[1] + left_padding + right_padding
        padded_image = kimage.pad_images(
            image,
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )
        ref_padded_image = tf.image.pad_to_bounding_box(
            image, top_padding, left_padding, _target_height, _target_width
        )
        self.assertEqual(
            tuple(padded_image.shape), tuple(ref_padded_image.shape)
        )
        self.assertAllClose(padded_image, ref_padded_image)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 3, 3)).astype("float32")
        padded_image = kimage.pad_images(
            image,
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )
        ref_padded_image = tf.image.pad_to_bounding_box(
            np.transpose(image, [1, 2, 0]),
            top_padding,
            left_padding,
            _target_height,
            _target_width,
        )
        ref_padded_image = tf.transpose(ref_padded_image, [2, 0, 1])
        self.assertEqual(
            tuple(padded_image.shape), tuple(ref_padded_image.shape)
        )
        self.assertAllClose(padded_image, ref_padded_image)

        # Test class
        padded_image = kimage.PadImages(
            top_padding,
            left_padding,
            bottom_padding,
            right_padding,
            target_height,
            target_width,
        )(image)
        self.assertAllClose(padded_image, ref_padded_image)

    @parameterized.parameters(
        [
            (0, 0, 3, 3, None, None),
            (1, 0, 4, 3, None, None),
            (0, 1, 3, 4, None, None),
            (0, 0, 4, 3, None, None),
            (0, 0, 3, 4, None, None),
            (0, 0, None, None, 0, 1),
            (0, 0, None, None, 1, 0),
            (1, 2, None, None, 3, 4),
        ]
    )
    def test_crop_images(
        self,
        top_cropping,
        left_cropping,
        target_height,
        target_width,
        bottom_cropping,
        right_cropping,
    ):
        # Test channels_last
        image = np.random.uniform(size=(10, 10, 1)).astype("float32")
        _target_height = target_height  # For `tf.image.pad_to_bounding_box`
        _target_width = target_width  # For `tf.image.pad_to_bounding_box`
        if _target_height is None:
            _target_height = image.shape[0] - top_cropping - bottom_cropping
        if _target_width is None:
            _target_width = image.shape[1] - left_cropping - right_cropping
        cropped_image = kimage.crop_images(
            image,
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )
        ref_cropped_image = tf.image.crop_to_bounding_box(
            image, top_cropping, left_cropping, _target_height, _target_width
        )
        self.assertEqual(
            tuple(cropped_image.shape), tuple(ref_cropped_image.shape)
        )
        self.assertAllClose(cropped_image, ref_cropped_image)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        image = np.random.uniform(size=(1, 10, 10)).astype("float32")
        cropped_image = kimage.crop_images(
            image,
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )
        ref_cropped_image = tf.image.crop_to_bounding_box(
            np.transpose(image, [1, 2, 0]),
            top_cropping,
            left_cropping,
            _target_height,
            _target_width,
        )
        ref_cropped_image = tf.transpose(ref_cropped_image, [2, 0, 1])
        self.assertEqual(
            tuple(cropped_image.shape), tuple(ref_cropped_image.shape)
        )
        self.assertAllClose(cropped_image, ref_cropped_image)

        # Test class
        cropped_image = kimage.CropImages(
            top_cropping,
            left_cropping,
            bottom_cropping,
            right_cropping,
            target_height,
            target_width,
        )(image)
        self.assertAllClose(cropped_image, ref_cropped_image)

    @parameterized.named_parameters(
        named_product(
            interpolation=["bilinear", "nearest"],
        )
    )
    def test_perspective_transform(self, interpolation):
        # Test channels_last
        np.random.seed(42)
        x = np.random.uniform(size=(50, 50, 3)).astype("float32")
        start_points = np.random.uniform(size=(1, 4, 2)).astype("float32")
        end_points = np.random.uniform(size=(1, 4, 2)).astype("float32")

        out = kimage.perspective_transform(
            x, start_points, end_points, interpolation=interpolation
        )

        ref_out = _perspective_transform_numpy(
            x, start_points, end_points, interpolation=interpolation
        )

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 50, 50)).astype("float32")
        start_points = np.random.uniform(size=(1, 4, 2)).astype("float32")
        end_points = np.random.uniform(size=(1, 4, 2)).astype("float32")

        out = kimage.perspective_transform(
            x, start_points, end_points, interpolation=interpolation
        )

        ref_out = _perspective_transform_numpy(
            x,
            start_points,
            end_points,
            interpolation=interpolation,
            data_format="channels_first",
        )

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

    def test_perspective_transform_bfloat16_no_crash(self):
        images = np.ones((10, 10, 3), dtype="bfloat16")
        start_points = np.array(
            [
                [
                    [0.95703125, 0.2080078125],
                    [0.828125, 0.1494140625],
                    [0.51171875, 0.1357421875],
                    [0.6875, 0.83984375],
                ]
            ],
            dtype="bfloat16",
        )
        end_points = np.array(
            [
                [
                    [0.42578125, 0.95703125],
                    [0.82421875, 0.337890625],
                    [0.57421875, 0.75390625],
                    [0.828125, 0.93359375],
                ]
            ],
            dtype="bfloat16",
        )

        out = kimage.perspective_transform(
            images, start_points, end_points, interpolation="bilinear"
        )
        self.assertEqual(tuple(out.shape), tuple(images.shape))
        self.assertEqual(standardize_dtype(out.dtype), "bfloat16")

    def test_gaussian_blur(self):
        # Test channels_last
        backend.set_image_data_format("channels_last")
        np.random.seed(42)
        x = np.random.uniform(size=(50, 50, 3)).astype("float32")
        kernel_size = np.array([3, 3])
        sigma = np.random.uniform(size=(2,)).astype("float32")

        out = kimage.gaussian_blur(
            x,
            kernel_size,
            sigma,
            data_format="channels_last",
        )

        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_last",
        )

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 50, 50)).astype("float32")
        kernel_size = np.array([3, 3])
        sigma = np.random.uniform(size=(2,)).astype("float32")

        out = kimage.gaussian_blur(
            x,
            kernel_size,
            sigma,
            data_format="channels_first",
        )

        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_first",
        )

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

    def test_gaussian_blur_asymmetric_kernel(self):
        image = np.zeros((9, 11, 1), dtype="float32")
        image[4, 5, 0] = 1.0
        kernel_size = (3, 5)
        sigma = (0.5, 2.0)

        def normalized_kernel(size, standard_deviation):
            positions = np.arange(size) - (size - 1) / 2
            values = np.exp(-0.5 * (positions / standard_deviation) ** 2)
            return values / np.sum(values)

        expected = np.zeros_like(image)
        expected[3:6, 3:8, 0] = np.outer(
            normalized_kernel(kernel_size[0], sigma[0]),
            normalized_kernel(kernel_size[1], sigma[1]),
        )

        for data_format in ("channels_last", "channels_first"):
            input_image = (
                image
                if data_format == "channels_last"
                else np.transpose(image, (2, 0, 1))
            )
            expected_image = (
                expected
                if data_format == "channels_last"
                else np.transpose(expected, (2, 0, 1))
            )
            out = kimage.gaussian_blur(
                input_image,
                kernel_size=kernel_size,
                sigma=sigma,
                data_format=data_format,
            )
            self.assertEqual(tuple(out.shape), tuple(input_image.shape))
            self.assertAllClose(
                out,
                expected_image,
                atol=1e-6,
                rtol=1e-6,
                tpu_atol=1e-2,
                tpu_rtol=1e-2,
            )

    def test_gaussian_blur_even_rectangular_kernel(self):
        image = (np.arange(9 * 11, dtype="float32").reshape(9, 11, 1) / 13) ** 2
        kernel_size = (4, 2)
        sigma = (1.0, 0.7)
        out = kimage.gaussian_blur(image, kernel_size, sigma)
        expected = gaussian_blur_np(image, kernel_size, sigma)

        self.assertEqual(tuple(out.shape), tuple(image.shape))
        self.assertAllClose(
            out, expected, atol=5e-5, rtol=1e-5, tpu_atol=1e-2, tpu_rtol=1e-2
        )

    def test_gaussian_blur_even_kernel_size(self):
        """Test gaussian_blur with even kernel sizes"""
        # This test is specific to the numpy backend fix
        if backend.backend() != "numpy":
            self.skipTest(
                "Test is specific to numpy backend, current backend: "
                f"{backend.backend()}"
            )

        backend.set_image_data_format("channels_last")
        np.random.seed(42)
        x = np.random.uniform(size=(32, 32, 3)).astype("float32")
        kernel_size = np.array([4, 4])  # Even kernel size
        sigma = np.array([0.8, 1.2]).astype("float32")

        out = kimage.gaussian_blur(
            x,
            kernel_size,
            sigma,
            data_format="channels_last",
        )

        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_last",
        )

        self.assertEqual(tuple(out.shape), (32, 32, 3))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

        # Test channels_first with different even kernel sizes
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 32, 32)).astype("float32")
        kernel_size = np.array([6, 4])  # Different even kernel sizes
        sigma = np.array([1.0, 1.5]).astype("float32")

        out = kimage.gaussian_blur(
            x,
            kernel_size,
            sigma,
            data_format="channels_first",
        )

        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_first",
        )

        self.assertEqual(tuple(out.shape), (3, 32, 32))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

    def test_gaussian_blur_default_data_format(self):
        # Omitting `data_format` must behave exactly like passing the
        # configured `image_data_format` explicitly.
        np.random.seed(42)
        kernel_size = np.array([3, 3])
        sigma = np.array([1.0, 2.0]).astype("float32")

        # Test channels_last
        backend.set_image_data_format("channels_last")
        x = np.random.uniform(size=(20, 20, 3)).astype("float32")

        out = kimage.gaussian_blur(x, kernel_size, sigma)
        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_last",
        )

        self.assertEqual(tuple(out.shape), (20, 20, 3))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 20, 20)).astype("float32")

        out = kimage.gaussian_blur(x, kernel_size, sigma)
        ref_out = gaussian_blur_np(
            x,
            kernel_size,
            sigma,
            data_format="channels_first",
        )

        self.assertEqual(tuple(out.shape), (3, 20, 20))
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-2, rtol=1e-2)

        backend.set_image_data_format("channels_last")

    def test_elastic_transform(self):
        # Test channels_last
        backend.set_image_data_format("channels_last")
        np.random.seed(42)
        x = np.random.uniform(size=(50, 50, 3)).astype("float32")
        alpha, sigma, seed = 20.0, 5.0, 42

        out = kimage.elastic_transform(
            x,
            alpha=alpha,
            sigma=sigma,
            seed=seed,
            data_format="channels_last",
        )

        ref_out = elastic_transform_np(
            x,
            alpha=alpha,
            sigma=sigma,
            seed=seed,
            data_format="channels_last",
        )

        out = backend.ops.convert_to_numpy(out)

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(
            np.mean(out), np.mean(ref_out), atol=1e-2, rtol=1e-2
        )
        self.assertAllClose(np.var(out), np.var(ref_out), atol=1e-2, rtol=1e-2)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(size=(3, 50, 50)).astype("float32")
        alpha, sigma, seed = 20.0, 5.0, 42

        ref_out = elastic_transform_np(
            x,
            alpha=alpha,
            sigma=sigma,
            seed=seed,
            data_format="channels_first",
        )

        out = kimage.elastic_transform(
            x,
            alpha=alpha,
            sigma=sigma,
            seed=seed,
            data_format="channels_first",
        )
        out = backend.ops.convert_to_numpy(out)

        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(
            np.mean(out), np.mean(ref_out), atol=1e-2, rtol=1e-2
        )
        self.assertAllClose(np.var(out), np.var(ref_out), atol=1e-2, rtol=1e-2)

    def test_map_coordinates_constant_padding(self):
        input_img = tf.ones((2, 2), dtype=tf.uint8)
        # one pixel outside of the input space around the edges
        grid = tf.stack(
            tf.meshgrid(
                tf.range(-1, 3, dtype=tf.float32),
                tf.range(-1, 3, dtype=tf.float32),
                indexing="ij",
            ),
            axis=0,
        )
        out = backend.ops.convert_to_numpy(
            kimage.map_coordinates(
                input_img, grid, order=0, fill_mode="constant", fill_value=0
            )
        )

        # check for ones in the middle and zeros around the edges
        self.assertTrue(np.all(out[:1] == 0))
        self.assertTrue(np.all(out[-1:] == 0))
        self.assertTrue(np.all(out[:, :1] == 0))
        self.assertTrue(np.all(out[:, -1:] == 0))
        self.assertTrue(np.all(out[1:3, 1:3] == 1))

    @parameterized.named_parameters(
        named_product(
            method=["linear", "cubic", "lanczos3", "lanczos5"],
            antialias=[True, False],
        )
    )
    def test_scale_and_translate(self, method, antialias):
        images = np.random.random((30, 30, 3)).astype("float32") * 255
        scale = np.array([2.0, 2.0]).astype("float32")
        translation = -(scale / 2.0 - 0.5)
        out = kimage.scale_and_translate(
            images,
            output_shape=(15, 15, 3),
            scale=scale,
            translation=translation,
            spatial_dims=(0, 1),
            method=method,
            antialias=antialias,
        )
        ref_out = jax.image.scale_and_translate(
            images,
            shape=(15, 15, 3),
            spatial_dims=(0, 1),
            scale=scale,
            translation=translation,
            method=method,
            antialias=antialias,
        )
        self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
        self.assertAllClose(out, ref_out, atol=1e-4)

    def test_reconstruct_patches(self):
        # Eager extract -> reconstruct round-trips exactly, 2D.
        for h, w, c, size, padding in [
            (20, 20, 3, (5, 5), "valid"),
            (32, 48, 1, (4, 8), "valid"),
            (59, 55, 3, (8, 8), "same"),
            (33, 41, 2, (5, 7), "same"),
        ]:
            x = _bf16_exact((2, h, w, c))
            patches = kimage.extract_patches(x, size=size, padding=padding)
            out = kimage.reconstruct_patches(
                patches, size=size, output_size=(h, w), padding=padding
            )
            self.assertEqual(tuple(out.shape), x.shape)
            self.assertAllClose(out, x, atol=1e-6)

        # Unbatched round-trip.
        x = _bf16_exact((16, 16, 3))
        patches = kimage.extract_patches(x, size=(4, 4), padding="valid")
        out = kimage.reconstruct_patches(
            patches, size=(4, 4), output_size=(16, 16)
        )
        self.assertEqual(tuple(out.shape), x.shape)
        self.assertAllClose(out, x, atol=1e-6)

        # Valid padding with non-divisible dims reconstructs the covered
        # (cropped) region: output_size is grid * size, not the original.
        x = _bf16_exact((2, 10, 11, 3))
        patches = kimage.extract_patches(x, size=(4, 4), padding="valid")
        out = kimage.reconstruct_patches(
            patches, size=(4, 4), output_size=(8, 8)
        )
        self.assertAllClose(out, x[:, :8, :8, :], atol=1e-6)

    def test_reconstruct_patches_3d(self):
        # Eager extract -> reconstruct round-trips exactly, 3D.
        for d, h, w, c, size, padding in [
            (20, 20, 20, 1, (5, 5, 5), "valid"),
            (9, 12, 15, 2, (3, 4, 5), "valid"),
            (11, 13, 15, 2, (4, 5, 6), "same"),
        ]:
            x = _bf16_exact((2, d, h, w, c))
            patches = kimage.extract_patches_3d(x, size=size, padding=padding)
            out = kimage.reconstruct_patches(
                patches, size=size, output_size=(d, h, w), padding=padding
            )
            self.assertEqual(tuple(out.shape), x.shape)
            self.assertAllClose(out, x, atol=1e-6)

    def test_reconstruct_patches_channels_first(self):
        # channels_first reconstruction equals the channels_last result
        # after transposing. Patches are extracted in channels_last and
        # transposed into channels_first layout, so the test runs on every
        # backend regardless of NCHW conv support in extract_patches (the
        # suite skips 2D channels_first extract_patches on tensorflow).
        for h, w, c, size, padding in [
            (20, 20, 3, (5, 5), "valid"),
            (33, 41, 2, (5, 7), "same"),
        ]:
            x = _bf16_exact((2, h, w, c))
            patches_cl = kimage.extract_patches(x, size=size, padding=padding)
            patches_cf = knp.transpose(patches_cl, (0, 3, 1, 2))
            recon_cf = kimage.reconstruct_patches(
                patches_cf,
                size=size,
                output_size=(h, w),
                padding=padding,
                data_format="channels_first",
            )
            self.assertEqual(tuple(recon_cf.shape), (2, c, h, w))
            self.assertAllClose(
                recon_cf, np.transpose(x, (0, 3, 1, 2)), atol=1e-6
            )

        # Unbatched 2D exercises the rank-3 transpose branch.
        x = _bf16_exact((8, 8, 2))
        patches_cl = kimage.extract_patches(x, size=(4, 4), padding="valid")
        patches_cf = knp.transpose(patches_cl, (2, 0, 1))
        recon_cf = kimage.reconstruct_patches(
            patches_cf,
            size=(4, 4),
            output_size=(8, 8),
            data_format="channels_first",
        )
        self.assertEqual(tuple(recon_cf.shape), (2, 8, 8))
        self.assertAllClose(recon_cf, np.transpose(x, (2, 0, 1)), atol=1e-6)

    def test_reconstruct_patches_3d_channels_first(self):
        # Batched 3D channels_first round-trip via transposed patches.
        x = _bf16_exact((2, 6, 6, 6, 3))
        patches_cl = kimage.extract_patches_3d(
            x, size=(3, 3, 3), padding="valid"
        )
        patches_cf = knp.transpose(patches_cl, (0, 4, 1, 2, 3))
        recon_cf = kimage.reconstruct_patches(
            patches_cf,
            size=(3, 3, 3),
            output_size=(6, 6, 6),
            padding="valid",
            data_format="channels_first",
        )
        self.assertEqual(tuple(recon_cf.shape), (2, 3, 6, 6, 6))
        self.assertAllClose(
            recon_cf, np.transpose(x, (0, 4, 1, 2, 3)), atol=1e-6
        )

        # Unbatched 3D exercises the rank-4 transpose branch.
        x = _bf16_exact((6, 6, 6, 2))
        patches_cl = kimage.extract_patches_3d(
            x, size=(3, 3, 3), padding="valid"
        )
        patches_cf = knp.transpose(patches_cl, (3, 0, 1, 2))
        recon_cf = kimage.reconstruct_patches(
            patches_cf,
            size=(3, 3, 3),
            output_size=(6, 6, 6),
            data_format="channels_first",
        )
        self.assertEqual(tuple(recon_cf.shape), (2, 6, 6, 6))
        self.assertAllClose(recon_cf, np.transpose(x, (3, 0, 1, 2)), atol=1e-6)

    def test_reconstruct_patches_autoinfer(self):
        # Omitting output_size reproduces the explicit result for valid.
        x = _bf16_exact((2, 20, 20, 3))
        patches = kimage.extract_patches(x, size=(5, 5), padding="valid")
        out = kimage.reconstruct_patches(patches, size=(5, 5))
        self.assertEqual(tuple(out.shape), x.shape)
        self.assertAllClose(out, x, atol=1e-6)

        # channels_first + auto-infer.
        patches_cf = knp.transpose(patches, (0, 3, 1, 2))
        out_cf = kimage.reconstruct_patches(
            patches_cf, size=(5, 5), data_format="channels_first"
        )
        self.assertAllClose(out_cf, np.transpose(x, (0, 3, 1, 2)), atol=1e-6)

        # 3D.
        v = _bf16_exact((2, 9, 9, 9, 1))
        patches_3d = kimage.extract_patches_3d(
            v, size=(3, 3, 3), padding="valid"
        )
        out_3d = kimage.reconstruct_patches(patches_3d, size=(3, 3, 3))
        self.assertEqual(tuple(out_3d.shape), v.shape)
        self.assertAllClose(out_3d, v, atol=1e-6)

    def test_reconstruct_patches_int_dtype(self):
        # reconstruct_patches only reshapes/transposes/slices, so integer
        # inputs survive exactly. Build the patches layout manually to stay
        # independent of conv-based extract_patches dtype support.
        x = np.arange(2 * 8 * 8 * 3, dtype="int32").reshape(2, 8, 8, 3)
        patches = x.reshape(2, 2, 4, 2, 4, 3)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(2, 2, 2, 48)
        out = kimage.reconstruct_patches(
            patches, size=(4, 4), output_size=(8, 8)
        )
        self.assertEqual(standardize_dtype(out.dtype), "int32")
        self.assertAllClose(out, x)


class ImageOpsDtypeTest(testing.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    INT_DTYPES = [x for x in dtypes.INT_TYPES if x not in ("uint64", "int64")]
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]

    if backend.backend() == "torch":
        INT_DTYPES = [x for x in INT_DTYPES if x not in ("uint16", "uint32")]

    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_affine_transform(self, dtype):
        images = knp.ones((50, 50, 3), dtype=dtype)
        transform = knp.ones((8,), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(
            kimage.affine_transform(images, transform), expected_dtype
        )
        self.assertDType(
            kimage.AffineTransform().symbolic_call(images, transform),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_crop_images(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.crop_images(images, 0, 0, 3, 3), expected_dtype)
        self.assertDType(
            kimage.CropImages(0, 0, 3, 3).symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_elastic_transform(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.elastic_transform(images), expected_dtype)
        self.assertDType(
            kimage.ElasticTransform().symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.extract_patches(images, (3, 3)), expected_dtype)
        self.assertDType(
            kimage.ExtractPatches((3, 3)).symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_gaussian_blur(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.gaussian_blur(images), expected_dtype)
        self.assertDType(
            kimage.GaussianBlur().symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_hsv_to_rgb(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.hsv_to_rgb(images), expected_dtype)
        self.assertDType(
            kimage.HSVToRGB().symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_map_coordinates(self, dtype):
        inputs = knp.ones((3, 4, 5), dtype=dtype)
        coordinates = knp.stack([knp.ones((2, 3, 4), dtype=dtype)] * 3)
        expected_dtype = dtype

        self.assertDType(
            kimage.map_coordinates(inputs, coordinates, 0), expected_dtype
        )
        self.assertDType(
            kimage.MapCoordinates(0).symbolic_call(inputs, coordinates),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_pad_images(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.pad_images(images, 0, 0, 3, 3), expected_dtype)
        self.assertDType(
            kimage.PadImages(0, 0, 3, 3).symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_perspective_transform(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        start_points = krandom.uniform((1, 4, 2), dtype=dtype)
        end_points = krandom.uniform((1, 4, 2), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(
            kimage.perspective_transform(images, start_points, end_points),
            expected_dtype,
        )
        self.assertDType(
            kimage.PerspectiveTransform().symbolic_call(
                images, start_points, end_points
            ),
            expected_dtype,
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_resize(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.resize(images, (5, 5)), expected_dtype)
        self.assertDType(
            kimage.Resize((5, 5)).symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_rgb_to_grayscale(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.rgb_to_grayscale(images), expected_dtype)
        self.assertDType(
            kimage.RGBToGrayscale().symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_rgb_to_hsv(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(kimage.rgb_to_hsv(images), expected_dtype)
        self.assertDType(
            kimage.RGBToHSV().symbolic_call(images), expected_dtype
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_scale_and_translate(self, dtype):
        images = knp.ones((10, 10, 3), dtype=dtype)
        scale = knp.ones((2,), dtype=dtype)
        translation = knp.ones((2,), dtype=dtype)
        expected_dtype = dtype

        self.assertDType(
            kimage.scale_and_translate(
                images,
                output_shape=(15, 15, 3),
                scale=scale,
                translation=translation,
                spatial_dims=(0, 1),
                method="linear",
            ),
            expected_dtype,
        )
        self.assertDType(
            kimage.ScaleAndTranslate(
                spatial_dims=(0, 1), method="linear"
            ).symbolic_call(images, (15, 15, 3), scale, translation),
            expected_dtype,
        )


class ImageOpsBehaviorTests(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_rgb_to_grayscale_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.rgb_to_grayscale(invalid_image)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.RGBToGrayscale()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError,
            "Invalid images rank: expected rank 3",
        ):
            kimage.rgb_to_grayscale(invalid_image)

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_rgb_to_hsv_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.rgb_to_hsv(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.RGBToHSV()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.rgb_to_hsv(invalid_image)

    def test_rgb_to_grayscale_invalid_channels(self):
        error_msg = (
            r"Invalid channel size: expected 3 \(RGB\) or 1 \(Grayscale\)\."
        )
        invalid_image = np.random.uniform(size=(4, 4, 4)).astype("float32")
        with self.assertRaisesRegex(ValueError, error_msg):
            kimage.rgb_to_grayscale(invalid_image)

        invalid_image = KerasTensor(shape=(None, 4, 20, 20), dtype="float32")
        with self.assertRaisesRegex(ValueError, error_msg):
            kimage.rgb_to_grayscale(invalid_image, data_format="channels_first")
        with self.assertRaisesRegex(ValueError, error_msg):
            kimage.RGBToGrayscale(data_format="channels_first").symbolic_call(
                invalid_image
            )

    def test_rgb_to_hsv_invalid_dtype(self):
        invalid_image = np.random.uniform(size=(10, 10, 3)).astype("int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.rgb_to_hsv(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.RGBToHSV()(invalid_image)
        invalid_image = KerasTensor(shape=(10, 10, 3), dtype="int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.rgb_to_hsv(invalid_image)

    def test_rgb_to_hsv_invalid_channels(self):
        invalid_image = np.random.uniform(size=(4, 4, 3)).astype("float32")
        with self.assertRaises(Exception):
            kimage.rgb_to_hsv(invalid_image, data_format="channels_first")

        invalid_image = KerasTensor(shape=(None, 4, 20, 20), dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.rgb_to_hsv(invalid_image, data_format="channels_first")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.RGBToHSV(data_format="channels_first").symbolic_call(
                invalid_image
            )

        invalid_image = KerasTensor(shape=(4, 20, 20), dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.rgb_to_hsv(invalid_image, data_format="channels_first")

    def test_hsv_to_rgb_invalid_channels(self):
        invalid_image = np.random.uniform(size=(4, 4, 3)).astype("float32")
        with self.assertRaises(Exception):
            kimage.hsv_to_rgb(invalid_image, data_format="channels_first")

        invalid_image = KerasTensor(shape=(None, 4, 20, 20), dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.hsv_to_rgb(invalid_image, data_format="channels_first")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.HSVToRGB(data_format="channels_first").symbolic_call(
                invalid_image
            )

        invalid_image = KerasTensor(shape=(4, 20, 20), dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            "Input images must have 3 channels, but received images with "
            "4 channels\\.",
        ):
            kimage.hsv_to_rgb(invalid_image, data_format="channels_first")

    @parameterized.named_parameters(named_product(rank=[2, 5]))
    def test_hsv_to_rgb_invalid_rank(self, rank):
        shape = [3] * rank
        invalid_image = np.random.uniform(size=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.hsv_to_rgb(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.HSVToRGB()(invalid_image)
        invalid_image = KerasTensor(shape=shape)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.hsv_to_rgb(invalid_image)

    def test_hsv_to_rgb_invalid_dtype(self):
        invalid_image = np.random.uniform(size=(10, 10, 3)).astype("int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.hsv_to_rgb(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.HSVToRGB()(invalid_image)
        invalid_image = KerasTensor(shape=(10, 10, 3), dtype="int32")
        with self.assertRaisesRegex(
            ValueError, "Invalid images dtype: expected float dtype."
        ):
            kimage.hsv_to_rgb(invalid_image)

    def test_resize_invalid_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.resize(invalid_image, (5, 5))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.Resize((5, 5))(invalid_image)

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.resize(invalid_image, (5, 5))

    def test_affine_transform_invalid_images_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        transform = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.AffineTransform()(invalid_image, transform)

        # Test rank=5
        invalid_image = np.random.uniform(size=(2, 10, 10, 3, 1))
        transform = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.AffineTransform()(invalid_image, transform)

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        transform = KerasTensor(shape=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.affine_transform(invalid_image, transform)

    def test_affine_transform_invalid_transform_rank(self):
        # Test rank=3
        images = np.random.uniform(size=(10, 10, 3))
        invalid_transform = np.random.uniform(size=(2, 3, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test rank=0
        invalid_transform = np.random.uniform(size=())
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test rank=3, symbolic tensor
        images = KerasTensor(shape=(10, 10, 3))
        invalid_transform = KerasTensor(shape=(2, 3, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform rank: expected rank 1"
        ):
            kimage.affine_transform(images, invalid_transform)

    def test_affine_transform_invalid_transform_shape(self):
        # Test last dimension 6, rank 1
        images = np.random.uniform(size=(10, 10, 3))
        invalid_transform = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform shape: expected the last dimension"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform shape: expected the last dimension"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test last dimension 9, rank 2
        images = np.random.uniform(size=(2, 10, 10, 3))
        invalid_transform = np.random.uniform(size=(2, 9))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform shape: expected the last dimension"
        ):
            kimage.affine_transform(images, invalid_transform)
        with self.assertRaisesRegex(
            ValueError, "Invalid transform shape: expected the last dimension"
        ):
            kimage.AffineTransform()(images, invalid_transform)

        # Test last dimension 6, symbolic tensor
        images = KerasTensor(shape=(None, 10, 10, 3))
        invalid_transform = KerasTensor(shape=(None, 6))
        with self.assertRaisesRegex(
            ValueError, "Invalid transform shape: expected the last dimension"
        ):
            kimage.affine_transform(images, invalid_transform)

        # Test unknown last dimension, symbolic tensor
        transform = KerasTensor(shape=(None, None))
        out = kimage.affine_transform(images, transform)
        self.assertEqual(out.shape, (None, 10, 10, 3))

    def test_extract_patches_invalid_size(self):
        size = "5"  # Invalid size type
        image = np.random.uniform(size=(2, 20, 20, 3))
        with self.assertRaisesRegex(TypeError, "Expected an int or a tuple"):
            kimage.extract_patches(image, size)

        size = (3, 3, 3, 3)  # Invalid size, too many dimensions
        with self.assertRaisesRegex(
            ValueError, "Expected a tuple of length 2 or 3"
        ):
            kimage.extract_patches(image, size)

    def test_extract_patches_unified_3d(self):
        # Test that extract_patches handles 3D volumes when size has 3 elements
        # channels_last
        volume = np.random.uniform(size=(2, 20, 20, 20, 3)).astype("float32")
        patches = kimage.extract_patches(volume, (5, 5, 5))
        self.assertEqual(patches.shape, (2, 4, 4, 4, 375))

        # unbatched
        volume = np.random.uniform(size=(20, 20, 20, 3)).astype("float32")
        patches = kimage.extract_patches(volume, (5, 5, 5))
        self.assertEqual(patches.shape, (4, 4, 4, 375))

    def test_map_coordinates_invalid_coordinates_rank(self):
        # Test mismatched dim of coordinates
        image = np.random.uniform(size=(10, 10, 3))
        coordinates = np.random.uniform(size=(2, 10, 10))
        with self.assertRaisesRegex(
            ValueError, "must be the same as the rank of `inputs`"
        ):
            kimage.map_coordinates(image, coordinates, 0)
        with self.assertRaisesRegex(
            ValueError, "must be the same as the rank of `inputs`"
        ):
            kimage.MapCoordinates(0)(image, coordinates)

        # Test rank=1
        coordinates = np.random.uniform(size=(3,))
        with self.assertRaisesRegex(ValueError, "expected at least rank 2"):
            kimage.map_coordinates(image, coordinates, 0)
        with self.assertRaisesRegex(ValueError, "expected at least rank 2"):
            kimage.MapCoordinates(0)(image, coordinates)

    def test_crop_images_unknown_shape(self):
        # Test unknown height and target_height
        x = KerasTensor([None, 10, 3])
        with self.assertRaisesRegex(
            ValueError, "When the height of the images is unknown"
        ):
            kimage.crop_images(x, 2, 3, 4, 5)

        # Test unknown width and target_width
        x = KerasTensor([10, None, 3])
        with self.assertRaisesRegex(
            ValueError, "When the width of the images is unknown"
        ):
            kimage.crop_images(x, 2, 3, 4, 5)

    def test_pad_images_negative_output_spec(self):
        x = KerasTensor([10, 10, 3])
        with self.assertRaisesRegex(ValueError, "top_padding must be >= 0"):
            kimage.pad_images(
                x,
                None,
                None,
                1,
                0,
                target_height=3,
                target_width=10,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.pad_images(
                x,
                top_padding=0,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=-1,
                target_width=10,
            )

        x = KerasTensor([None, None, 3])
        with self.assertRaisesRegex(ValueError, "top_padding must be >= 0"):
            kimage.pad_images(
                x,
                top_padding=-1,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=5,
                target_width=5,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.pad_images(
                x,
                top_padding=0,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=-1,
                target_width=5,
            )

    def test_pad_images_negative_arguments(self):
        image = np.ones((10, 10, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, "top_padding must be >= 0"):
            kimage.pad_images(
                image,
                top_padding=-1,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=12,
                target_width=12,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.pad_images(
                image,
                top_padding=0,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=-1,
                target_width=12,
            )
        with self.assertRaisesRegex(ValueError, "target_width must be >= 0"):
            kimage.PadImages(
                top_padding=0,
                left_padding=0,
                bottom_padding=None,
                right_padding=None,
                target_height=12,
                target_width=-1,
            )(image)

    def test_crop_images_negative_output_spec(self):
        x = KerasTensor([10, 10, 3])
        with self.assertRaisesRegex(ValueError, "top_cropping must be >= 0"):
            kimage.crop_images(
                x,
                None,
                None,
                1,
                0,
                target_height=12,
                target_width=10,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.crop_images(
                x,
                top_cropping=0,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=-1,
                target_width=10,
            )

        x = KerasTensor([None, None, 3])
        with self.assertRaisesRegex(ValueError, "top_cropping must be >= 0"):
            kimage.crop_images(
                x,
                top_cropping=-1,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=5,
                target_width=5,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.crop_images(
                x,
                top_cropping=0,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=-1,
                target_width=5,
            )

    def test_crop_images_negative_arguments(self):
        image = np.ones((10, 10, 3), dtype="float32")
        with self.assertRaisesRegex(ValueError, "top_cropping must be >= 0"):
            kimage.crop_images(
                image,
                top_cropping=-1,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=8,
                target_width=8,
            )
        with self.assertRaisesRegex(ValueError, "target_height must be >= 0"):
            kimage.crop_images(
                image,
                top_cropping=0,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=-1,
                target_width=8,
            )
        with self.assertRaisesRegex(ValueError, "target_width must be >= 0"):
            kimage.CropImages(
                top_cropping=0,
                left_cropping=0,
                bottom_cropping=None,
                right_cropping=None,
                target_height=8,
                target_width=-1,
            )(image)

    def test_perspective_transform_invalid_images_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        start_points = np.random.uniform(size=(6,))
        end_points = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.perspective_transform(
                invalid_image, start_points, end_points
            )
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.PerspectiveTransform()(
                invalid_image, start_points, end_points
            )

        # Test rank=5
        invalid_image = np.random.uniform(size=(2, 10, 10, 3, 1))
        start_points = np.random.uniform(size=(6,))
        end_points = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.perspective_transform(
                invalid_image, start_points, end_points
            )
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.PerspectiveTransform()(
                invalid_image, start_points, end_points
            )

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        start_points = KerasTensor(shape=(6,))
        end_points = np.random.uniform(size=(6,))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.perspective_transform(
                invalid_image, start_points, end_points
            )

    def test_perspective_transform_invalid_points_rank(self):
        # Test rank=3
        images = np.random.uniform(size=(10, 10, 3))
        start_points = np.random.uniform(size=(2, 2, 4, 2))
        end_points = np.random.uniform(size=(2, 2, 4, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid start_points shape: expected"
        ):
            kimage.perspective_transform(images, start_points, end_points)
        with self.assertRaisesRegex(
            ValueError, "Invalid start_points shape: expected"
        ):
            kimage.PerspectiveTransform()(images, start_points, end_points)

        # Test rank=0
        start_points = np.random.uniform(size=())
        end_points = np.random.uniform(size=())
        with self.assertRaisesRegex(
            ValueError, "Invalid start_points shape: expected"
        ):
            kimage.perspective_transform(images, start_points, end_points)
        with self.assertRaisesRegex(
            ValueError, "Invalid start_points shape: expected"
        ):
            kimage.PerspectiveTransform()(images, start_points, end_points)

        # Test rank=3, symbolic tensor
        images = KerasTensor(shape=(10, 10, 3))
        start_points = KerasTensor(shape=(2, 3, 2))
        with self.assertRaisesRegex(
            ValueError, "Invalid start_points shape: expected"
        ):
            kimage.perspective_transform(images, start_points, end_points)

    def test_gaussian_blur_invalid_images_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        kernel_size = (3, 3)
        sigma = (0.1, 0.1)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.gaussian_blur(
                invalid_image, kernel_size=kernel_size, sigma=sigma
            )
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(
                invalid_image
            )

        # Test rank=5
        invalid_image = np.random.uniform(size=(2, 10, 10, 3, 1))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.gaussian_blur(
                invalid_image, kernel_size=kernel_size, sigma=sigma
            )
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(
                invalid_image
            )

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.gaussian_blur(
                invalid_image, kernel_size=kernel_size, sigma=sigma
            )

    def test_elastic_transform_invalid_images_rank(self):
        # Test rank=2
        invalid_image = np.random.uniform(size=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.elastic_transform(
                invalid_image,
            )
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.ElasticTransform()(invalid_image)

        # Test rank=5
        invalid_image = np.random.uniform(size=(2, 10, 10, 3, 1))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.elastic_transform(invalid_image)
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.ElasticTransform()(invalid_image)

        # Test rank=2, symbolic tensor
        invalid_image = KerasTensor(shape=(10, 10))
        with self.assertRaisesRegex(
            ValueError, "Invalid images rank: expected rank 3"
        ):
            kimage.elastic_transform(invalid_image)


class ExtractPatches3DTest(testing.TestCase):
    FLOAT_DTYPES = [x for x in dtypes.FLOAT_TYPES if x not in ("float64",)]

    def setUp(self):
        super().setUp()
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")

    def tearDown(self):
        super().tearDown()
        backend.set_image_data_format(self.data_format)

    @parameterized.named_parameters(
        named_product(
            dtype=FLOAT_DTYPES, data_format=["channels_last", "channels_first"]
        )
    )
    def test_extract_patches_3d_basic(self, dtype, data_format):
        if data_format == "channels_last":
            volume = np.ones((1, 96, 96, 96, 4), dtype=dtype)
            expected_shape = (1, 24, 24, 24, 256)
        else:
            volume = np.ones((1, 4, 96, 96, 96), dtype=dtype)
            expected_shape = (1, 256, 24, 24, 24)
        patches = kimage.extract_patches_3d(
            volume, size=(4, 4, 4), strides=(4, 4, 4), data_format=data_format
        )

        self.assertEqual(patches.shape, expected_shape)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_valid_padding(self, dtype):
        volume = np.random.rand(2, 32, 32, 32, 3)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(
            volume, size=(8, 8, 8), strides=(8, 8, 8), padding="valid"
        )
        self.assertEqual(patches.shape, (2, 4, 4, 4, 1536))

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_same_padding(self, dtype):
        volume = np.random.rand(1, 33, 33, 33, 1)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(
            volume, size=(4, 4, 4), strides=(4, 4, 4), padding="same"
        )
        expected_patches = (33 + 3) // 4  # = 9
        self.assertEqual(
            patches.shape,
            (1, expected_patches, expected_patches, expected_patches, 64),
        )

    @parameterized.named_parameters(
        named_product(
            dtype=FLOAT_DTYPES, data_format=["channels_last", "channels_first"]
        )
    )
    def test_extract_patches_3d_with_dilation(self, dtype, data_format):
        # Shape input according to data_format
        if data_format == "channels_last":
            volume = np.random.rand(1, 64, 64, 64, 2).astype(dtype)
        else:
            volume = np.random.rand(1, 2, 64, 64, 64).astype(dtype)

        # All backends (including TensorFlow) support dilation > 1 combined
        # with strides > 1.
        patches = kimage.extract_patches_3d(
            volume,
            size=(3, 3, 3),
            strides=(8, 8, 8),
            dilation_rate=(2, 2, 2),
            data_format=data_format,
        )
        # eff_p = 3 + (3 - 1) * (2 - 1) = 5
        # out = (64 - 5) // 8 + 1 = 8
        expected_patches = 8
        if data_format == "channels_last":
            expected_shape = (
                1,
                expected_patches,
                expected_patches,
                expected_patches,
                54,  # 2*3*3*3
            )
        else:
            expected_shape = (
                1,
                54,  # 2*3*3*3
                expected_patches,
                expected_patches,
                expected_patches,
            )
        self.assertEqual(patches.shape, expected_shape)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_overlapping(self, dtype):
        volume = np.random.rand(1, 16, 16, 16, 1)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(
            volume, size=(4, 4, 4), strides=(2, 2, 2)
        )
        expected_patches = (16 - 4) // 2 + 1  # = 7
        self.assertEqual(
            patches.shape,
            (1, expected_patches, expected_patches, expected_patches, 64),
        )

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_int_size(self, dtype):
        volume = np.random.rand(1, 24, 24, 24, 2)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(volume, size=6, strides=6)
        self.assertEqual(patches.shape, (1, 4, 4, 4, 432))

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_no_stride_provided(self, dtype):
        volume = np.random.rand(1, 24, 24, 24, 2)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(volume, size=6)
        # should default to strides = size - same results as above test
        self.assertEqual(patches.shape, (1, 4, 4, 4, 432))

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_unbatched(self, dtype):
        volume = np.random.rand(24, 24, 24, 2)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(volume, size=6)
        self.assertEqual(patches.shape, (4, 4, 4, 432))

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_value_check(self, dtype):
        volume = np.arange(8 * 8 * 8).reshape(1, 8, 8, 8, 1)
        volume = volume.astype(dtype)
        patches = kimage.extract_patches_3d(
            volume, size=(2, 2, 2), strides=(2, 2, 2)
        )
        first_patch = patches[0, 0, 0, 0, :]
        first_patch_np = backend.ops.convert_to_numpy(first_patch)

        expected = volume[0, 0:2, 0:2, 0:2, 0].flatten()
        np.testing.assert_array_equal(first_patch_np, expected)

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_invalid_size(self, dtype):
        volume = np.random.rand(1, 32, 32, 32, 1).astype(dtype)
        with self.assertRaises(TypeError):
            kimage.extract_patches_3d(volume, size=(4, 4))

    @parameterized.named_parameters(named_product(dtype=FLOAT_DTYPES))
    def test_extract_patches_3d_invalid_strides(self, dtype):
        volume = np.random.rand(1, 32, 32, 32, 1).astype(dtype)
        with self.assertRaises(ValueError):
            kimage.extract_patches_3d(volume, size=(4, 4, 4), strides=(2, 2))

    @parameterized.named_parameters(
        named_product(
            dtype=FLOAT_DTYPES, data_format=["channels_last", "channels_first"]
        )
    )
    def test_extract_patches_3d_non_cubic(self, dtype, data_format):
        if data_format == "channels_last":
            volume = np.random.rand(1, 32, 32, 32, 3).astype(dtype)
            expected_shape = (1, 16, 10, 8, 72)
        else:
            volume = np.random.rand(1, 3, 32, 32, 32).astype(dtype)
            expected_shape = (1, 72, 16, 10, 8)
        patches = kimage.extract_patches_3d(
            volume, size=(2, 3, 4), strides=(2, 3, 4), data_format=data_format
        )
        self.assertEqual(patches.shape, expected_shape)

    def test_ssim_identical_images(self):
        # Identical images should have SSIM close to 1.0
        x = np.random.random((32, 32, 3)).astype("float32")
        out = kimage.ssim(x, x, max_val=1.0)
        self.assertAllClose(out, 1.0, atol=1e-4)

        # Batched case
        x = np.random.random((2, 32, 32, 3)).astype("float32")
        out = kimage.ssim(x, x, max_val=1.0)
        self.assertEqual(out.shape, (2,))
        self.assertAllClose(out, [1.0, 1.0], atol=1e-4)

    def test_ssim_different_images(self):
        # Different random images should have lower SSIM
        np.random.seed(42)
        x1 = np.random.random((32, 32, 3)).astype("float32")
        x2 = np.random.random((32, 32, 3)).astype("float32")
        out = kimage.ssim(x1, x2, max_val=1.0)
        # SSIM of different images should be less than 1
        self.assertTrue(float(out) < 0.5)

    def test_ssim_vs_tensorflow(self):
        # Compare with TensorFlow's implementation
        np.random.seed(123)
        x1 = np.random.random((2, 32, 32, 3)).astype("float32")
        x2 = np.random.random((2, 32, 32, 3)).astype("float32")

        out = kimage.ssim(x1, x2, max_val=1.0)
        ref_out = tf.image.ssim(x1, x2, max_val=1.0)

        self.assertEqual(out.shape, ref_out.shape)
        self.assertAllClose(out, ref_out, atol=0.01)

    def test_ssim_channels_first(self):
        backend.set_image_data_format("channels_first")
        # Identical images should have SSIM close to 1.0
        x = np.random.random((3, 32, 32)).astype("float32")
        out = kimage.ssim(x, x, max_val=1.0)
        self.assertAllClose(out, 1.0, atol=1e-4)

        # Batched case
        x = np.random.random((2, 3, 32, 32)).astype("float32")
        out = kimage.ssim(x, x, max_val=1.0)
        self.assertEqual(out.shape, (2,))
        self.assertAllClose(out, [1.0, 1.0], atol=1e-4)

    def test_ssim_max_val_255(self):
        # Test with max_val=255 for uint8-like images
        x = (np.random.random((32, 32, 3)) * 255).astype("float32")
        out = kimage.ssim(x, x, max_val=255.0)
        self.assertAllClose(out, 1.0, atol=1e-4)

    def test_ssim_custom_parameters(self):
        # Test with custom filter_size and k values
        x = np.random.random((32, 32, 3)).astype("float32")
        out = kimage.ssim(
            x, x, max_val=1.0, filter_size=7, filter_sigma=1.0, k1=0.02, k2=0.06
        )
        self.assertAllClose(out, 1.0, atol=1e-4)


class SobelEdgesTest(testing.TestCase):
    def test_sobel_edges_shape_channels_last(self):
        image = np.random.random((2, 16, 16, 3)).astype("float32")
        edges = kimage.sobel_edges(image, data_format="channels_last")
        self.assertEqual(edges.shape, (2, 16, 16, 3, 2))

    def test_sobel_edges_shape_channels_first(self):
        image = np.random.random((2, 3, 16, 16)).astype("float32")
        edges = kimage.sobel_edges(image, data_format="channels_first")
        self.assertEqual(edges.shape, (2, 3, 16, 16, 2))

    def test_sobel_edges_single_channel(self):
        image = np.random.random((1, 16, 16, 1)).astype("float32")
        edges = kimage.sobel_edges(image, data_format="channels_last")
        self.assertEqual(edges.shape, (1, 16, 16, 1, 2))

    def test_sobel_edges_detects_vertical_edge(self):
        # Create image with vertical edge in the middle
        image = np.zeros((1, 8, 8, 1), dtype="float32")
        image[0, :, 4:, 0] = 1.0
        edges = kimage.sobel_edges(image, data_format="channels_last")

        # Horizontal gradient (dx) should be non-zero at the edge
        dx = backend.ops.convert_to_numpy(edges[0, :, :, 0, 1])
        # The edge is at column 4, so dx should have non-zero values there
        self.assertTrue(np.any(np.abs(dx[:, 3:5]) > 0))

    def test_sobel_edges_detects_horizontal_edge(self):
        # Create image with horizontal edge in the middle
        image = np.zeros((1, 8, 8, 1), dtype="float32")
        image[0, 4:, :, 0] = 1.0
        edges = kimage.sobel_edges(image, data_format="channels_last")

        # Vertical gradient (dy) should be non-zero at the edge
        dy = backend.ops.convert_to_numpy(edges[0, :, :, 0, 0])
        # The edge is at row 4, so dy should have non-zero values there
        self.assertTrue(np.any(np.abs(dy[3:5, :]) > 0))

    def test_sobel_edges_uniform_image(self):
        # Uniform image should have near-zero gradients (except at borders)
        image = np.ones((1, 16, 16, 1), dtype="float32") * 0.5
        edges = kimage.sobel_edges(image, data_format="channels_last")

        # Interior gradients should be zero
        edges_np = backend.ops.convert_to_numpy(edges)
        interior = edges_np[0, 2:-2, 2:-2, 0, :]
        self.assertAllClose(interior, np.zeros_like(interior), atol=1e-5)

    def test_sobel_edges_multi_channel(self):
        # Test with RGB image
        image = np.random.random((1, 16, 16, 3)).astype("float32")
        edges = kimage.sobel_edges(image, data_format="channels_last")
        self.assertEqual(edges.shape, (1, 16, 16, 3, 2))
