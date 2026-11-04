import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.common import standardize_dtype


def _to_data_format(x, data_format):
    if data_format == "channels_first":
        return np.transpose(x, (0, 3, 1, 2))
    return x


def _to_channels_last(x, data_format):
    if data_format == "channels_first":
        return np.transpose(x, (0, 2, 3, 1))
    return x


def _reference_mosaic(images, permutation_order, mosaic_centers):
    """NumPy reference: build the 2x2 grid and slice the window per image."""
    batch_size, height, width, _ = images.shape
    outputs = np.empty_like(images)
    for i in range(batch_size):
        top_left, top_right, bottom_left, bottom_right = images[
            permutation_order[i]
        ]
        grid = np.concatenate(
            [
                np.concatenate([top_left, top_right], axis=1),
                np.concatenate([bottom_left, bottom_right], axis=1),
            ],
            axis=0,
        )
        center_x = int(np.round(mosaic_centers[i, 0] * width))
        center_y = int(np.round(mosaic_centers[i, 1] * height))
        outputs[i] = grid[
            height - center_y : 2 * height - center_y,
            width - center_x : 2 * width - center_x,
        ]
    return outputs


class MosaicTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            layers.Mosaic,
            init_kwargs={
                "offset": (0.25, 0.75),
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_mosaic_inference(self):
        layer = layers.Mosaic(seed=1)
        inputs = np.random.random((4, 8, 8, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_unbatched_matches_batch_of_one(self):
        data_format = backend.config.image_data_format()
        image = _to_data_format(
            np.random.random((1, 10, 12, 3)).astype("float32"), data_format
        )[0]
        mask = _to_data_format(
            np.random.randint(0, 4, (1, 10, 12, 1)).astype("uint8"),
            data_format,
        )[0]
        label = np.array([0.0, 1.0, 0.0], dtype="float32")
        boxes = np.array([[2, 3, 8, 9]], dtype="float32")
        box_labels = np.array([1])

        def mosaic(data):
            layer = layers.Mosaic(
                seed=1, bounding_box_format="xyxy", data_format=data_format
            )
            return layer(data)

        unbatched = mosaic(
            {
                "images": image,
                "labels": label,
                "segmentation_masks": mask,
                "bounding_boxes": {"boxes": boxes, "labels": box_labels},
            }
        )
        batched = mosaic(
            {
                "images": image[None],
                "labels": label[None],
                "segmentation_masks": mask[None],
                "bounding_boxes": {
                    "boxes": boxes[None],
                    "labels": box_labels[None],
                },
            }
        )

        self.assertAllClose(unbatched["images"], batched["images"][0])
        self.assertAllClose(unbatched["labels"], batched["labels"][0])
        self.assertAllEqual(
            unbatched["segmentation_masks"], batched["segmentation_masks"][0]
        )
        self.assertAllClose(
            unbatched["bounding_boxes"]["boxes"],
            batched["bounding_boxes"]["boxes"][0],
        )
        self.assertAllEqual(
            unbatched["bounding_boxes"]["labels"],
            batched["bounding_boxes"]["labels"][0],
        )
        self.assertEqual(
            tuple(unbatched["bounding_boxes"]["boxes"].shape), (4, 4)
        )

    @parameterized.named_parameters(
        ("tuple", (0.2, 0.6), (0.2, 0.6)),
        ("list", [0.2, 0.6], (0.2, 0.6)),
        ("single_float", 0.5, (0.0, 0.5)),
        ("constant", (0.5, 0.5), (0.5, 0.5)),
    )
    def test_offset(self, offset, expected_offset):
        layer = layers.Mosaic(offset=offset)
        self.assertEqual(layer.offset, expected_offset)

    @parameterized.named_parameters(
        ("below_range", (-0.5, 0.5)),
        ("above_range", (0.5, 1.5)),
        ("reversed", (0.8, 0.2)),
        ("wrong_length", (0.1, 0.2, 0.3)),
        ("single_float_out_of_range", 1.5),
        ("wrong_type", "0.5"),
    )
    def test_invalid_offset(self, offset):
        with self.assertRaises(ValueError):
            layers.Mosaic(offset=offset)

    def test_random_transformation(self):
        data_format = backend.config.image_data_format()
        images = _to_data_format(np.random.random((4, 10, 12, 3)), data_format)
        layer = layers.Mosaic(offset=(0.3, 0.7), seed=1)
        transformation = layer.get_random_transformation({"images": images})

        permutation_order = backend.ops.convert_to_numpy(
            transformation["permutation_order"]
        )
        self.assertEqual(permutation_order.shape, (4, 4))
        self.assertAllEqual(permutation_order[:, 0], np.arange(4))
        self.assertTrue(np.all(permutation_order >= 0))
        self.assertTrue(np.all(permutation_order < 4))

        mosaic_centers = backend.ops.convert_to_numpy(
            transformation["mosaic_centers"]
        )
        self.assertEqual(mosaic_centers.shape, (4, 2))
        self.assertTrue(np.all(mosaic_centers >= 0.3 - 1e-6))
        self.assertTrue(np.all(mosaic_centers <= 0.7 + 1e-6))
        # Centers are snapped to whole pixels.
        pixels = mosaic_centers * np.array([12, 10])
        self.assertAllClose(pixels, np.round(pixels), atol=1e-4)

    def test_mosaic_basic(self):
        data_format = backend.config.image_data_format()
        # Image `k` is filled with the value `k`.
        images = np.stack([np.full((4, 4, 1), k) for k in range(4)])
        images = _to_data_format(images.astype("float32"), data_format)
        # Center at 1 pixel from the left and 2 pixels from the top.
        transformation = {
            "permutation_order": np.array(
                [[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 3], [3, 2, 1, 0]]
            ),
            "mosaic_centers": np.array([[0.25, 0.5]] * 4),
        }
        expected_output = np.array(
            [
                [[0, 1, 1, 1], [0, 1, 1, 1], [2, 3, 3, 3], [2, 3, 3, 3]],
                [[3, 2, 2, 2], [3, 2, 2, 2], [1, 0, 0, 0], [1, 0, 0, 0]],
            ]
            * 2,
            dtype="float32",
        )[..., None]

        layer = layers.Mosaic(data_format=data_format)
        output = layer.transform_images(images, transformation)

        output = _to_channels_last(
            backend.ops.convert_to_numpy(output), data_format
        )
        self.assertAllClose(output, expected_output)

    def test_mosaic_matches_reference(self):
        data_format = backend.config.image_data_format()
        images = np.random.random((4, 10, 12, 3)).astype("float32")
        layer = layers.Mosaic(seed=1, data_format=data_format)
        inputs = _to_data_format(images, data_format)
        transformation = layer.get_random_transformation({"images": inputs})

        output = layer.transform_images(inputs, transformation)

        expected_output = _reference_mosaic(
            images,
            backend.ops.convert_to_numpy(transformation["permutation_order"]),
            backend.ops.convert_to_numpy(transformation["mosaic_centers"]),
        )
        output = _to_channels_last(
            backend.ops.convert_to_numpy(output), data_format
        )
        self.assertAllClose(output, expected_output)

    @parameterized.named_parameters(
        ("float_labels", "float32"),
        ("float16_labels", "float16"),
    )
    def test_mosaic_labels(self, dtype):
        # Center (0.3, 0.6): the 4 images cover 18%, 42%, 12% and 28% of
        # the output.
        labels = np.eye(3)[[0, 1, 2, 0]].astype(dtype)
        transformation = {
            "permutation_order": np.array([[0, 1, 2, 3]] * 4),
            "mosaic_centers": np.array([[0.3, 0.6]] * 4, dtype="float32"),
        }
        layer = layers.Mosaic()
        output = layer.transform_labels(labels, transformation)
        self.assertAllClose(output, np.array([[0.46, 0.42, 0.12]] * 4))

    def test_mosaic_labels_match_pixel_area(self):
        data_format = backend.config.image_data_format()
        # Image `k` is filled with the value `k` and has label `k`.
        images = np.stack([np.full((10, 12, 3), k) for k in range(4)])
        images = _to_data_format(images.astype("float32"), data_format)
        labels = np.eye(4, dtype="float32")
        layer = layers.Mosaic(seed=1, data_format=data_format)

        output = layer({"images": images, "labels": labels})

        output_images = _to_channels_last(
            backend.ops.convert_to_numpy(output["images"]), data_format
        )
        source = output_images[..., 0].astype("int32")
        pixel_share = np.stack(
            [(source == k).mean(axis=(1, 2)) for k in range(4)], axis=1
        )
        self.assertAllClose(output["labels"], pixel_share, atol=1e-5)

    def test_mosaic_segmentation_masks(self):
        data_format = backend.config.image_data_format()
        classes = np.array([1, 3, 4, 7], dtype="uint8")
        # Image `k` is filled with the value `k`, mask `k` with `classes[k]`.
        images = np.stack([np.full((10, 12, 3), k) for k in range(4)])
        masks = np.stack([np.full((10, 12, 1), c) for c in classes])
        images = _to_data_format(images.astype("float32"), data_format)
        masks = _to_data_format(masks.astype("uint8"), data_format)
        layer = layers.Mosaic(seed=1, data_format=data_format)

        output = layer({"images": images, "segmentation_masks": masks})

        output_masks = backend.ops.convert_to_numpy(
            output["segmentation_masks"]
        )
        self.assertEqual(output_masks.dtype, np.uint8)
        self.assertTrue(set(np.unique(output_masks)).issubset(set(classes)))
        # Every mask pixel comes from the same image as its image pixel.
        output_images = _to_channels_last(
            backend.ops.convert_to_numpy(output["images"]), data_format
        )
        output_masks = _to_channels_last(output_masks, data_format)
        self.assertAllEqual(
            classes[output_images[..., 0].astype("int32")],
            output_masks[..., 0],
        )

    def test_mosaic_bounding_boxes(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_shape = (4, 100, 100, 3)
        else:
            input_shape = (4, 3, 100, 100)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [[80, 70, 95, 90], [5, 5, 20, 20]],
                    [[10, 50, 60, 95], [0, 0, 0, 0]],
                    [[60, 10, 100, 40], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                dtype="float32",
            ),
            "labels": np.array([[0, 0], [1, -1], [2, -1], [-1, -1]]),
        }
        # Center (0.3, 0.6) of a 100x100 image is at pixel (30, 60).
        transformation = {
            "permutation_order": np.array([[0, 1, 2, 3]] * 4),
            "mosaic_centers": np.array([[0.3, 0.6]] * 4, dtype="float32"),
            "input_shape": input_shape,
        }
        layer = layers.Mosaic(
            bounding_box_format="xyxy", data_format=data_format
        )

        output = layer.transform_bounding_boxes(bounding_boxes, transformation)

        output_boxes = backend.ops.convert_to_numpy(output["boxes"])
        output_labels = backend.ops.convert_to_numpy(output["labels"])
        self.assertEqual(output_boxes.shape, (4, 8, 4))
        # The second box of image 0 ends up outside of the output.
        self.assertAllEqual(output_labels[0], [0, -1, 1, -1, 2, -1, -1, -1])
        self.assertAllClose(output_boxes[0, 0], [10, 30, 25, 50])
        self.assertAllClose(output_boxes[0, 2], [40, 10, 90, 55])
        # Clipped at the left edge of the output.
        self.assertAllClose(output_boxes[0, 4], [0, 70, 30, 100])

    def test_tf_data_compatibility(self):
        tf_data = pytest.importorskip("tensorflow").data
        data_format = backend.config.image_data_format()
        images = _to_data_format(np.random.random((8, 8, 8, 3)), data_format)
        labels = np.eye(4, dtype="float32")[np.arange(8) % 4]
        bounding_boxes = {
            "boxes": np.array([[[1, 1, 3, 3]]] * 8, dtype="float32"),
            "labels": np.array([[1]] * 8),
        }
        layer = layers.Mosaic(
            seed=1, bounding_box_format="xyxy", data_format=data_format
        )

        ds = (
            tf_data.Dataset.from_tensor_slices(
                {
                    "images": images,
                    "labels": labels,
                    "bounding_boxes": bounding_boxes,
                }
            )
            .batch(4)
            .map(layer)
        )
        for output in ds.take(1):
            self.assertEqual(output["images"].shape, (4,) + images.shape[1:])
            self.assertEqual(output["labels"].shape, (4, 4))
            self.assertEqual(output["bounding_boxes"]["boxes"].shape, (4, 4, 4))

    def test_grain_compatibility(self):
        grain = pytest.importorskip("grain")
        data_format = backend.config.image_data_format()
        images = _to_data_format(
            np.random.random((8, 8, 8, 3)).astype("float32"), data_format
        )
        labels = np.eye(4, dtype="float32")[np.arange(8) % 4]
        elements = [
            {
                "images": images[i],
                "labels": labels[i],
                "bounding_boxes": {
                    "boxes": np.array([[1, 1, 3, 3]], dtype="float32"),
                    "labels": np.array([1]),
                },
            }
            for i in range(8)
        ]
        layer = layers.Mosaic(
            seed=1, bounding_box_format="xyxy", data_format=data_format
        )

        ds = grain.MapDataset.source(elements).batch(4).map(layer)
        output = ds[0]
        self.assertEqual(tuple(output["images"].shape), (4,) + images.shape[1:])
        self.assertEqual(tuple(output["labels"].shape), (4, 4))
        self.assertEqual(
            tuple(output["bounding_boxes"]["boxes"].shape), (4, 4, 4)
        )

    @parameterized.named_parameters(
        ("sparse", np.array([0, 1, 2, 0])),
        ("class_ids", np.array([[3], [7], [0], [9]], dtype="uint8")),
        ("int_one_hot", np.eye(3, dtype="int32")[[0, 1, 2, 0]]),
    )
    def test_invalid_labels_raise(self, labels):
        images = np.random.random((4, 8, 8, 3))
        layer = layers.Mosaic(seed=1)
        with self.assertRaisesRegex(ValueError, "one-hot"):
            layer({"images": images, "labels": labels})

    def test_batch_of_one(self):
        data_format = backend.config.image_data_format()
        images = np.random.random((1, 10, 12, 3)).astype("float32")
        labels = np.array([[0.0, 1.0, 0.0]], dtype="float32")
        layer = layers.Mosaic(seed=1, data_format=data_format)
        inputs = _to_data_format(images, data_format)
        transformation = layer.get_random_transformation({"images": inputs})

        output_images = layer.transform_images(inputs, transformation)
        output_labels = layer.transform_labels(labels, transformation)

        # The only image is combined with itself.
        expected_images = _reference_mosaic(
            images,
            backend.ops.convert_to_numpy(transformation["permutation_order"]),
            backend.ops.convert_to_numpy(transformation["mosaic_centers"]),
        )
        output_images = _to_channels_last(
            backend.ops.convert_to_numpy(output_images), data_format
        )
        self.assertAllClose(output_images, expected_images)
        self.assertAllClose(output_labels, labels)

    @parameterized.named_parameters(
        ("left_top_edge", (0.0, 0.0)),
        ("right_bottom_edge", (1.0, 1.0)),
        ("full_range", (0.0, 1.0)),
    )
    def test_extreme_offsets(self, offset):
        data_format = backend.config.image_data_format()
        images = np.random.random((4, 10, 12, 3)).astype("float32")
        labels = np.eye(4, dtype="float32")
        layer = layers.Mosaic(offset=offset, seed=1, data_format=data_format)
        inputs = _to_data_format(images, data_format)
        transformation = layer.get_random_transformation({"images": inputs})

        output_images = layer.transform_images(inputs, transformation)
        output_labels = layer.transform_labels(labels, transformation)

        expected_images = _reference_mosaic(
            images,
            backend.ops.convert_to_numpy(transformation["permutation_order"]),
            backend.ops.convert_to_numpy(transformation["mosaic_centers"]),
        )
        output_images = _to_channels_last(
            backend.ops.convert_to_numpy(output_images), data_format
        )
        self.assertAllClose(output_images, expected_images)
        output_labels = backend.ops.convert_to_numpy(output_labels)
        self.assertAllClose(np.sum(output_labels, axis=-1), np.ones(4))

    def test_seed_determinism(self):
        images = np.random.random((4, 8, 8, 3)).astype("float32")
        labels = np.eye(4, dtype="float32")
        data = {"images": images, "labels": labels}

        output_a = layers.Mosaic(seed=7)(dict(data))
        output_b = layers.Mosaic(seed=7)(dict(data))
        self.assertAllClose(output_a["images"], output_b["images"])
        self.assertAllClose(output_a["labels"], output_b["labels"])

        # Repeated calls on the same layer draw new random values.
        layer = layers.Mosaic(seed=7)
        output_1 = layer(dict(data))
        output_2 = layer(dict(data))
        self.assertNotAllClose(output_1["images"], output_2["images"])

    def test_mixed_precision(self):
        data_format = backend.config.image_data_format()
        layer = layers.Mosaic(
            seed=1, dtype="mixed_float16", data_format=data_format
        )

        # Centers stay on the pixel grid even for large images.
        large_images = _to_data_format(
            np.zeros((4, 640, 640, 1), dtype="float32"), data_format
        )
        transformation = layer.get_random_transformation(
            {"images": large_images}
        )
        # Multiply in float64, a float16 product would hide the error.
        mosaic_centers = backend.ops.convert_to_numpy(
            transformation["mosaic_centers"]
        ).astype("float64")
        pixels = mosaic_centers * 640
        self.assertAllClose(pixels, np.round(pixels), atol=1e-3)

        images = _to_data_format(
            np.random.random((4, 10, 12, 3)).astype("float32"), data_format
        )
        output = layer(
            {
                "images": images,
                "labels": np.eye(4, dtype="float32"),
            }
        )
        self.assertEqual(standardize_dtype(output["images"].dtype), "float16")
        self.assertEqual(standardize_dtype(output["labels"].dtype), "float16")
        self.assertAllClose(
            np.sum(backend.ops.convert_to_numpy(output["labels"]), axis=-1),
            np.ones(4),
            atol=1e-2,
        )

    def test_bounding_boxes_formats_agree(self):
        data_format = backend.config.image_data_format()
        images = _to_data_format(
            np.random.random((4, 20, 20, 3)).astype("float32"), data_format
        )
        boxes_xyxy = np.array(
            [[[2, 3, 8, 9], [10, 12, 18, 19]]] * 4, dtype="float32"
        )
        boxes_xywh = boxes_xyxy.copy()
        boxes_xywh[..., 2:] -= boxes_xywh[..., :2]
        labels = np.array([[1, 2]] * 4)

        output_xyxy = layers.Mosaic(
            seed=3, bounding_box_format="xyxy", data_format=data_format
        )(
            {
                "images": images,
                "bounding_boxes": {"boxes": boxes_xyxy, "labels": labels},
            }
        )
        output_xywh = layers.Mosaic(
            seed=3, bounding_box_format="xywh", data_format=data_format
        )(
            {
                "images": images,
                "bounding_boxes": {"boxes": boxes_xywh, "labels": labels},
            }
        )

        result_xyxy = backend.ops.convert_to_numpy(
            output_xyxy["bounding_boxes"]["boxes"]
        )
        result_xywh = backend.ops.convert_to_numpy(
            output_xywh["bounding_boxes"]["boxes"]
        )
        result_xywh[..., 2:] += result_xywh[..., :2]
        self.assertAllClose(result_xywh, result_xyxy)
        self.assertAllEqual(
            output_xywh["bounding_boxes"]["labels"],
            output_xyxy["bounding_boxes"]["labels"],
        )
