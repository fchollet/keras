import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class MaxNumBoundingBoxesTest(testing.TestCase):
    def test_max_num_bounding_boxes_basics(self):
        self.run_layer_test(
            layers.MaxNumBoundingBoxes,
            init_kwargs={
                "max_number": 40,
                "fill_value": -1,
            },
            input_shape=(12, 12, 3),
            expected_output_shape=(12, 12, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_output_shapes(self):
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
            "labels": np.array([1, 2]),  # Dummy labels
        }
        layer = layers.MaxNumBoundingBoxes(
            max_number=40, bounding_box_format="xyxy"
        )

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        output = layer(input_data)
        self.assertEqual(output["bounding_boxes"]["boxes"].shape, (40, 4))
        self.assertEqual(output["bounding_boxes"]["labels"].shape, (40,))

    def test_output_shapes_with_tf_data(self):
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
        layer = layers.MaxNumBoundingBoxes(
            max_number=40, bounding_box_format="xyxy"
        )
        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        ds = tf_data.Dataset.from_tensor_slices(input_data)
        ds = ds.map(layer)
        ds = ds.batch(1)
        output = next(iter(ds))
        self.assertEqual(output["bounding_boxes"]["boxes"].shape, (1, 40, 4))
        self.assertEqual(output["bounding_boxes"]["labels"].shape, (1, 40))

    def test_fill_value_in_config(self):
        layer = layers.MaxNumBoundingBoxes(
            max_number=4, fill_value=0, bounding_box_format="xyxy"
        )
        revived = layers.MaxNumBoundingBoxes.from_config(layer.get_config())
        self.assertEqual(revived.fill_value, 0)

        bounding_boxes = {
            "boxes": np.array([[[1, 1, 2, 2]]]),
            "labels": np.array([[5]]),
        }
        output = revived(
            {"images": np.zeros((1, 8, 8, 3)), "bounding_boxes": bounding_boxes}
        )
        self.assertAllClose(
            output["bounding_boxes"]["labels"], np.array([[5, 0, 0, 0]])
        )
