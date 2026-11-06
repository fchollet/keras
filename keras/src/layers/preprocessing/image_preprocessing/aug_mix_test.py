import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class AugMixTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            layers.AugMix,
            init_kwargs={
                "value_range": (0, 255),
                "num_chains": 2,
                "chain_depth": 2,
                "factor": 1,
                "alpha": 1.0,
                "all_ops": True,
                "interpolation": "nearest",
                "seed": 43,
                "data_format": "channels_last",
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_aug_mix_inference(self):
        seed = 3481
        layer = layers.AugMix()

        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_random_augment_randomness(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))

        layer = layers.AugMix(
            num_chains=11, all_ops=True, data_format=data_format
        )
        augmented_image = layer(input_data)

        self.assertNotAllClose(
            backend.ops.convert_to_numpy(augmented_image), input_data
        )

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.AugMix(data_format=data_format)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    def test_seed_reproducibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))

        first = layers.AugMix(seed=1234, data_format=data_format)(input_data)
        second = layers.AugMix(seed=1234, data_format=data_format)(input_data)

        self.assertAllClose(
            backend.ops.convert_to_numpy(first),
            backend.ops.convert_to_numpy(second),
        )

    def test_graph_issue(self):
        input_data = np.random.random((10, 8, 8, 3))
        layer = layers.AugMix()
        ds = (
            tf_data.Dataset.from_tensor_slices(input_data)
            .batch(2)
            .map(lambda x: layer.get_random_transformation(x)["layer_idxes"])
        )

        key_list = []
        for output in ds:
            key_list.append(backend.ops.convert_to_numpy(output))

        self.assertGreater(len(np.unique(np.stack(key_list), axis=0)), 1)
