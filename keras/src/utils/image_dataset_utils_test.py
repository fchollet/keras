import os
from unittest import mock

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.utils import file_utils
from keras.src.utils import image_dataset_utils
from keras.src.utils import image_utils
from keras.src.utils.module_utils import LazyModule
from keras.src.utils.module_utils import tensorflow as tf


class ImageDatasetFromDirectoryTest(testing.TestCase):
    def _get_images(self, count=16, color_mode="rgb"):
        width = height = 24
        imgs = []
        for _ in range(count):
            if color_mode == "grayscale":
                img = np.random.randint(0, 256, size=(height, width, 1))
            elif color_mode == "rgba":
                img = np.random.randint(0, 256, size=(height, width, 4))
            else:
                img = np.random.randint(0, 256, size=(height, width, 3))
            if backend.config.image_data_format() == "channels_first":
                img = np.transpose(img, (2, 0, 1))
            img = image_utils.array_to_img(img)
            imgs.append(img)
        return imgs

    def _prepare_directory(
        self,
        num_classes=2,
        nested_dirs=False,
        color_mode="rgb",
        count=16,
    ):
        # Generate paths to class subdirectories
        temp_dir = self.get_temp_dir()
        paths = []
        for class_index in range(num_classes):
            class_directory = f"class_{class_index}"
            if nested_dirs:
                class_paths = [
                    class_directory,
                    os.path.join(class_directory, "subfolder_1"),
                    os.path.join(class_directory, "subfolder_2"),
                    os.path.join(
                        class_directory, "subfolder_1", "sub-subfolder"
                    ),
                ]
            else:
                class_paths = [class_directory]
            for path in class_paths:
                os.mkdir(os.path.join(temp_dir, path))
            paths += class_paths

        # Save images to the paths
        i = 0
        for img in self._get_images(color_mode=color_mode, count=count):
            path = paths[i % len(paths)]
            if color_mode == "rgb":
                ext = "jpg"
            else:
                ext = "png"
            filename = os.path.join(path, f"image_{i}.{ext}")
            img.save(os.path.join(temp_dir, filename))
            i += 1
        return temp_dir

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_no_labels(self, format):
        # Test retrieving images without labels from a directory and its
        # subdirs.

        # Save a few extra images in the parent directory.
        directory = self._prepare_directory(count=7, num_classes=2)
        for i, img in enumerate(self._get_images(3)):
            filename = f"image_{i}.jpg"
            img.save(os.path.join(directory, filename))

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=5,
            image_size=(18, 18),
            labels=None,
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [5, 18, 18, 3]
        else:
            output_shape = [5, 3, 18, 18]
        self.assertEqual(dataset.class_names, None)
        batch = next(iter(dataset))
        # We return plain images
        self.assertEqual(list(batch.shape), output_shape)
        self.assertDType(batch, "float32")
        # Count samples
        batch_count = 0
        sample_count = 0
        for batch in dataset:
            batch_count += 1
            sample_count += batch.shape[0]
        self.assertEqual(batch_count, 2)
        self.assertEqual(sample_count, 10)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_binary(self, format):
        directory = self._prepare_directory(num_classes=2)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode="int",
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [8, 18, 18, 3]
        else:
            output_shape = [8, 3, 18, 18]
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")
        self.assertEqual(list(batch[1].shape), [8])
        self.assertDType(batch[1], "int32")

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode="binary",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")
        self.assertEqual(list(batch[1].shape), [8, 1])
        self.assertDType(batch[1], "float32")

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode="categorical",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")
        self.assertEqual(list(batch[1].shape), [8, 2])
        self.assertDType(batch[1], "float32")

    def test_static_shape_in_graph(self):
        directory = self._prepare_directory(num_classes=2)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory, batch_size=8, image_size=(18, 18), label_mode="int"
        )
        test_obj = self
        if backend.config.image_data_format() == "channels_last":
            output_shape = [None, 18, 18, 3]
        else:
            output_shape = [None, 3, 18, 18]

        @tf.function
        def symbolic_fn(ds):
            for x, _ in ds.take(1):
                test_obj.assertListEqual(x.shape.as_list(), output_shape)

        symbolic_fn(dataset)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_sample_count(self, format):
        directory = self._prepare_directory(num_classes=4, count=15)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            format=format,
        )
        sample_count = 0
        for batch in dataset:
            sample_count += batch.shape[0]
        self.assertEqual(sample_count, 15)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_multiclass(self, format):
        directory = self._prepare_directory(num_classes=4, count=15)

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [8, 18, 18, 3]
        else:
            output_shape = [8, 3, 18, 18]
        batch = next(iter(dataset))
        self.assertEqual(list(batch.shape), output_shape)

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            format=format,
        )
        sample_count = 0
        iterator = iter(dataset)
        for batch in dataset:
            sample_count += next(iterator).shape[0]
        self.assertEqual(sample_count, 15)

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode="int",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")
        self.assertEqual(list(batch[1].shape), [8])
        self.assertDType(batch[1], "int32")

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode="categorical",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")
        self.assertEqual(list(batch[1].shape), [8, 4])
        self.assertDType(batch[1], "float32")

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_color_modes(self, format):
        directory = self._prepare_directory(num_classes=4, color_mode="rgba")
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            color_mode="rgba",
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [8, 18, 18, 4]
        else:
            output_shape = [8, 4, 18, 18]
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")

        directory = self._prepare_directory(
            num_classes=4, color_mode="grayscale"
        )
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            color_mode="grayscale",
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [8, 18, 18, 1]
        else:
            output_shape = [8, 1, 18, 18]
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)
        self.assertDType(batch[0], "float32")

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_validation_split(self, format):
        directory = self._prepare_directory(num_classes=2, count=10)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=10,
            image_size=(18, 18),
            validation_split=0.2,
            subset="training",
            seed=1337,
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if backend.config.image_data_format() == "channels_last":
            train_output_shape = [8, 18, 18, 3]
            val_output_shape = [2, 18, 18, 3]
        else:
            train_output_shape = [8, 3, 18, 18]
            val_output_shape = [2, 3, 18, 18]
        self.assertEqual(list(batch[0].shape), train_output_shape)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=10,
            image_size=(18, 18),
            validation_split=0.2,
            subset="validation",
            seed=1337,
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), val_output_shape)

        (
            train_dataset,
            val_dataset,
        ) = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=10,
            image_size=(18, 18),
            validation_split=0.2,
            subset="both",
            seed=1337,
            format=format,
        )
        batch = next(iter(train_dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), train_output_shape)
        batch = next(iter(val_dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), val_output_shape)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_manual_labels(self, format):
        # Case: wrong number of labels
        directory = self._prepare_directory(num_classes=1, count=4)
        with self.assertRaisesRegex(ValueError, "match the number of files"):
            image_dataset_utils.image_dataset_from_directory(
                directory,
                batch_size=8,
                image_size=(18, 18),
                labels=[0, 1, 0],
                shuffle=False,
                format=format,
            )

        # Case: single directory
        directory = self._prepare_directory(num_classes=1, count=4)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            labels=[0, 1, 0, 1],
            shuffle=False,
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [18, 18, 3]
        else:
            output_shape = [3, 18, 18]
        self.assertEqual(dataset.class_names, ["0", "1"])
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), [4] + output_shape)
        self.assertAllClose(batch[1], [0, 1, 0, 1])

        # Case: multiple directories
        directory = self._prepare_directory(num_classes=3, count=6)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            labels=[0, 1, 0, 1, 1, 1],
            shuffle=False,
            format=format,
        )
        self.assertEqual(dataset.class_names, ["0", "1"])
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), [6] + output_shape)
        self.assertAllClose(batch[1], [0, 1, 0, 1, 1, 1])

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_follow_links(self, format):
        directory = self._prepare_directory(
            num_classes=2, count=25, nested_dirs=True
        )
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            format=format,
        )
        sample_count = 0
        for batch in dataset:
            sample_count += batch.shape[0]
        self.assertEqual(sample_count, 25)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_no_images(self, format):
        directory = self._prepare_directory(num_classes=2, count=0)
        with self.assertRaisesRegex(ValueError, "No images found."):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, format=format
            )

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_crop_to_aspect_ratio(self, format):
        directory = self._prepare_directory(num_classes=2, count=5)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=5,
            image_size=(18, 18),
            crop_to_aspect_ratio=True,
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [5, 18, 18, 3]
        else:
            output_shape = [5, 3, 18, 18]
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_pad_to_aspect_ratio(self, format):
        directory = self._prepare_directory(num_classes=2, count=5)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=5,
            image_size=(18, 18),
            pad_to_aspect_ratio=True,
            format=format,
        )
        if backend.config.image_data_format() == "channels_last":
            output_shape = [5, 18, 18, 3]
        else:
            output_shape = [5, 3, 18, 18]
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertEqual(list(batch[0].shape), output_shape)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_errors(self, format):
        directory = self._prepare_directory(num_classes=3, count=5)

        with self.assertRaisesRegex(ValueError, "`labels` argument should be"):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, labels="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`label_mode` argument must be"
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, label_mode="other", format=format
            )

        with self.assertRaisesRegex(ValueError, "`color_mode` must be one of"):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, color_mode="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, 'only pass `class_names` if `labels="inferred"`'
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory,
                labels=[0, 0, 1, 1, 1],
                class_names=["class_0", "class_1", "class_2"],
                format=format,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected the lengths of `labels` to match the number of files",
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, labels=[0, 0, 1, 1], format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`class_names` passed did not match"
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, class_names=["class_0", "wrong_class"], format=format
            )

        with self.assertRaisesRegex(ValueError, "there must be exactly 2"):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, label_mode="binary", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`validation_split` must be between 0 and 1"
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, validation_split=2, format=format
            )

        with self.assertRaisesRegex(
            ValueError,
            '`subset` must be either "training", "validation" or "both"',
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory, validation_split=0.2, subset="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`validation_split` must be set"
        ):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory,
                validation_split=0.0,
                subset="training",
                format=format,
            )

        with self.assertRaisesRegex(ValueError, "must provide a `seed`"):
            _ = image_dataset_utils.image_dataset_from_directory(
                directory,
                validation_split=0.2,
                subset="training",
                format=format,
            )

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_not_batched(self, format):
        directory = self._prepare_directory(num_classes=2, count=2)
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=None,
            image_size=(18, 18),
            label_mode=None,
            shuffle=False,
            format=format,
        )
        sample = next(iter(dataset))
        self.assertEqual(len(sample.shape), 3)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_shuffle(self, format):
        directory = self._prepare_directory(
            num_classes=2, count=25, nested_dirs=True
        )
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=False,
            format=format,
        )
        unshuffled_batches = []
        for b in dataset:
            unshuffled_batches.append(self.convert_to_numpy(b))
        unshuffled_batches = np.concatenate(unshuffled_batches, axis=0)

        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=True,
            seed=1337,
            format=format,
        )
        batches_1 = []
        batches_2 = []
        for b in dataset:
            batches_1.append(self.convert_to_numpy(b))
        batches_1 = np.concatenate(batches_1, axis=0)
        for b in dataset:
            batches_2.append(self.convert_to_numpy(b))
        batches_2 = np.concatenate(batches_2, axis=0)
        self.assertNotAllClose(unshuffled_batches, batches_1, atol=1e-6)

        if format == "tf":
            self.assertNotAllClose(batches_1, batches_2, atol=1e-6)
        else:
            # Grain shuffles deterministically, so we expect the same batches.
            self.assertAllClose(batches_1, batches_2, atol=1e-6)

        # Test random seed determinism
        dataset = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=True,
            seed=1337,
            format=format,
        )
        batches_1_alt = []
        for b in dataset:
            batches_1_alt.append(self.convert_to_numpy(b))
        batches_1_alt = np.concatenate(batches_1_alt, axis=0)
        self.assertAllClose(batches_1, batches_1_alt, atol=1e-6)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_image_dataset_from_directory_shuffle_with_split(self, format):
        directory = self._prepare_directory(
            num_classes=2, count=25, nested_dirs=True
        )
        train_ds, val_ds = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=False,
            validation_split=0.2,
            subset="both",
            seed=1337,
            format=format,
        )
        unshuffled_train_batches = []
        for b in train_ds:
            unshuffled_train_batches.append(self.convert_to_numpy(b))
        unshuffled_train_batches = np.concatenate(
            unshuffled_train_batches, axis=0
        )

        train_ds, val_ds = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=True,
            seed=1337,
            validation_split=0.2,
            subset="both",
            format=format,
        )
        train_batches_1 = []
        val_batches_1 = []
        for b in train_ds:
            train_batches_1.append(self.convert_to_numpy(b))
        train_batches_1 = np.concatenate(train_batches_1, axis=0)
        for b in val_ds:
            val_batches_1.append(self.convert_to_numpy(b))
        val_batches_1 = np.concatenate(val_batches_1, axis=0)

        train_batches_2 = []
        val_batches_2 = []
        for b in train_ds:
            train_batches_2.append(self.convert_to_numpy(b))
        train_batches_2 = np.concatenate(train_batches_2, axis=0)
        for b in val_ds:
            val_batches_2.append(self.convert_to_numpy(b))
        val_batches_2 = np.concatenate(val_batches_2, axis=0)

        self.assertNotAllClose(
            unshuffled_train_batches, train_batches_1, atol=1e-6
        )

        if format == "tf":
            self.assertNotAllClose(train_batches_1, train_batches_2, atol=1e-6)
            # validation dataset is never shuffled, so it will always
            # yield the same order
            self.assertAllClose(val_batches_1, val_batches_2, atol=1e-6)
        else:
            # Grain shuffles deterministically, so we expect the same batches.
            self.assertAllClose(train_batches_1, train_batches_2, atol=1e-6)
            self.assertAllClose(val_batches_1, val_batches_2, atol=1e-6)

        # Test random seed determinism
        train_ds, val_ds = image_dataset_utils.image_dataset_from_directory(
            directory,
            batch_size=8,
            image_size=(18, 18),
            label_mode=None,
            follow_links=True,
            shuffle=True,
            seed=1337,
            validation_split=0.2,
            subset="both",
            format=format,
        )
        train_batches_1_alt = []
        val_batches_1_alt = []
        for b in train_ds:
            train_batches_1_alt.append(self.convert_to_numpy(b))
        train_batches_1_alt = np.concatenate(train_batches_1_alt, axis=0)
        for b in val_ds:
            val_batches_1_alt.append(self.convert_to_numpy(b))
        val_batches_1_alt = np.concatenate(val_batches_1_alt, axis=0)

        self.assertAllClose(train_batches_1, train_batches_1_alt, atol=1e-6)
        self.assertAllClose(val_batches_1, val_batches_1_alt, atol=1e-6)

    def test_image_dataset_from_directory_remote(self):
        directory = self._prepare_directory(
            num_classes=2, count=8, nested_dirs=True
        )
        with (
            mock.patch.object(file_utils, "is_remote_path", lambda path: True),
            mock.patch.object(
                file_utils, "gfile", LazyModule("keras_test_unavailable_module")
            ),
        ):
            dataset = image_dataset_utils.image_dataset_from_directory(
                directory,
                batch_size=8,
                image_size=(18, 18),
                follow_links=True,
                format="grain",
            )
            images, labels = next(iter(dataset))
            self.assertEqual(images.shape[0], 8)
            self.assertEqual(list(labels.shape), [8])

            dataset = image_dataset_utils.image_dataset_from_directory(
                directory,
                batch_size=8,
                image_size=(18, 18),
                labels=None,
                follow_links=True,
                format="grain",
            )
            images = next(iter(dataset))
            self.assertEqual(images.shape[0], 8)
