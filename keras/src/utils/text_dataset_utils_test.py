import os
import random
import string
from unittest import mock

from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.utils import file_utils
from keras.src.utils import text_dataset_utils
from keras.src.utils.module_utils import LazyModule


class TextDatasetFromDirectoryTest(testing.TestCase):
    def _prepare_directory(
        self, num_classes=2, nested_dirs=False, count=16, length=20
    ):
        # Get a unique temp directory
        temp_dir = self.get_temp_dir()

        # Generate paths to class subdirectories
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

        for i in range(count):
            path = paths[i % len(paths)]
            filename = os.path.join(path, f"text_{i}.txt")
            with open(os.path.join(temp_dir, filename), "w") as f:
                text = "".join(
                    [random.choice(string.printable) for _ in range(length)]
                )
                f.write(text)
        return temp_dir

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_standalone(self, format):
        # Test retrieving txt files without labels from a directory and its
        # subdirs. Save a few extra files in the parent directory.
        directory = self._prepare_directory(count=7, num_classes=2)
        for i in range(3):
            filename = f"text_{i}.txt"
            with open(os.path.join(directory, filename), "w") as f:
                text = "".join(
                    [random.choice(string.printable) for _ in range(20)]
                )
                f.write(text)

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=5,
            label_mode=None,
            max_length=10,
            format=format,
        )
        batch = next(iter(dataset))
        # We just return the texts, no labels
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(list(batch.shape), [5])
            self.assertDType(batch, "string")
        else:
            self.assertLen(batch, 5)
            self.assertIsInstance(batch[0], str)
        # Count samples
        batch_count = 0
        sample_count = 0
        for batch in dataset:
            batch_count += 1
            sample_count += len(batch)
        self.assertEqual(batch_count, 2)
        self.assertEqual(sample_count, 10)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_binary(self, format=format):
        directory = self._prepare_directory(num_classes=2)
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=8,
            label_mode="int",
            max_length=10,
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(batch[0].shape, (8,))
            self.assertDType(batch[0], "string")
            self.assertEqual(len(batch[0].numpy()[0]), 10)  # Test max_length
        else:
            self.assertLen(batch[0], 8)
            self.assertIsInstance(batch[0][0], str)
            self.assertLen(batch[0][0], 10)  # Test max_length
        self.assertEqual(list(batch[1].shape), [8])
        self.assertDType(batch[1], "int32")

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=8,
            label_mode="binary",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(list(batch[0].shape), [8])
            self.assertEqual(batch[0].dtype.name, "string")
        else:
            self.assertLen(batch[0], 8)
            self.assertIsInstance(batch[0][0], str)
        self.assertEqual(list(batch[1].shape), [8, 1])
        self.assertDType(batch[1], "float32")

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=8,
            label_mode="categorical",
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(list(batch[0].shape), [8])
            self.assertEqual(batch[0].dtype.name, "string")
        else:
            self.assertLen(batch[0], 8)
            self.assertIsInstance(batch[0][0], str)
        self.assertEqual(list(batch[1].shape), [8, 2])
        self.assertDType(batch[1], "float32")

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_sample_count(self, format):
        directory = self._prepare_directory(num_classes=4, count=15)
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, label_mode=None, format=format
        )
        sample_count = 0
        for batch in dataset:
            sample_count += len(batch)
        self.assertEqual(sample_count, 15)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_multiclass(self, format):
        directory = self._prepare_directory(num_classes=4, count=15)

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, label_mode=None, format=format
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 8)

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, label_mode=None, format=format
        )
        sample_count = 0
        iterator = iter(dataset)
        for batch in dataset:
            sample_count += len(next(iterator))
        self.assertEqual(sample_count, 15)

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, label_mode="int", format=format
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(list(batch[0].shape), [8])
            self.assertEqual(batch[0].dtype.name, "string")
        else:
            self.assertLen(batch[0], 8)
            self.assertIsInstance(batch[0][0], str)
        self.assertEqual(list(batch[1].shape), [8])
        self.assertDType(batch[1], "int32")

        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, label_mode="categorical", format=format
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        if format == "tf" or backend.backend() == "tensorflow":
            self.assertEqual(list(batch[0].shape), [8])
            self.assertEqual(batch[0].dtype.name, "string")
        else:
            self.assertLen(batch[0], 8)
            self.assertIsInstance(batch[0][0], str)
        self.assertEqual(list(batch[1].shape), [8, 4])
        self.assertDType(batch[1], "float32")

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_validation_split(self, format):
        directory = self._prepare_directory(num_classes=2, count=10)
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=10,
            validation_split=0.2,
            subset="training",
            seed=1337,
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertLen(batch[0], 8)
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=10,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            format=format,
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertLen(batch[0], 2)

        (
            train_dataset,
            val_dataset,
        ) = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=10,
            validation_split=0.2,
            subset="both",
            seed=1337,
            format=format,
        )
        batch = next(iter(train_dataset))
        self.assertLen(batch, 2)
        self.assertLen(batch[0], 8)
        batch = next(iter(val_dataset))
        self.assertLen(batch, 2)
        self.assertLen(batch[0], 2)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_manual_labels(self, format):
        directory = self._prepare_directory(num_classes=2, count=2)
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory, batch_size=8, labels=[0, 1], shuffle=False, format=format
        )
        batch = next(iter(dataset))
        self.assertLen(batch, 2)
        self.assertAllClose(batch[1], [0, 1])

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_follow_links(self, format):
        directory = self._prepare_directory(
            num_classes=2, count=25, nested_dirs=True
        )
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=8,
            label_mode=None,
            follow_links=True,
            format=format,
        )
        sample_count = 0
        for batch in dataset:
            sample_count += len(batch)
        self.assertEqual(sample_count, 25)

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_no_files(self, format):
        directory = self._prepare_directory(num_classes=2, count=0)
        with self.assertRaisesRegex(ValueError, "No text files found"):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, format=format
            )

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_errors(self, format):
        directory = self._prepare_directory(num_classes=3, count=5)

        with self.assertRaisesRegex(ValueError, "`labels` argument should be"):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, labels="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`label_mode` argument must be"
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, label_mode="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, 'only pass `class_names` if `labels="inferred"`'
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory,
                labels=[0, 0, 1, 1, 1],
                class_names=["class_0", "class_1", "class_2"],
                format=format,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected the lengths of `labels` to match the number of files",
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, labels=[0, 0, 1, 1], format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`class_names` passed did not match"
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, class_names=["class_0", "wrong_class"], format=format
            )

        with self.assertRaisesRegex(ValueError, "there must be exactly 2"):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, label_mode="binary", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`validation_split` must be between 0 and 1"
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, validation_split=2, format=format
            )

        with self.assertRaisesRegex(
            ValueError,
            '`subset` must be either "training", "validation" or "both"',
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory, validation_split=0.2, subset="other", format=format
            )

        with self.assertRaisesRegex(
            ValueError, "`validation_split` must be set"
        ):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory,
                validation_split=0.0,
                subset="training",
                format=format,
            )

        with self.assertRaisesRegex(ValueError, "must provide a `seed`"):
            _ = text_dataset_utils.text_dataset_from_directory(
                directory,
                validation_split=0.2,
                subset="training",
                format=format,
            )

    @parameterized.named_parameters(
        ("tf", "tf"),
        ("grain", "grain"),
    )
    def test_text_dataset_from_directory_not_batched(self, format):
        directory = self._prepare_directory()
        dataset = text_dataset_utils.text_dataset_from_directory(
            directory,
            batch_size=None,
            label_mode=None,
            follow_links=True,
            format=format,
        )

        sample = next(iter(dataset))
        if format == "tf":
            self.assertEqual(len(sample.shape), 0)
        else:
            self.assertIsInstance(sample, str)

    def test_text_dataset_from_directory_remote(self):
        directory = self._prepare_directory(
            num_classes=2, count=8, nested_dirs=True
        )
        with (
            mock.patch.object(file_utils, "is_remote_path", lambda path: True),
            mock.patch.object(
                file_utils, "gfile", LazyModule("keras_test_unavailable_module")
            ),
        ):
            dataset = text_dataset_utils.text_dataset_from_directory(
                directory,
                batch_size=8,
                follow_links=True,
                format="grain",
            )
            texts, labels = next(iter(dataset))
            self.assertLen(texts, 8)
            self.assertEqual(list(labels.shape), [8])

            dataset = text_dataset_utils.text_dataset_from_directory(
                directory,
                batch_size=8,
                labels=None,
                follow_links=True,
                format="grain",
            )
            texts = next(iter(dataset))
            self.assertLen(texts, 8)
