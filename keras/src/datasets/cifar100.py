"""CIFAR100 small images classification dataset."""

import os

import numpy as np

from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.datasets.cifar import load_batch
from keras.src.utils.file_utils import get_file


@keras_export("keras.datasets.cifar100.load_data")
def load_data(label_mode="fine"):
    """Loads the CIFAR100 dataset.

    This is a dataset of 50,000 32x32 color training images and
    10,000 test images, labeled over 100 fine-grained classes that are
    grouped into 20 coarse-grained classes. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Args:
        label_mode: one of `"fine"`, `"coarse"`.
            If it is `"fine"`, the category labels
            are the fine-grained labels, and if it is `"coarse"`,
            the output labels are the coarse-grained superclasses.

    Returns:
        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **`x_train`**: `uint8` NumPy array of grayscale image data with shapes
      `(50000, 32, 32, 3)`, containing the training data. Pixel values range
      from 0 to 255.

    **`y_train`**: `uint8` NumPy array of labels (integers in range 0-99)
      with shape `(50000, 1)` for the training data.

    **`x_test`**: `uint8` NumPy array of grayscale image data with shapes
      `(10000, 32, 32, 3)`, containing the test data. Pixel values range
      from 0 to 255.

    **`y_test`**: `uint8` NumPy array of labels (integers in range 0-99)
      with shape `(10000, 1)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    ```
    """
    if label_mode not in ["fine", "coarse"]:
        raise ValueError(
            '`label_mode` must be one of `"fine"`, `"coarse"`. '
            f"Received: label_mode={label_mode}."
        )

    dirname = "cifar-100-python-target"
    origin = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        "cifar-100-python.tar.gz"
    )
    path = get_file(
        fname=dirname,
        origin=origin,
        extract=True,
        file_hash=(  # noqa: E501
            "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
        ),
    )

    path = os.path.join(path, "cifar-100-python")
    fpath = os.path.join(path, "train")
    x_train, y_train = load_batch(fpath, label_key=f"{label_mode}_labels")

    fpath = os.path.join(path, "test")
    x_test, y_test = load_batch(fpath, label_key=f"{label_mode}_labels")

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
