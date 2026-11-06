"""Fashion-MNIST dataset."""

import gzip
import os

import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils.file_utils import get_file

# SHA-256 of each archive, so a corrupted transfer or a planted cache
# entry is rejected instead of being read as training data.
FILE_HASHES = {
    "train-labels-idx1-ubyte.gz": (
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"
    ),
    "train-images-idx3-ubyte.gz": (
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5"
    ),
    "t10k-images-idx3-ubyte.gz": (
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"
    ),
}


@keras_export("keras.datasets.fashion_mnist.load_data")
def load_data():
    """Loads the Fashion-MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
    along with a test set of 10,000 images. This dataset can be used as
    a drop-in replacement for MNIST.

    The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | T-shirt/top |
    |   1   | Trouser     |
    |   2   | Pullover    |
    |   3   | Dress       |
    |   4   | Coat        |
    |   5   | Sandal      |
    |   6   | Shirt       |
    |   7   | Sneaker     |
    |   8   | Bag         |
    |   9   | Ankle boot  |

    Returns:

    Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **`x_train`**: `uint8` NumPy array of grayscale image data with shapes
      `(60000, 28, 28)`, containing the training data.

    **`y_train`**: `uint8` NumPy array of labels (integers in range 0-9)
      with shape `(60000,)` for the training data.

    **`x_test`**: `uint8` NumPy array of grayscale image data with shapes
      (10000, 28, 28), containing the test data.

    **`y_test`**: `uint8` NumPy array of labels (integers in range 0-9)
      with shape `(10000,)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    ```

    License:

    The copyright for Fashion-MNIST is held by Zalando SE.
    Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
    """
    dirname = os.path.join("datasets", "fashion-mnist")
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "train-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    paths = []
    for fname in files:
        paths.append(
            get_file(
                fname,
                origin=base + fname,
                cache_subdir=dirname,
                file_hash=FILE_HASHES[fname],
            )
        )

    with gzip.open(paths[0], "rb") as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )

    with gzip.open(paths[2], "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_test), 28, 28
        )

    return (x_train, y_train), (x_test, y_test)
