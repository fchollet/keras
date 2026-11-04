import contextlib

import numpy as np

from keras.src.backend import config
from keras.src.utils.module_utils import jax


def get_test_data(
    train_samples, test_samples, input_shape, num_classes, random_seed=None
):
    """Generates balanced, stratified synthetic test data to train a model on.

    Args:
        train_samples: Integer, how many training samples to generate.
        test_samples: Integer, how many test samples to generate.
        input_shape: Tuple of integers, shape of the inputs.
        num_classes: Integer, number of classes for the data and targets.
        random_seed: Integer, random seed used by Numpy to generate data.

    Returns:
        A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    np.random.seed(random_seed)

    # Total samples
    total_samples = train_samples + test_samples

    # Ensure that we generate a balanced dataset
    samples_per_class = total_samples // num_classes
    y = np.array(
        [i for i in range(num_classes) for _ in range(samples_per_class)],
        dtype=np.int32,
    )

    # Generate extra samples in a deterministic manner
    extra_samples = total_samples - len(y)
    y_extra = np.array(
        [i % num_classes for i in range(extra_samples)], dtype=np.int64
    )
    y = np.concatenate([y, y_extra])

    # Generate data
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    x = np.zeros((total_samples,) + input_shape, dtype=np.float32)
    for i in range(total_samples):
        x[i] = templates[y[i]] + np.random.normal(
            loc=0, scale=1.0, size=input_shape
        )

    # Shuffle the entire dataset to ensure randomness based on seed
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Stratified Shuffle Split
    x_train, y_train, x_test, y_test = [], [], [], []
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        train_count = int(train_samples / num_classes)

        x_train.extend(x[cls_indices[:train_count]])
        y_train.extend(y[cls_indices[:train_count]])

        x_test.extend(x[cls_indices[train_count:]])
        y_test.extend(y[cls_indices[train_count:]])

    # Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Shuffle training and test sets after stratified split
    train_indices = np.arange(len(x_train))
    test_indices = np.arange(len(x_test))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    x_train, y_train = x_train[train_indices], y_train[train_indices]
    x_test, y_test = x_test[test_indices], y_test[test_indices]

    return (x_train, y_train), (x_test, y_test)


def named_product(*args, **kwargs):
    """Utility to generate the cartesian product of parameters values and
    generate a test case names for each combination.

    The result of this function is to be used with the
    `@parameterized.named_parameters` decorator. It is a replacement for
    `@parameterized.product` which adds explicit test case names.

    For example, this code:
    ```
    class NamedExample(parameterized.TestCase):
        @parameterized.named_parameters(
            named_product(
                [
                    {'testcase_name': 'negative', 'x': -1},
                    {'testcase_name': 'positive', 'x': 1},
                    {'testcase_name': 'zero', 'x': 0},
                ],
                numeral_type=[float, int],
            )
        )
        def test_conversion(self, x, numeral_type):
            self.assertEqual(numeral_type(x), x)
    ```
    produces six tests (note that absl will reorder them by name):
    - `NamedExample::test_conversion_negative_float`
    - `NamedExample::test_conversion_positive_float`
    - `NamedExample::test_conversion_zero_float`
    - `NamedExample::test_conversion_negative_int`
    - `NamedExample::test_conversion_positive_int`
    - `NamedExample::test_conversion_zero_int`

    This function is also useful in the case where there is no product to
    generate test case names for one argument:
    ```
    @parameterized.named_parameters(named_product(numeral_type=[float, int]))
    ```

    Args:
        *args: Each positional parameter is a sequence of keyword arg dicts.
            Every test case generated will include exactly one dict from each
            positional parameter. These will then be merged to form an overall
            list of arguments for the test case. Each dict must contain a
            `"testcase_name"` key whose value is combined with others to
            generate the test case name.
        **kwargs: A mapping of parameter names and their possible values.
            Possible values should given as either a list or a tuple. A string
            representation of each value is used to generate the test case name.

    Returns:
        A list of maps for the test parameters combinations to pass to
        `@parameterized.named_parameters`.
    """

    def value_to_str(value):
        if hasattr(value, "__name__"):
            return value.__name__.lower()
        return str(value).lower()

    # Convert the keyword arguments in the same dict format as the args
    all_test_dicts = args + tuple(
        tuple({"testcase_name": value_to_str(v), key: v} for v in values)
        for key, values in kwargs.items()
    )

    # The current list of tests, start with one empty test
    tests = [{}]
    for test_dicts in all_test_dicts:
        new_tests = []
        for test_dict in test_dicts:
            for test in tests:
                # Augment the testcase name by appending
                testcase_name = test.get("testcase_name", "")
                testcase_name += "_" if testcase_name else ""
                testcase_name += test_dict["testcase_name"]
                new_test = test.copy()
                # Augment the test by adding all the parameters
                new_test.update(test_dict)
                new_test["testcase_name"] = testcase_name
                new_tests.append(new_test)
        # Overwrite the list of tests with the product obtained so far
        tests = new_tests

    return tests


@contextlib.contextmanager
def x64_mode_scope(mode):
    """Runs the block with `keras.config.x64_mode()` set to `mode`.

    JAX is the reference for the dtype tests, so when it is installed it is
    put in the matching mode on every backend. Both settings are restored
    afterwards.

    Args:
        mode: `"off"` or `"explicit_x64"`.
    """
    # JAX's flag is set first and restored first, so it always matches the
    # Keras mode when `set_x64_mode` reconciles the two.
    previous_mode = config.x64_mode()
    try:
        with _jax_explicit_x64_scope(mode == "explicit_x64"):
            config.set_x64_mode(mode)
            yield
    finally:
        config.set_x64_mode(previous_mode)


@contextlib.contextmanager
def _jax_explicit_x64_scope(allow):
    """Sets `jax_explicit_x64_dtypes` for the block, if JAX is installed."""
    if not jax.available:
        yield
        return
    previous = jax.config.jax_explicit_x64_dtypes
    jax.config.update("jax_explicit_x64_dtypes", "allow" if allow else "warn")
    try:
        yield
    finally:
        jax.config.update("jax_explicit_x64_dtypes", previous)


def serialized_variable(layer, name):
    """Returns the underlying variable a serialization-spec `name` maps to.

    `save_own_variables`/`load_own_variables` serialize the raw
    `_kernel`/`_embeddings` variables for the `"kernel"`/`"embeddings"` spec
    entries (a quantized layer's `_kernel` holds its codes). All other spec
    names map directly to the attribute of the same name. Returns `None` if
    the attribute does not exist on the layer.
    """
    if name == "kernel":
        return getattr(layer, "_kernel", None)
    if name == "embeddings":
        return getattr(layer, "_embeddings", None)
    return getattr(layer, name, None)


def is_serialized_variable(layer, name):
    """Mirrors the skip conditions used by `save_own_variables`.

    Optional variables (`bias`, `kernel_zero`, `g_idx`, `embeddings_zero` and
    the `reverse_*` variables of `ReversibleEmbedding`) are attributes that
    hold `None` when the layer's configuration omits them, and are only
    serialized when they hold a variable.
    """
    return serialized_variable(layer, name) is not None


def serialized_variable_names(layer):
    """Spec names actually written to the store for `layer`, in spec order."""
    mode = layer.quantization_mode
    return [
        name
        for name in layer.variable_serialization_spec[mode]
        if is_serialized_variable(layer, name)
    ]


def randomize_serialized_variables(layer, seed=1234):
    """Assigns deterministic pseudo-random values to every serialized variable.

    Ensures serialization round-trips are exercised with non-trivial values
    (rather than the layer's initializer defaults) for every quantization mode.
    """
    rng = np.random.RandomState(seed)
    for name in serialized_variable_names(layer):
        variable = serialized_variable(layer, name)
        dtype = str(variable.dtype)
        if "int" in dtype:
            info = np.iinfo(np.dtype(dtype))
            low, high = max(int(info.min), -8), min(int(info.max), 8)
            value = rng.randint(low, high + 1, size=variable.shape)
            value = value.astype(dtype)
        else:
            value = rng.random(variable.shape).astype("float32")
        variable.assign(value)


def positional_store(layer):
    """Builds a store using the positional (`"0"`, `"1"`, ...) layout.

    Reproduces the on-disk layout of `save_own_variables`, which keys
    serialized variables by their integer position in
    `variable_serialization_spec`. Useful to hand-build a loadable store for
    modes whose variables `save_own_variables` cannot emit directly (e.g.
    uncalibrated GPTQ/AWQ layers).
    """
    return {
        str(index): np.asarray(serialized_variable(layer, name))
        for index, name in enumerate(serialized_variable_names(layer))
    }


def assert_serialized_variables_equal(test_case, expected_layer, actual_layer):
    """Asserts two layers hold identical values for all serialized variables."""
    for name in serialized_variable_names(expected_layer):
        test_case.assertAllClose(
            np.asarray(serialized_variable(expected_layer, name)),
            np.asarray(serialized_variable(actual_layer, name)),
            msg=f"Mismatch for serialized variable '{name}'.",
        )
