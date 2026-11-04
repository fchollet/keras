from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import is_int_dtype


@keras_export("keras.random.normal")
def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw random samples from a normal (Gaussian) distribution.

    Args:
        shape: The shape of the random values to generate.
        mean: Float, defaults to 0. Mean of the random values to generate.
        stddev: Float, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`).
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value `seed=None`
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


@keras_export("keras.random.categorical")
def categorical(logits, num_samples, dtype="int32", seed=None):
    """Draws samples from a categorical distribution.

    This function takes as input `logits`, a 2-D input tensor with shape
    (batch_size, num_classes). Each row of the input represents a categorical
    distribution, with each column index containing the log-probability for a
    given class.

    The function will output a 2-D tensor with shape (batch_size, num_samples),
    where each row contains samples from the corresponding row in `logits`.
    Each column index contains an independent samples drawn from the input
    distribution.

    Args:
        logits: 2-D Tensor with shape (batch_size, num_classes). Each row
            should define a categorical distribution with the unnormalized
            log-probabilities for all classes.
        num_samples: Int, the number of independent samples to draw for each
            row of the input. This will be the second dimension of the output
            tensor's shape.
        dtype: Optional dtype of the output tensor.
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.

    Returns:
        A 2-D tensor with (batch_size, num_samples).
    """
    logits_shape = list(backend.ops.convert_to_tensor(logits).shape)
    if len(logits_shape) != 2:
        raise ValueError(
            "`logits` should be a 2-D tensor with shape "
            f"[batch_size, num_classes]. Received: logits={logits}"
        )
    return backend.random.categorical(
        logits, num_samples, dtype=dtype, seed=seed
    )


@keras_export("keras.random.uniform")
def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Draw samples from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    `dtype` must be a floating point type, the default range is `[0, 1)`.

    Args:
        shape: The shape of the random values to generate.
        minval: Float, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Float, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`)
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    if dtype and not is_float_dtype(dtype):
        raise ValueError(
            "`keras.random.uniform` requires a floating point `dtype`. "
            f"Received: dtype={dtype} "
        )
    return backend.random.uniform(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )


@keras_export("keras.random.randint")
def randint(shape, minval, maxval, dtype="int32", seed=None):
    """Draw random integers from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range,
    while the upper bound `maxval` is excluded.

    `dtype` must be an integer type.

    Args:
        shape: The shape of the random values to generate.
        minval: Float, defaults to 0. Lower bound of the range of
            random values to generate (inclusive).
        maxval: Float, defaults to 1. Upper bound of the range of
            random values to generate (exclusive).
        dtype: Optional dtype of the tensor. Only integer types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`)
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    if dtype and not is_int_dtype(dtype):
        raise ValueError(
            "`keras.random.randint` requires an integer `dtype`. "
            f"Received: dtype={dtype} "
        )
    return backend.random.randint(
        shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed
    )


@keras_export("keras.random.truncated_normal")
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    """Draw samples from a truncated normal distribution.

    The values are drawn from a normal distribution with specified mean and
    standard deviation, discarding and re-drawing any samples that are more
    than two standard deviations from the mean.

    Args:
        shape: The shape of the random values to generate.
        mean: Float, defaults to 0. Mean of the random values to generate.
        stddev: Float, defaults to 1. Standard deviation of the random values
            to generate.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`)
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.truncated_normal(
        shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed
    )


@keras_export("keras.random.dropout")
def dropout(inputs, rate, noise_shape=None, seed=None):
    return backend.random.dropout(
        inputs, rate, noise_shape=noise_shape, seed=seed
    )


@keras_export("keras.random.shuffle")
def shuffle(x, axis=0, seed=None):
    """Shuffle the elements of a tensor uniformly at random along an axis.

    Args:
        x: The tensor to be shuffled.
        axis: An integer specifying the axis along which to shuffle. Defaults to
            `0`.
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.shuffle(x, axis=axis, seed=seed)


@keras_export("keras.random.gamma")
def gamma(shape, alpha, dtype=None, seed=None):
    """Draw random samples from the Gamma distribution.

    Args:
        shape: The shape of the random values to generate.
        alpha: Float, the parameter of the distribution.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`).
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.gamma(shape, alpha=alpha, dtype=dtype, seed=seed)


@keras_export("keras.random.binomial")
def binomial(shape, counts, probabilities, dtype=None, seed=None):
    """Draw samples from a Binomial distribution.

    The values are drawn from a Binomial distribution with
    specified trial count and probability of success.

    Args:
        shape: The shape of the random values to generate.
        counts: A number or array of numbers representing the
            number of trials. It must be broadcastable with `probabilities`.
        probabilities: A float or array of floats representing the
            probability of success of an individual event.
            It must be broadcastable with `counts`.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`).
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.binomial(
        shape,
        counts=counts,
        probabilities=probabilities,
        dtype=dtype,
        seed=seed,
    )


@keras_export("keras.random.beta")
def beta(shape, alpha, beta, dtype=None, seed=None):
    """Draw samples from a Beta distribution.

    The values are drawn from a Beta distribution parametrized
    by alpha and beta.

    Args:
        shape: The shape of the random values to generate.
        alpha: Float or an array of floats representing the first
            parameter alpha. Must be broadcastable with `beta` and `shape`.
        beta: Float or an array of floats representing the second
            parameter beta. Must be broadcastable with `alpha` and `shape`.
        dtype: Optional dtype of the tensor. Only floating point types are
            supported. If not specified, `keras.config.floatx()` is used,
            which defaults to `float32` unless you configured it otherwise (via
            `keras.config.set_floatx(float_dtype)`).
        seed: Optional Python integer or instance of
           `keras.random.SeedGenerator`.
            By default, the `seed` argument is `None`, and an internal global
            `keras.random.SeedGenerator` is used. The `seed` argument can be
            used to ensure deterministic (repeatable) random number generation.
            Note that passing an integer as the `seed` value will produce the
            same random values for each call. To generate different random
            values for repeated calls, an instance of
            `keras.random.SeedGenerator` must be provided as the `seed` value.
            Remark concerning the JAX backend: When tracing functions with the
            JAX backend the global `keras.random.SeedGenerator` is not
            supported. Therefore, during tracing the default value seed=None
            will produce an error, and a `seed` argument must be provided.
    """
    return backend.random.beta(
        shape=shape, alpha=alpha, beta=beta, dtype=dtype, seed=seed
    )
