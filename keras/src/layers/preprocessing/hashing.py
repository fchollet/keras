from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.utils import backend_utils
from keras.src.utils import numerical_utils
from keras.src.utils import tf_utils
from keras.src.utils.module_utils import tensorflow as tf


@keras_export("keras.layers.Hashing")
class Hashing(Layer):
    """A preprocessing layer which hashes and bins categorical features.

    This layer transforms categorical inputs to hashed output. It element-wise
    converts a ints or strings to ints in a fixed range. The stable hash
    function uses `tensorflow::ops::Fingerprint` to produce the same output
    consistently across all platforms.

    This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
    which provides a consistent hashed output across different platforms and is
    stable across invocations, regardless of device and context, by mixing the
    input bits thoroughly.

    If you want to obfuscate the hashed output, you can also pass a random
    `salt` argument in the constructor. In that case, the layer will use the
    [SipHash64](https://github.com/google/highwayhash) hash function, with
    the `salt` value serving as additional input to the hash function.

    **Note:** This layer internally uses TensorFlow. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    **Example (FarmHash64)**

    >>> layer = keras.layers.Hashing(num_bins=3)
    >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
    >>> layer(inp)
    array([[1],
            [0],
            [1],
            [1],
            [2]])>

    **Example (FarmHash64) with a mask value**

    >>> layer = keras.layers.Hashing(num_bins=3, mask_value='')
    >>> inp = [['A'], ['B'], [''], ['C'], ['D']]
    >>> layer(inp)
    array([[1],
            [1],
            [0],
            [2],
            [2]])

    **Example (SipHash64)**

    >>> layer = keras.layers.Hashing(num_bins=3, salt=[133, 137])
    >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
    >>> layer(inp)
    array([[1],
            [2],
            [1],
            [0],
            [2]])

    **Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**

    >>> layer = keras.layers.Hashing(num_bins=3, salt=133)
    >>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
    >>> layer(inp)
    array([[0],
            [0],
            [2],
            [1],
            [0]])

    Args:
        num_bins: Number of hash bins. Note that this includes the `mask_value`
            bin, so the effective number of bins is `(num_bins - 1)`
            if `mask_value` is set.
        mask_value: A value that represents masked inputs, which are mapped to
            index 0. `None` means no mask term will be added and the
            hashing will start at index 0. Defaults to `None`.
        salt: A single unsigned integer or None.
            If passed, the hash function used will be SipHash64,
            with these values used as an additional input
            (known as a "salt" in cryptography).
            These should be non-zero. If `None`, uses the FarmHash64 hash
            function. It also supports tuple/list of 2 unsigned
            integer numbers, see reference paper for details.
            Defaults to `None`.
        output_mode: Specification for the output of the layer. Values can be
            `"int"`, `"one_hot"`, `"multi_hot"`, or
            `"count"` configuring the layer as follows:
            - `"int"`: Return the integer bin indices directly.
            - `"one_hot"`: Encodes each individual element in the input into an
                array the same size as `num_bins`, containing a 1
                at the input's bin index. If the last dimension is size 1,
                will encode on that dimension.
                If the last dimension is not size 1, will append a new
                dimension for the encoded output.
            - `"multi_hot"`: Encodes each sample in the input into a
                single array the same size as `num_bins`,
                containing a 1 for each bin index
                index present in the sample. Treats the last dimension
                as the sample dimension, if input shape is
                `(..., sample_length)`, output shape will be
                `(..., num_tokens)`.
            - `"count"`: As `"multi_hot"`, but the int array contains a count of
                the number of times the bin index appeared in the sample.
            Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
            and `"count"` output modes. Only supported with TensorFlow
            backend. If `True`, returns a `SparseTensor` instead of
            a dense `Tensor`. Defaults to `False`.
        **kwargs: Keyword arguments to construct a layer.

    Input shape:
        A single string, a list of strings, or an `int32` or `int64` tensor
        of shape `(batch_size, ...,)`.

    Output shape:
        An `int32` tensor of shape `(batch_size, ...)`.

    Reference:

    - [SipHash with salt](https://www.131002.net/siphash/siphash.pdf)
    """

    def __init__(
        self,
        num_bins,
        mask_value=None,
        salt=None,
        output_mode="int",
        sparse=False,
        **kwargs,
    ):
        if not tf.available:
            raise ImportError(
                "Layer Hashing requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )

        # By default, output int32 when output_mode='int' and floats otherwise.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = (
                "int64" if output_mode == "int" else backend.floatx()
            )

        super().__init__(**kwargs)

        if num_bins is None or num_bins <= 0:
            raise ValueError(
                "The `num_bins` for `Hashing` cannot be `None` or "
                f"non-positive values. Received: num_bins={num_bins}."
            )

        if output_mode == "int" and (
            self.dtype_policy.name not in ("int32", "int64")
        ):
            raise ValueError(
                'When `output_mode="int"`, `dtype` should be an integer '
                f"type, 'int32' or 'in64'. Received: dtype={kwargs['dtype']}"
            )

        # 'output_mode' must be one of (INT, ONE_HOT, MULTI_HOT, COUNT)
        accepted_output_modes = ("int", "one_hot", "multi_hot", "count")
        if output_mode not in accepted_output_modes:
            raise ValueError(
                "Invalid value for argument `output_mode`. "
                f"Expected one of {accepted_output_modes}. "
                f"Received: output_mode={output_mode}"
            )

        if sparse and output_mode == "int":
            raise ValueError(
                "`sparse` may only be true if `output_mode` is "
                '`"one_hot"`, `"multi_hot"`, or `"count"`. '
                f"Received: sparse={sparse} and "
                f"output_mode={output_mode}"
            )

        self.num_bins = num_bins
        self.mask_value = mask_value
        self.strong_hash = True if salt is not None else False
        self.output_mode = output_mode
        self.sparse = sparse
        self.salt = None
        if salt is not None:
            if isinstance(salt, (tuple, list)) and len(salt) == 2:
                self.salt = list(salt)
            elif isinstance(salt, int):
                self.salt = [salt, salt]
            else:
                raise ValueError(
                    "The `salt` argument for `Hashing` can only be a tuple of "
                    "size 2 integers, or a single integer. "
                    f"Received: salt={salt}."
                )
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.supports_jit = False

    def compute_output_shape(self, input_shape):
        if self.output_mode == "int":
            return tuple(input_shape)
        # `one_hot` adds a `num_bins` axis, unless the last axis is statically
        # 1, in which case it replaces that axis. `multi_hot` and `count`
        # encode the last (sample) axis into a `num_bins`-sized axis.
        if (
            self.output_mode == "one_hot"
            and input_shape
            and (not isinstance(input_shape[-1], int) or input_shape[-1] != 1)
        ):
            return tuple(input_shape) + (self.num_bins,)
        if len(input_shape) == 0:
            return (self.num_bins,)
        return tuple(input_shape[:-1]) + (self.num_bins,)

    def call(self, inputs):
        from keras.src.backend import tensorflow as tf_backend

        with tf.device("CPU:0"):
            inputs = tf_utils.ensure_tensor(inputs)
            if isinstance(inputs, tf.SparseTensor):
                indices = tf.SparseTensor(
                    indices=inputs.indices,
                    values=self._hash_values_to_bins(inputs.values),
                    dense_shape=inputs.dense_shape,
                )
            else:
                indices = self._hash_values_to_bins(inputs)
            outputs = numerical_utils.encode_categorical_inputs(
                indices,
                output_mode=self.output_mode,
                depth=self.num_bins,
                sparse=self.sparse,
                dtype=self.dtype,
                backend_module=tf_backend,
            )
        return backend_utils.convert_tf_tensor(outputs)

    def _hash_values_to_bins(self, values):
        """Converts a non-sparse tensor of values to bin indices.
        Must be called inside a `with tf.device("CPU:0")` context (as done
        in `call`) since all TF string ops produce CPU-resident tensors.
        """
        hash_bins = self.num_bins
        mask = None
        # If mask_value is set, the zeroth bin is reserved for it.
        if self.mask_value is not None and hash_bins > 1:
            hash_bins -= 1
            mask = tf.equal(values, self.mask_value)
        # Convert all values to strings before hashing.
        # Floats are first normalized to int64.
        if values.dtype.is_floating:
            values = tf.cast(values, dtype="int64")
        if values.dtype != tf.string:
            values = tf.as_string(values)
        # Hash the strings.
        if self.strong_hash:
            values = tf.strings.to_hash_bucket_strong(
                values, hash_bins, name="hash", key=self.salt
            )
        else:
            values = tf.strings.to_hash_bucket_fast(
                values, hash_bins, name="hash"
            )
        if mask is not None:
            values = tf.add(values, tf.ones_like(values))
            values = tf.where(mask, tf.zeros_like(values), values)
        return values

    def compute_output_spec(self, inputs):
        return backend.KerasTensor(
            shape=self.compute_output_shape(inputs.shape),
            dtype=self.dtype,
            sparse=self.sparse,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_bins": self.num_bins,
                "salt": self.salt,
                "mask_value": self.mask_value,
                "output_mode": self.output_mode,
                "sparse": self.sparse,
            }
        )
        return config
