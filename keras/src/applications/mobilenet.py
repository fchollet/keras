import warnings

from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/"
)
# SHA-256 of each ImageNet checkpoint, keyed by "<alpha_text>_<rows>", as
# `(weights, weights_no_top)`.
WEIGHTS_HASHES = {
    "1_0_128": (
        "ef1480dcb7f11a75bc5187a74863b3b97c4cfed6b6dd5c78f003ba002a4d7fe6",
        "8fbb8332b956430de61ad4fa1ab261ab5b7922ef1d0f38c30056a485ec0d4ecb",
    ),
    "1_0_160": (
        "a41aa0195ebe0a5301ab90af589155b42671f78704fd9a2e7dda919c478cd479",
        "22d25b67a8936869b5f0d0c840ba150c90dc89c0b07f8aacd387614be67845d2",
    ),
    "1_0_192": (
        "bd7e7f6a7cd6690a10aedba3e93bec811240418bd9179546e50c0f4c35899a01",
        "d67556231564ded95aaa7a44004c73aa9696fba267a6817f430eee097af11782",
    ),
    "1_0_224": (
        "006a2d9484fbea625dd4e6518652c101a3eca052671671c56265fc84f2e1181f",
        "9f9cfd8c925431c06ce96744e1d201575f6a681bf38ae2de38e2ae8f7d7829d3",
    ),
    "7_5_128": (
        "cafcb7455c36187c93eb7fb3a68e29f76b5761cb2d946d57d1967578e3b05361",
        "dcc5aaa418bbf397a431e79a52be0ad1a61fb6364a9bb6d68b5258f13669c8ba",
    ),
    "7_5_160": (
        "3ee6f1daa0f9da149bec3311eacfb4451ca236bc126dc25ee191f5bdc6e641be",
        "6056d87ad4e3b341cc8272d68b54a8a113ef0b0e5ab1399e4a801d2942a064fa",
    ),
    "7_5_192": (
        "a7a2364ee404c19301edb827de693f46eeb46aad13e2c5e73494545e57749eef",
        "b8ea4352b99a87552c0b84c59b59d146b823b684766026ed8f0c68d3aab9e677",
    ),
    "7_5_224": (
        "811e0307a7e8895a180b81c191033482d99470a0604ea8e88e8d947005067d84",
        "1da9019b66f05a0256dedfed0f321fcaf8bfa71f9facf3f9117e62820be39aa5",
    ),
    "5_0_128": (
        "26752cdb9ab8459c85fb4cd0f45475bbd3bd25f9225b87bbedd474f5976da145",
        "7f9871530e19990a9a180626566fcc32fb487c70880306592cae219d67179635",
    ),
    "5_0_160": (
        "71c742137645f19426ea3540bb2933310b0a5f2d3a2d09ee1aea91e348f6fba7",
        "7b15a4d22079138dd889a78cdea8d921a3b73cf7d313ff12f23634c4f71bcace",
    ),
    "5_0_192": (
        "fabfe055bd26ea40c1c0798c4e22fe1eb437feab111be9f1f722360dff81e040",
        "b848a83f140e284973d80805dc4c184a0a885dadcc1700630a86879c9103d5b1",
    ),
    "5_0_224": (
        "b4b134ed41de14fd8d6717ab4e88b53e3113194acae392bc41c012ba463906bb",
        "df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7",
    ),
    "2_5_128": (
        "fe295db7f78635f1403127e3ba8cdc6237f83262bdb745063a67c1d81190c7f8",
        "7d8b7f3c9555cb4579f9659ee233a540e66fd7ad514c0a8b14493b667447c5d6",
    ),
    "2_5_160": (
        "74b32700eb3f6fb8d0b3f416c3fb52ab5eca35dc82393341eb7bb52676b6b003",
        "c2f18432229c945b816f8b9f9cfa40d2964454b033efea804f6cd03942865bd4",
    ),
    "2_5_192": (
        "22a302e8a39e545c7372ca96fe8698e3cbbc7d3906f60316e0617c15616f8894",
        "4b7370b9880577f12f31942049d19610a4559c8af38f39dc67d1f90ec9663aa1",
    ),
    "2_5_224": (
        "98d047950e59f478af63681720906940efc8ce8cecb95bce5c7b47394f8bd63d",
        "dbdb03ee2a22fd895301636cd328b234bb3a9952358f436d82f46b81e0d5b0bf",
    ),
}


@keras_export(
    [
        "keras.applications.mobilenet.MobileNet",
        "keras.applications.MobileNet",
    ]
)
def MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name=None,
):
    """Instantiates the MobileNet architecture.

    Reference:
    - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)

    This function returns a Keras image classification model,
    optionally loaded with weights pre-trained on ImageNet.

    For image classification use cases, see
    [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNet, call `keras.applications.mobilenet.preprocess_input`
    on your inputs before passing them to the model.
    `mobilenet.preprocess_input` will scale input pixels between -1 and 1.

    Args:
        input_shape: Optional shape tuple, only to be specified if `include_top`
            is `False` (otherwise the input shape has to be `(224, 224, 3)`
            (with `"channels_last"` data format) or `(3, 224, 224)`
            (with `"channels_first"` data format).
            It should have exactly 3 inputs channels, and width and
            height should be no smaller than 32. E.g. `(200, 200, 3)` would
            be one valid value. Defaults to `None`.
            `input_shape` will be ignored if the `input_tensor` is provided.
        alpha: Controls the width of the network. This is known as the width
            multiplier in the MobileNet paper.
            When `weights` is `"imagenet"`, `alpha` can be one of `0.25`,
            `0.50`, `0.75` or `1.0` only.
            - If `alpha < 1.0`, proportionally decreases the number
                of filters in each layer.
            - If `alpha > 1.0`, proportionally increases the number
                of filters in each layer.
            - If `alpha == 1`, default number of filters from the paper
                are used at each layer. Defaults to `1.0`.
        depth_multiplier: Depth multiplier for depthwise convolution.
            It sets the number of depthwise filters for each input channel.
            When `weights` is `"imagenet"`, `depth_multiplier` must be `1`.
            Defaults to `1`.
        dropout: Dropout rate. Defaults to `0.001`.
        include_top: Boolean, whether to include the fully-connected layer
            at the top of the network. Defaults to `True`.
        weights: One of `None` (random initialization), `"imagenet"`
            (pre-training on ImageNet), or the path to the weights file
            to be loaded. Defaults to `"imagenet"`.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model. `input_tensor` is useful
            for sharing inputs between multiple different networks.
            Defaults to `None`.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` (default) means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: Optional number of classes to classify images into,
            only to be specified if `include_top` is `True`, and if
            no `weights` argument is specified. Defaults to `1000`.
        classifier_activation: A `str` or callable. The activation function
            to use on the "top" layer. Ignored unless `include_top=True`.
            Set `classifier_activation=None` to return the logits of the "top"
            layer. When loading pretrained weights, `classifier_activation`
            can only be `None` or `"softmax"`.
        name: String, the name of the model.

    Returns:
        A model instance.
    """
    if not (weights in {"imagenet", None} or file_utils.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), 'imagenet' "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded. "
            f"Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            "If using `weights='imagenet'` with `include_top=True`, "
            "`classes` should be 1000.  "
            f"Received classes={classes}"
        )

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if depth_multiplier != 1:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "depth multiplier must be 1.  "
                f"Received depth_multiplier={depth_multiplier}"
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of "
                "`0.25`, `0.50`, `0.75` or `1.0` only.  "
                f"Received alpha={alpha}"
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            warnings.warn(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not in [128, 160, 192, 224]. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default.",
                stacklevel=2,
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2
    )
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4
    )
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6
    )
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12
    )
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
        x = layers.Reshape((classes,), name="reshape_2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(
            activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if name is None:
        name = f"mobilenet_{alpha:0.2f}_{rows}"
    model = Functional(inputs, x, name=name)

    # Load weights.
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        weights_hashes = WEIGHTS_HASHES[f"{alpha_text}_{rows}"]
        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            file_hash = weights_hashes[0]
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            file_hash = weights_hashes[1]
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = file_utils.get_file(
            model_name,
            weight_path,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).

    Args:
        inputs: Input tensor of shape `(rows, cols, 3)` (with `channels_last`
            data format) or (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels, and width and height
            should be no smaller than 32. E.g. `(224, 224, 3)` would be
            one valid value.
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the convolution).
        alpha: controls the width of the network. - If `alpha` < 1.0,
            proportionally decreases the number of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number of filters
                in each layer.
            - If `alpha` = 1, default number of filters from the paper are
                used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the width
            and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for all
            spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate`
            value != 1.

    Input shape:
        4D tensor with shape: `(samples, channels, rows, cols)` if
            data_format='channels_first'
        or 4D tensor with shape: `(samples, rows, cols, channels)` if
            data_format='channels_last'. # Output shape
        4D tensor with shape: `(samples, filters, new_rows, new_cols)`
            if data_format='channels_first'
        or 4D tensor with shape: `(samples, new_rows, new_cols, filters)`
            if data_format='channels_last'. `rows` and `cols` values
            might have changed due to stride.

    Returns:
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = int(filters * alpha)
    x = layers.Conv2D(
        filters,
        kernel,
        padding="same",
        use_bias=False,
        strides=strides,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name="conv1_bn")(x)
    return layers.ReLU(6.0, name="conv1_relu")(x)


def _depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    Args:
        inputs: Input tensor of shape `(rows, cols, channels)` (with
            `channels_last` data format) or (channels, rows, cols) (with
            `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network. - If `alpha` < 1.0,
            proportionally decreases the number of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number of filters
                in each layer.
            - If `alpha` = 1, default number of filters from the paper are
                used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel. The total number of depthwise convolution
            output channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    Input shape:
        4D tensor with shape: `(batch, channels, rows, cols)` if
            data_format='channels_first'
        or 4D tensor with shape: `(batch, rows, cols, channels)` if
            data_format='channels_last'. # Output shape
        4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
            data_format='channels_first'
        or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
            data_format='channels_last'. `rows` and `cols` values might have
            changed due to stride.

    Returns:
        Output tensor of block.
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(
            ((0, 1), (0, 1)), name="conv_pad_%d" % block_id
        )(inputs)
    x = layers.DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_dw_%d_bn" % block_id
    )(x)
    x = layers.ReLU(6.0, name="conv_dw_%d_relu" % block_id)(x)

    x = layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_pw_%d_bn" % block_id
    )(x)
    return layers.ReLU(6.0, name="conv_pw_%d_relu" % block_id)(x)


@keras_export("keras.applications.mobilenet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


@keras_export("keras.applications.mobilenet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
