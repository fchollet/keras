from keras.src.api_export import keras_export
from keras.src.layers.activations.activation import Activation
from keras.src.layers.activations.elu import ELU
from keras.src.layers.activations.leaky_relu import LeakyReLU
from keras.src.layers.activations.prelu import PReLU
from keras.src.layers.activations.relu import ReLU
from keras.src.layers.activations.softmax import Softmax
from keras.src.layers.attention.additive_attention import AdditiveAttention
from keras.src.layers.attention.attention import Attention
from keras.src.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)
from keras.src.layers.attention.multi_head_attention import MultiHeadAttention
from keras.src.layers.convolutional.conv1d import Conv1D
from keras.src.layers.convolutional.conv1d_transpose import Conv1DTranspose
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras.src.layers.convolutional.conv3d import Conv3D
from keras.src.layers.convolutional.conv3d_transpose import Conv3DTranspose
from keras.src.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from keras.src.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from keras.src.layers.convolutional.separable_conv1d import SeparableConv1D
from keras.src.layers.convolutional.separable_conv2d import SeparableConv2D
from keras.src.layers.core.dense import Dense
from keras.src.layers.core.einsum_dense import EinsumDense
from keras.src.layers.core.embedding import Embedding
from keras.src.layers.core.identity import Identity
from keras.src.layers.core.input_layer import Input
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.core.lambda_layer import Lambda
from keras.src.layers.core.masking import Masking
from keras.src.layers.core.reversible_embedding import ReversibleEmbedding
from keras.src.layers.core.ternary_dense import TernaryDense
from keras.src.layers.core.wrapper import Wrapper
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.merging.add import Add
from keras.src.layers.merging.add import add
from keras.src.layers.merging.average import Average
from keras.src.layers.merging.average import average
from keras.src.layers.merging.concatenate import Concatenate
from keras.src.layers.merging.concatenate import concatenate
from keras.src.layers.merging.dot import Dot
from keras.src.layers.merging.dot import dot
from keras.src.layers.merging.maximum import Maximum
from keras.src.layers.merging.maximum import maximum
from keras.src.layers.merging.minimum import Minimum
from keras.src.layers.merging.minimum import minimum
from keras.src.layers.merging.multiply import Multiply
from keras.src.layers.merging.multiply import multiply
from keras.src.layers.merging.subtract import Subtract
from keras.src.layers.merging.subtract import subtract
from keras.src.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from keras.src.layers.normalization.group_normalization import (
    GroupNormalization,
)
from keras.src.layers.normalization.layer_normalization import (
    LayerNormalization,
)
from keras.src.layers.normalization.rms_normalization import RMSNormalization
from keras.src.layers.normalization.spectral_normalization import (
    SpectralNormalization,
)
from keras.src.layers.normalization.unit_normalization import UnitNormalization
from keras.src.layers.pooling.adaptive_average_pooling1d import (
    AdaptiveAveragePooling1D,
)
from keras.src.layers.pooling.adaptive_average_pooling2d import (
    AdaptiveAveragePooling2D,
)
from keras.src.layers.pooling.adaptive_average_pooling3d import (
    AdaptiveAveragePooling3D,
)
from keras.src.layers.pooling.adaptive_max_pooling1d import AdaptiveMaxPooling1D
from keras.src.layers.pooling.adaptive_max_pooling2d import AdaptiveMaxPooling2D
from keras.src.layers.pooling.adaptive_max_pooling3d import AdaptiveMaxPooling3D
from keras.src.layers.pooling.average_pooling1d import AveragePooling1D
from keras.src.layers.pooling.average_pooling2d import AveragePooling2D
from keras.src.layers.pooling.average_pooling3d import AveragePooling3D
from keras.src.layers.pooling.global_average_pooling1d import (
    GlobalAveragePooling1D,
)
from keras.src.layers.pooling.global_average_pooling2d import (
    GlobalAveragePooling2D,
)
from keras.src.layers.pooling.global_average_pooling3d import (
    GlobalAveragePooling3D,
)
from keras.src.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from keras.src.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from keras.src.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D
from keras.src.layers.pooling.max_pooling1d import MaxPooling1D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.pooling.max_pooling3d import MaxPooling3D
from keras.src.layers.preprocessing.category_encoding import CategoryEncoding
from keras.src.layers.preprocessing.discretization import Discretization
from keras.src.layers.preprocessing.hashed_crossing import HashedCrossing
from keras.src.layers.preprocessing.hashing import Hashing
from keras.src.layers.preprocessing.image_preprocessing.aug_mix import AugMix
from keras.src.layers.preprocessing.image_preprocessing.auto_contrast import (
    AutoContrast,
)
from keras.src.layers.preprocessing.image_preprocessing.center_crop import (
    CenterCrop,
)
from keras.src.layers.preprocessing.image_preprocessing.clahe import (
    ContrastLimitedAdaptiveHistogramEqualization,
)
from keras.src.layers.preprocessing.image_preprocessing.cut_mix import CutMix
from keras.src.layers.preprocessing.image_preprocessing.equalization import (
    Equalization,
)
from keras.src.layers.preprocessing.image_preprocessing.max_num_bounding_box import (
    MaxNumBoundingBoxes,
)
from keras.src.layers.preprocessing.image_preprocessing.mix_up import MixUp
from keras.src.layers.preprocessing.image_preprocessing.mosaic import Mosaic
from keras.src.layers.preprocessing.image_preprocessing.rand_augment import (
    RandAugment,
)
from keras.src.layers.preprocessing.image_preprocessing.random_brightness import (
    RandomBrightness,
)
from keras.src.layers.preprocessing.image_preprocessing.random_color_degeneration import (
    RandomColorDegeneration,
)
from keras.src.layers.preprocessing.image_preprocessing.random_color_jitter import (
    RandomColorJitter,
)
from keras.src.layers.preprocessing.image_preprocessing.random_contrast import (
    RandomContrast,
)
from keras.src.layers.preprocessing.image_preprocessing.random_crop import (
    RandomCrop,
)
from keras.src.layers.preprocessing.image_preprocessing.random_elastic_transform import (
    RandomElasticTransform,
)
from keras.src.layers.preprocessing.image_preprocessing.random_erasing import (
    RandomErasing,
)
from keras.src.layers.preprocessing.image_preprocessing.random_flip import (
    RandomFlip,
)
from keras.src.layers.preprocessing.image_preprocessing.random_gaussian_blur import (
    RandomGaussianBlur,
)
from keras.src.layers.preprocessing.image_preprocessing.random_grayscale import (
    RandomGrayscale,
)
from keras.src.layers.preprocessing.image_preprocessing.random_hue import (
    RandomHue,
)
from keras.src.layers.preprocessing.image_preprocessing.random_invert import (
    RandomInvert,
)
from keras.src.layers.preprocessing.image_preprocessing.random_perspective import (
    RandomPerspective,
)
from keras.src.layers.preprocessing.image_preprocessing.random_posterization import (
    RandomPosterization,
)
from keras.src.layers.preprocessing.image_preprocessing.random_rotation import (
    RandomRotation,
)
from keras.src.layers.preprocessing.image_preprocessing.random_saturation import (
    RandomSaturation,
)
from keras.src.layers.preprocessing.image_preprocessing.random_sharpness import (
    RandomSharpness,
)
from keras.src.layers.preprocessing.image_preprocessing.random_shear import (
    RandomShear,
)
from keras.src.layers.preprocessing.image_preprocessing.random_translation import (
    RandomTranslation,
)
from keras.src.layers.preprocessing.image_preprocessing.random_zoom import (
    RandomZoom,
)
from keras.src.layers.preprocessing.image_preprocessing.resizing import Resizing
from keras.src.layers.preprocessing.image_preprocessing.solarization import (
    Solarization,
)
from keras.src.layers.preprocessing.index_lookup import IndexLookup
from keras.src.layers.preprocessing.integer_lookup import IntegerLookup
from keras.src.layers.preprocessing.mel_spectrogram import MelSpectrogram
from keras.src.layers.preprocessing.normalization import Normalization
from keras.src.layers.preprocessing.pipeline import Pipeline
from keras.src.layers.preprocessing.rescaling import Rescaling
from keras.src.layers.preprocessing.stft_spectrogram import STFTSpectrogram
from keras.src.layers.preprocessing.string_lookup import StringLookup
from keras.src.layers.preprocessing.text_vectorization import TextVectorization
from keras.src.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from keras.src.layers.regularization.alpha_dropout import AlphaDropout
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.regularization.gaussian_dropout import GaussianDropout
from keras.src.layers.regularization.gaussian_noise import GaussianNoise
from keras.src.layers.regularization.spatial_dropout import SpatialDropout1D
from keras.src.layers.regularization.spatial_dropout import SpatialDropout2D
from keras.src.layers.regularization.spatial_dropout import SpatialDropout3D
from keras.src.layers.reshaping.cropping1d import Cropping1D
from keras.src.layers.reshaping.cropping2d import Cropping2D
from keras.src.layers.reshaping.cropping3d import Cropping3D
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.reshaping.permute import Permute
from keras.src.layers.reshaping.repeat_vector import RepeatVector
from keras.src.layers.reshaping.reshape import Reshape
from keras.src.layers.reshaping.up_sampling1d import UpSampling1D
from keras.src.layers.reshaping.up_sampling2d import UpSampling2D
from keras.src.layers.reshaping.up_sampling3d import UpSampling3D
from keras.src.layers.reshaping.zero_padding1d import ZeroPadding1D
from keras.src.layers.reshaping.zero_padding2d import ZeroPadding2D
from keras.src.layers.reshaping.zero_padding3d import ZeroPadding3D
from keras.src.layers.rnn.bidirectional import Bidirectional
from keras.src.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras.src.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras.src.layers.rnn.conv_lstm3d import ConvLSTM3D
from keras.src.layers.rnn.gru import GRU
from keras.src.layers.rnn.gru import GRUCell
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.rnn.lstm import LSTMCell
from keras.src.layers.rnn.rnn import RNN
from keras.src.layers.rnn.simple_rnn import SimpleRNN
from keras.src.layers.rnn.simple_rnn import SimpleRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.layers.rnn.time_distributed import TimeDistributed
from keras.src.saving import serialization_lib


@keras_export("keras.layers.serialize")
def serialize(layer):
    """Returns the layer configuration as a Python dict.

    Args:
        layer: A `keras.layers.Layer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the layer.
    """
    return serialization_lib.serialize_keras_object(layer)


@keras_export("keras.layers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras layer object via its configuration.

    Args:
        config: A python dict containing a serialized layer configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras layer instance.
    """
    obj = serialization_lib.deserialize_keras_object(
        config,
        custom_objects=custom_objects,
    )
    if not isinstance(obj, Layer):
        raise ValueError(
            "`keras.layers.deserialize` was passed a `config` object that is "
            f"not a `keras.layers.Layer`. Received: {config}"
        )
    return obj
