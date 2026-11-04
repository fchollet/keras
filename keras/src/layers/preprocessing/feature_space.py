import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend.common import standardize_dtype
from keras.src.layers.layer import Layer
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
from keras.src.utils.progbar import Progbar


class Cross(KerasSaveable):
    def __init__(self, feature_names, crossing_dim, output_mode="one_hot"):
        if output_mode not in {"int", "one_hot"}:
            raise ValueError(
                "Invalid value for argument `output_mode`. "
                "Expected one of {'int', 'one_hot'}. "
                f"Received: output_mode={output_mode}"
            )
        self.feature_names = tuple(feature_names)
        self.crossing_dim = crossing_dim
        self.output_mode = output_mode

    def _obj_type(self):
        return "Cross"

    @property
    def name(self):
        return "_X_".join(self.feature_names)

    def get_config(self):
        return {
            "feature_names": self.feature_names,
            "crossing_dim": self.crossing_dim,
            "output_mode": self.output_mode,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Feature(KerasSaveable):
    def __init__(self, dtype, preprocessor, output_mode):
        if output_mode not in {"int", "one_hot", "float"}:
            raise ValueError(
                "Invalid value for argument `output_mode`. "
                "Expected one of {'int', 'one_hot', 'float'}. "
                f"Received: output_mode={output_mode}"
            )
        self.dtype = dtype
        if isinstance(preprocessor, dict):
            preprocessor = serialization_lib.deserialize_keras_object(
                preprocessor
            )
        self.preprocessor = preprocessor
        self.output_mode = output_mode

    def _obj_type(self):
        return "Feature"

    def get_config(self):
        return {
            "dtype": self.dtype,
            "preprocessor": serialization_lib.serialize_keras_object(
                self.preprocessor
            ),
            "output_mode": self.output_mode,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras_export("keras.utils.FeatureSpace")
class FeatureSpace(Layer):
    """One-stop utility for preprocessing and encoding structured data.

    Arguments:
        feature_names: Dict mapping the names of your features to their
            type specification, e.g. `{"my_feature": "integer_categorical"}`
            or `{"my_feature": FeatureSpace.integer_categorical()}`.
            For a complete list of all supported types, see
            "Available feature types" paragraph below.
        output_mode: One of `"concat"` or `"dict"`. In concat mode, all
            features get concatenated together into a single vector.
            In dict mode, the FeatureSpace returns a dict of individually
            encoded features (with the same keys as the input dict keys).
        crosses: List of features to be crossed together, e.g.
            `crosses=[("feature_1", "feature_2")]`. The features will be
            "crossed" by hashing their combined value into
            a fixed-length vector.
        crossing_dim: Default vector size for hashing crossed features.
            Defaults to `32`.
        hashing_dim: Default vector size for hashing features of type
            `"integer_hashed"` and `"string_hashed"`. Defaults to `32`.
        num_discretization_bins: Default number of bins to be used for
            discretizing features of type `"float_discretized"`.
            Defaults to `32`.

    **Available feature types:**

    Note that all features can be referred to by their string name,
    e.g. `"integer_categorical"`. When using the string name, the default
    argument values are used.

    ```python
    # Plain float values.
    FeatureSpace.float(name=None)

    # Float values to be preprocessed via featurewise standardization
    # (i.e. via a `keras.layers.Normalization` layer).
    FeatureSpace.float_normalized(name=None)

    # Float values to be preprocessed via linear rescaling
    # (i.e. via a `keras.layers.Rescaling` layer).
    FeatureSpace.float_rescaled(scale=1., offset=0., name=None)

    # Float values to be discretized. By default, the discrete
    # representation will then be one-hot encoded.
    FeatureSpace.float_discretized(
        num_bins, bin_boundaries=None, output_mode="one_hot", name=None)

    # Integer values to be indexed. By default, the discrete
    # representation will then be one-hot encoded.
    FeatureSpace.integer_categorical(
        max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)

    # String values to be indexed. By default, the discrete
    # representation will then be one-hot encoded.
    FeatureSpace.string_categorical(
        max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)

    # Integer values to be hashed into a fixed number of bins.
    # By default, the discrete representation will then be one-hot encoded.
    FeatureSpace.integer_hashed(num_bins, output_mode="one_hot", name=None)

    # String values to be hashed into a fixed number of bins.
    # By default, the discrete representation will then be one-hot encoded.
    FeatureSpace.string_hashed(num_bins, output_mode="one_hot", name=None)
    ```

    Examples:

    **Basic usage with a dict of input data:**

    ```python
    raw_data = {
        "float_values": [0.0, 0.1, 0.2, 0.3],
        "string_values": ["zero", "one", "two", "three"],
        "int_values": [0, 1, 2, 3],
    }
    dataset = tf.data.Dataset.from_tensor_slices(raw_data)

    feature_space = FeatureSpace(
        features={
            "float_values": "float_normalized",
            "string_values": "string_categorical",
            "int_values": "integer_categorical",
        },
        crosses=[("string_values", "int_values")],
        output_mode="concat",
    )
    # Before you start using the FeatureSpace,
    # you must `adapt()` it on some data.
    feature_space.adapt(dataset)

    # You can call the FeatureSpace on a dict of data (batched or unbatched).
    output_vector = feature_space(raw_data)
    ```

    **Basic usage with `tf.data`:**

    ```python
    # Unlabeled data
    preprocessed_ds = unlabeled_dataset.map(feature_space)

    # Labeled data
    preprocessed_ds = labeled_dataset.map(lambda x, y: (feature_space(x), y))
    ```

    **Basic usage with the Keras Functional API:**

    ```python
    # Retrieve a dict Keras Input objects
    inputs = feature_space.get_inputs()
    # Retrieve the corresponding encoded Keras tensors
    encoded_features = feature_space.get_encoded_features()
    # Build a Functional model
    outputs = keras.layers.Dense(1, activation="sigmoid")(encoded_features)
    model = keras.Model(inputs, outputs)
    ```

    **Customizing each feature or feature cross:**

    ```python
    feature_space = FeatureSpace(
        features={
            "float_values": FeatureSpace.float_normalized(),
            "string_values": FeatureSpace.string_categorical(max_tokens=10),
            "int_values": FeatureSpace.integer_categorical(max_tokens=10),
        },
        crosses=[
            FeatureSpace.cross(("string_values", "int_values"), crossing_dim=32)
        ],
        output_mode="concat",
    )
    ```

    **Returning a dict of integer-encoded features:**

    ```python
    feature_space = FeatureSpace(
        features={
            "string_values": FeatureSpace.string_categorical(output_mode="int"),
            "int_values": FeatureSpace.integer_categorical(output_mode="int"),
        },
        crosses=[
            FeatureSpace.cross(
                feature_names=("string_values", "int_values"),
                crossing_dim=32,
                output_mode="int",
            )
        ],
        output_mode="dict",
    )
    ```

    **Specifying your own Keras preprocessing layer:**

    ```python
    # Let's say that one of the features is a short text paragraph that
    # we want to encode as a vector (one vector per paragraph) via TF-IDF.
    data = {
        "text": ["1st string", "2nd string", "3rd string"],
    }

    # There's a Keras layer for this: TextVectorization.
    custom_layer = layers.TextVectorization(output_mode="tf_idf")

    # We can use FeatureSpace.feature to create a custom feature
    # that will use our preprocessing layer.
    feature_space = FeatureSpace(
        features={
            "text": FeatureSpace.feature(
                preprocessor=custom_layer, dtype="string", output_mode="float"
            ),
        },
        output_mode="concat",
    )
    feature_space.adapt(tf.data.Dataset.from_tensor_slices(data))
    output_vector = feature_space(data)
    ```

    **Retrieving the underlying Keras preprocessing layers:**

    ```python
    # The preprocessing layer of each feature is available in `.preprocessors`.
    preprocessing_layer = feature_space.preprocessors["feature1"]

    # The crossing layer of each feature cross is available in `.crossers`.
    # It's an instance of keras.layers.HashedCrossing.
    crossing_layer = feature_space.crossers["feature1_X_feature2"]
    ```

    **Saving and reloading a FeatureSpace:**

    ```python
    feature_space.save("featurespace.keras")
    reloaded_feature_space = keras.models.load_model("featurespace.keras")
    ```
    """

    @classmethod
    def cross(cls, feature_names, crossing_dim, output_mode="one_hot"):
        return Cross(feature_names, crossing_dim, output_mode=output_mode)

    @classmethod
    def feature(cls, dtype, preprocessor, output_mode):
        return Feature(dtype, preprocessor, output_mode)

    @classmethod
    def float(cls, name=None):
        name = name or auto_name("float")
        preprocessor = TFDIdentity(dtype="float32", name=f"{name}_preprocessor")
        return Feature(
            dtype="float32", preprocessor=preprocessor, output_mode="float"
        )

    @classmethod
    def float_rescaled(cls, scale=1.0, offset=0.0, name=None):
        name = name or auto_name("float_rescaled")
        preprocessor = layers.Rescaling(
            scale=scale, offset=offset, name=f"{name}_preprocessor"
        )
        return Feature(
            dtype="float32", preprocessor=preprocessor, output_mode="float"
        )

    @classmethod
    def float_normalized(cls, name=None):
        name = name or auto_name("float_normalized")
        preprocessor = layers.Normalization(
            axis=-1, name=f"{name}_preprocessor"
        )
        return Feature(
            dtype="float32", preprocessor=preprocessor, output_mode="float"
        )

    @classmethod
    def float_discretized(
        cls, num_bins, bin_boundaries=None, output_mode="one_hot", name=None
    ):
        name = name or auto_name("float_discretized")
        preprocessor = layers.Discretization(
            num_bins=num_bins,
            bin_boundaries=bin_boundaries,
            name=f"{name}_preprocessor",
        )
        return Feature(
            dtype="float32", preprocessor=preprocessor, output_mode=output_mode
        )

    @classmethod
    def integer_categorical(
        cls,
        max_tokens=None,
        num_oov_indices=1,
        output_mode="one_hot",
        name=None,
    ):
        name = name or auto_name("integer_categorical")
        preprocessor = layers.IntegerLookup(
            name=f"{name}_preprocessor",
            max_tokens=max_tokens,
            num_oov_indices=num_oov_indices,
        )
        return Feature(
            dtype="int32", preprocessor=preprocessor, output_mode=output_mode
        )

    @classmethod
    def string_categorical(
        cls,
        max_tokens=None,
        num_oov_indices=1,
        output_mode="one_hot",
        name=None,
    ):
        name = name or auto_name("string_categorical")
        preprocessor = layers.StringLookup(
            name=f"{name}_preprocessor",
            max_tokens=max_tokens,
            num_oov_indices=num_oov_indices,
        )
        return Feature(
            dtype="string", preprocessor=preprocessor, output_mode=output_mode
        )

    @classmethod
    def string_hashed(cls, num_bins, output_mode="one_hot", name=None):
        name = name or auto_name("string_hashed")
        preprocessor = layers.Hashing(
            name=f"{name}_preprocessor", num_bins=num_bins
        )
        return Feature(
            dtype="string", preprocessor=preprocessor, output_mode=output_mode
        )

    @classmethod
    def integer_hashed(cls, num_bins, output_mode="one_hot", name=None):
        name = name or auto_name("integer_hashed")
        preprocessor = layers.Hashing(
            name=f"{name}_preprocessor", num_bins=num_bins
        )
        return Feature(
            dtype="int32", preprocessor=preprocessor, output_mode=output_mode
        )

    def __init__(
        self,
        features,
        output_mode="concat",
        crosses=None,
        crossing_dim=32,
        hashing_dim=32,
        num_discretization_bins=32,
        name=None,
    ):
        super().__init__(name=name)
        if not features:
            raise ValueError("The `features` argument cannot be None or empty.")
        self.crossing_dim = crossing_dim
        self.hashing_dim = hashing_dim
        self.num_discretization_bins = num_discretization_bins
        self.features = {
            name: self._standardize_feature(name, value)
            for name, value in features.items()
        }
        self.crosses = []
        if crosses:
            feature_set = set(features.keys())
            for cross in crosses:
                if isinstance(cross, dict):
                    cross = serialization_lib.deserialize_keras_object(cross)
                if isinstance(cross, Cross):
                    self.crosses.append(cross)
                else:
                    if not crossing_dim:
                        raise ValueError(
                            "When specifying `crosses`, the argument "
                            "`crossing_dim` "
                            "(dimensionality of the crossing space) "
                            "should be specified as well."
                        )
                    for key in cross:
                        if key not in feature_set:
                            raise ValueError(
                                "All features referenced "
                                "in the `crosses` argument "
                                "should be present in the `features` dict. "
                                f"Received unknown features: {cross}"
                            )
                    self.crosses.append(Cross(cross, crossing_dim=crossing_dim))
        self.crosses_by_name = {cross.name: cross for cross in self.crosses}

        if output_mode not in {"dict", "concat"}:
            raise ValueError(
                "Invalid value for argument `output_mode`. "
                "Expected one of {'dict', 'concat'}. "
                f"Received: output_mode={output_mode}"
            )
        self.output_mode = output_mode

        self.inputs = {
            name: self._feature_to_input(name, value)
            for name, value in self.features.items()
        }
        self.preprocessors = {
            name: value.preprocessor for name, value in self.features.items()
        }
        self.encoded_features = None
        self.crossers = {
            cross.name: self._cross_to_crosser(cross) for cross in self.crosses
        }
        self.one_hot_encoders = {}
        self._is_adapted = False
        self.concat = None
        self._preprocessed_features_names = None
        self._crossed_features_names = None
        self._sublayers_built = False

    def _feature_to_input(self, name, feature):
        return layers.Input(shape=(1,), dtype=feature.dtype, name=name)

    def _standardize_feature(self, name, feature):
        if isinstance(feature, Feature):
            return feature

        if isinstance(feature, dict):
            return serialization_lib.deserialize_keras_object(feature)

        if feature == "float":
            return self.float(name=name)
        elif feature == "float_normalized":
            return self.float_normalized(name=name)
        elif feature == "float_rescaled":
            return self.float_rescaled(name=name)
        elif feature == "float_discretized":
            return self.float_discretized(
                name=name, num_bins=self.num_discretization_bins
            )
        elif feature == "integer_categorical":
            return self.integer_categorical(name=name)
        elif feature == "string_categorical":
            return self.string_categorical(name=name)
        elif feature == "integer_hashed":
            return self.integer_hashed(self.hashing_dim, name=name)
        elif feature == "string_hashed":
            return self.string_hashed(self.hashing_dim, name=name)
        else:
            raise ValueError(f"Invalid feature type: {feature}")

    def _cross_to_crosser(self, cross):
        return layers.HashedCrossing(cross.crossing_dim, name=cross.name)

    def _list_adaptable_preprocessors(self):
        adaptable_preprocessors = []
        for name in self.features.keys():
            preprocessor = self.preprocessors[name]
            # Special case: a Normalization layer with preset mean/variance.
            # Not adaptable.
            if isinstance(preprocessor, layers.Normalization):
                if preprocessor.input_mean is not None:
                    continue
            # Special case: a TextVectorization layer with provided vocabulary.
            elif isinstance(preprocessor, layers.TextVectorization):
                if preprocessor._has_input_vocabulary:
                    continue
            if hasattr(preprocessor, "adapt"):
                adaptable_preprocessors.append(name)
        return adaptable_preprocessors

    def adapt(self, dataset, verbose=1):
        if not isinstance(dataset, tf.data.Dataset):
            if isinstance(dataset, dict):
                dataset = tf.data.Dataset.from_tensor_slices(dataset)
            else:
                raise ValueError(
                    "`adapt()` can only be called on a tf.data.Dataset or a "
                    "dict of arrays/lists. "
                    f"Received instead: {dataset} (of type {type(dataset)})"
                )

        adaptable_preprocessors = self._list_adaptable_preprocessors()
        if not adaptable_preprocessors:
            self._is_adapted = True
            self.get_encoded_features()
            self.built = True
            self._sublayers_built = True
            return

        # Check if the dataset needs batching
        x = next(iter(dataset))
        if len(x[adaptable_preprocessors[0]].shape) == 0:
            dataset = dataset.batch(32)

        steps = None
        if hasattr(dataset, "cardinality"):
            cardinality = dataset.cardinality()
            if cardinality.numpy() not in (
                tf.data.UNKNOWN_CARDINALITY,
                tf.data.INFINITE_CARDINALITY,
            ):
                steps = int(cardinality.numpy())

        if verbose:
            progbar = Progbar(
                target=steps,
                unit_name="step",
            )

        for name in adaptable_preprocessors:
            preprocessor = self.preprocessors[name]
            if hasattr(preprocessor, "reset_state"):
                preprocessor.reset_state()

        # Handle layers without state-update hooks separately.
        # Let's check which adaptable preprocessors have `update_state`
        update_state_preprocessors = [
            name
            for name in adaptable_preprocessors
            if hasattr(self.preprocessors[name], "update_state")
        ]
        adapt_preprocessors = [
            name
            for name in adaptable_preprocessors
            if not hasattr(self.preprocessors[name], "update_state")
        ]

        for i, batch in enumerate(dataset):
            for name in update_state_preprocessors:
                feature_batch = batch[name]
                if len(feature_batch.shape) in {0, 1}:
                    feature_batch = tf.expand_dims(feature_batch, -1)
                self.preprocessors[name].update_state(feature_batch)
            if verbose:
                progbar.update(i + 1)

        for name in update_state_preprocessors:
            self.preprocessors[name].finalize_state()

        if verbose:
            progbar.update(steps if steps is not None else i + 1, finalize=True)

        # Handle layers that only have `adapt` (like Normalization)
        for name in adapt_preprocessors:
            feature_dataset = dataset.map(lambda x: x[name])
            x = next(iter(feature_dataset))
            if len(x.shape) in {0, 1}:
                feature_dataset = feature_dataset.map(
                    lambda x: tf.expand_dims(x, -1)
                )
            self.preprocessors[name].adapt(feature_dataset)

        self._is_adapted = True
        self.get_encoded_features()  # Finish building the layer
        self.built = True
        self._sublayers_built = True

    def get_inputs(self):
        self._check_if_built()
        return self.inputs

    def get_encoded_features(self):
        self._check_if_adapted()

        if self.encoded_features is None:
            preprocessed_features = self._preprocess_features(self.inputs)
            crossed_features = self._cross_features(preprocessed_features)
            merged_features = self._merge_features(
                preprocessed_features, crossed_features
            )
            self.encoded_features = merged_features
        return self.encoded_features

    def _preprocess_features(self, features):
        return {
            name: self.preprocessors[name](features[name])
            for name in features.keys()
        }

    def _cross_features(self, features):
        all_outputs = {}
        for cross in self.crosses:
            inputs = [features[name] for name in cross.feature_names]
            outputs = self.crossers[cross.name](inputs)
            all_outputs[cross.name] = outputs
        return all_outputs

    def _merge_features(self, preprocessed_features, crossed_features):
        if not self._preprocessed_features_names:
            self._preprocessed_features_names = sorted(
                preprocessed_features.keys()
            )
            self._crossed_features_names = sorted(crossed_features.keys())

        all_names = (
            self._preprocessed_features_names + self._crossed_features_names
        )
        all_features = [
            preprocessed_features[name]
            for name in self._preprocessed_features_names
        ] + [crossed_features[name] for name in self._crossed_features_names]

        if self.output_mode == "dict":
            output_dict = {}
        else:
            features_to_concat = []

        if self._sublayers_built:
            # Fast mode.
            for name, feature in zip(all_names, all_features):
                encoder = self.one_hot_encoders.get(name, None)
                if encoder:
                    feature = encoder(feature)
                if self.output_mode == "dict":
                    output_dict[name] = feature
                else:
                    features_to_concat.append(feature)
            if self.output_mode == "dict":
                return output_dict
            else:
                return self.concat(features_to_concat)

        # If the object isn't built,
        # we create the encoder and concat layers below
        all_specs = [
            self.features[name] for name in self._preprocessed_features_names
        ] + [
            self.crosses_by_name[name] for name in self._crossed_features_names
        ]

        for name, feature, spec in zip(all_names, all_features, all_specs):
            if tree.is_nested(feature):
                dtype = tree.flatten(feature)[0].dtype
            else:
                dtype = feature.dtype
            dtype = standardize_dtype(dtype)

            if spec.output_mode == "one_hot":
                preprocessor = self.preprocessors.get(
                    name
                ) or self.crossers.get(name)

                cardinality = None
                if not dtype.startswith("int"):
                    raise ValueError(
                        f"Feature '{name}' has `output_mode='one_hot'`. "
                        "Thus its preprocessor should return an integer dtype. "
                        f"Instead it returns a {dtype} dtype."
                    )

                if isinstance(
                    preprocessor, (layers.IntegerLookup, layers.StringLookup)
                ):
                    cardinality = preprocessor.vocabulary_size()
                elif isinstance(preprocessor, layers.CategoryEncoding):
                    cardinality = preprocessor.num_tokens
                elif isinstance(preprocessor, layers.Discretization):
                    cardinality = preprocessor.num_bins
                elif isinstance(
                    preprocessor, (layers.HashedCrossing, layers.Hashing)
                ):
                    cardinality = preprocessor.num_bins
                else:
                    raise ValueError(
                        f"Feature '{name}' has `output_mode='one_hot'`. "
                        "However it isn't a standard feature and the "
                        "dimensionality of its output space is not known, "
                        "thus it cannot be one-hot encoded. "
                        "Try using `output_mode='int'`."
                    )
                if cardinality is not None:
                    encoder = layers.CategoryEncoding(
                        num_tokens=cardinality, output_mode="multi_hot"
                    )
                    self.one_hot_encoders[name] = encoder
                    feature = encoder(feature)

            if self.output_mode == "concat":
                dtype = feature.dtype
                if dtype.startswith("int") or dtype == "string":
                    raise ValueError(
                        f"Cannot concatenate features because feature '{name}' "
                        f"has not been encoded (it has dtype {dtype}). "
                        "Consider using `output_mode='dict'`."
                    )
                features_to_concat.append(feature)
            else:
                output_dict[name] = feature

        if self.output_mode == "concat":
            self.concat = TFDConcat(axis=-1)
            return self.concat(features_to_concat)
        else:
            return output_dict

    def _check_if_adapted(self):
        if not self._is_adapted:
            if not self._list_adaptable_preprocessors():
                self._is_adapted = True
            else:
                raise ValueError(
                    "You need to call `.adapt(dataset)` on the FeatureSpace "
                    "before you can start using it."
                )

    def _check_if_built(self):
        if not self._sublayers_built:
            self._check_if_adapted()
            # Finishes building
            self.get_encoded_features()
            self._sublayers_built = True

    def _convert_input(self, x):
        if not isinstance(x, (tf.Tensor, tf.SparseTensor, tf.RaggedTensor)):
            if not isinstance(x, (list, tuple, int, float)):
                x = backend.ops.convert_to_numpy(x)
            x = tf.convert_to_tensor(x)
        return x

    def __call__(self, data):
        self._check_if_built()
        if not isinstance(data, dict):
            raise ValueError(
                "A FeatureSpace can only be called with a dict. "
                f"Received: data={data} (of type {type(data)}"
            )

        # Many preprocessing layers support all backends but many do not.
        # Switch to TF to make FeatureSpace work universally.
        data = {key: self._convert_input(value) for key, value in data.items()}
        rebatched = False
        for name, x in data.items():
            if len(x.shape) == 0:
                data[name] = tf.reshape(x, (1, 1))
                rebatched = True
            elif len(x.shape) == 1:
                data[name] = tf.expand_dims(x, -1)

        with backend_utils.TFGraphScope():
            # This scope is to make sure that inner DataLayers
            # will not convert outputs back to backend-native --
            # they should be TF tensors throughout
            preprocessed_data = self._preprocess_features(data)
            preprocessed_data = tree.map_structure(
                lambda x: self._convert_input(x), preprocessed_data
            )

            crossed_data = self._cross_features(preprocessed_data)
            crossed_data = tree.map_structure(
                lambda x: self._convert_input(x), crossed_data
            )

            merged_data = self._merge_features(preprocessed_data, crossed_data)

        if rebatched:
            if self.output_mode == "concat":
                if merged_data.shape[0] != 1:
                    raise ValueError(
                        "Expected rebatched data to have batch size 1. "
                        f"Received: shape={merged_data.shape}"
                    )
                if (
                    backend.backend() != "tensorflow"
                    and not backend_utils.in_tf_graph()
                ):
                    merged_data = np.array(merged_data)
                merged_data = tf.squeeze(merged_data, axis=0)
            else:
                for name, x in merged_data.items():
                    if len(x.shape) == 2 and x.shape[0] == 1:
                        merged_data[name] = tf.squeeze(x, axis=0)

        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            merged_data = tree.map_structure(
                lambda x: backend.ops.convert_to_tensor(x, dtype=x.dtype),
                merged_data,
            )
        return merged_data

    def get_config(self):
        return {
            "features": serialization_lib.serialize_keras_object(self.features),
            "output_mode": self.output_mode,
            "crosses": serialization_lib.serialize_keras_object(self.crosses),
            "crossing_dim": self.crossing_dim,
            "hashing_dim": self.hashing_dim,
            "num_discretization_bins": self.num_discretization_bins,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_build_config(self):
        return {
            name: feature.preprocessor.get_build_config()
            for name, feature in self.features.items()
        }

    def build_from_config(self, config):
        for name in config.keys():
            preprocessor = self.features[name].preprocessor
            if not preprocessor.built:
                preprocessor.build_from_config(config[name])
        self._is_adapted = True

    def save(self, filepath):
        """Save the `FeatureSpace` instance to a `.keras` file.

        You can reload it via `keras.models.load_model()`:

        ```python
        feature_space.save("featurespace.keras")
        reloaded_fs = keras.models.load_model("featurespace.keras")
        ```
        """
        saving_lib.save_model(self, filepath)

    def save_own_variables(self, store):
        return

    def load_own_variables(self, store):
        return


class TFDConcat(DataLayer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, xs):
        return self.backend.ops.numpy.concatenate(xs, axis=self.axis)


class TFDIdentity(DataLayer):
    def call(self, x):
        return x
