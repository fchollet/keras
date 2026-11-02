import tensorflow as tf

from keras.src.export.saved_model_export_archive import (
    BaseSavedModelExportArchive,
)


class SavedModelExportArchive(BaseSavedModelExportArchive):
    """TensorFlow backend implementation of SavedModel export archive."""

    def _backend_track_layer(self, layer):
        # Unwrap Keras `Variable` wrappers to the underlying `tf.Variable`
        # at tracking time. Endpoint functions (traced as `tf.function`s)
        # capture these same underlying `tf.Variable` objects, so once both
        # paths reference the identical Python object, TensorFlow's own
        # Trackable object-graph machinery dedupes them automatically when
        # the SavedModel is written. No post-hoc filtering needed.
        #
        # Use `._value`, not the public `.value` property: `.value` runs
        # through `_maybe_autocast()` and can return a cast tensor rather
        # than the actual `tf.Variable`, which would break the identity
        # match this dedup relies on.
        variables = [v._value for v in layer.variables]
        trainable_variables = [v._value for v in layer.trainable_variables]
        non_trainable_variables = [
            v._value for v in layer.non_trainable_variables
        ]
        self._tf_trackable.variables += variables
        self._tf_trackable.trainable_variables += trainable_variables
        self._tf_trackable.non_trainable_variables += non_trainable_variables

    def _backend_add_endpoint(self, name, fn, input_signature, **kwargs):
        decorated_fn = tf.function(
            fn, input_signature=input_signature, autograph=False
        )
        return decorated_fn

    def _convert_to_tf_variable(self, backend_variable):
        # Used by `add_variable_collection()`. Input may already be a plain
        # `tf.Variable`, or a Keras `Variable` wrapper to unwrap.
        if isinstance(backend_variable, tf.Variable):
            return backend_variable
        return backend_variable._value
