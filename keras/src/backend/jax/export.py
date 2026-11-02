import copy
import inspect
import itertools
import string
import warnings

from keras.src import tree
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.export.saved_model_export_archive import (
    BaseSavedModelExportArchive,
)
from keras.src.utils.module_utils import tensorflow as tf


class SavedModelExportArchive(BaseSavedModelExportArchive):
    """JAX backend implementation of SavedModel export archive."""

    def _backend_init(self):
        """JAX-specific initialization."""
        self._backend_variables = []
        self._backend_trainable_variables = []
        self._backend_non_trainable_variables = []

    def _backend_track_layer(self, layer):
        # Variables in the lists below are actually part of the trackables
        # that get saved, because the lists are created in __init__.
        trainable_variables = layer.trainable_variables
        non_trainable_variables = layer.non_trainable_variables

        self._tf_trackable.trainable_variables += tree.map_structure(
            self._convert_to_tf_variable, trainable_variables
        )
        self._tf_trackable.non_trainable_variables += tree.map_structure(
            self._convert_to_tf_variable, non_trainable_variables
        )
        self._tf_trackable.variables = (
            self._tf_trackable.trainable_variables
            + self._tf_trackable.non_trainable_variables
        )

        self._backend_trainable_variables += trainable_variables
        self._backend_non_trainable_variables += non_trainable_variables
        self._backend_variables = (
            self._backend_trainable_variables
            + self._backend_non_trainable_variables
        )

    def _backend_add_endpoint(self, name, fn, input_signature, **kwargs):
        jax2tf_kwargs = kwargs.pop("jax2tf_kwargs", None)
        # Use `copy.copy()` to avoid modification issues.
        jax2tf_kwargs = copy.copy(jax2tf_kwargs) or {}
        is_static = bool(kwargs.pop("is_static", False))

        # Configure `jax2tf_kwargs`
        if "native_serialization" not in jax2tf_kwargs:
            jax2tf_kwargs["native_serialization"] = (
                self._check_device_compatible()
            )
        if "polymorphic_shapes" not in jax2tf_kwargs:
            jax2tf_kwargs["polymorphic_shapes"] = self._to_polymorphic_shape(
                input_signature
            )

        # Note: we truncate the number of parameters to what is specified by
        # `input_signature`.
        fn_signature = inspect.signature(fn)
        fn_parameters = list(fn_signature.parameters.values())

        if is_static:
            from jax.experimental import jax2tf

            jax_fn = jax2tf.convert(fn, **jax2tf_kwargs)
            jax_fn.__signature__ = inspect.Signature(
                parameters=fn_parameters[0 : len(input_signature)],
                return_annotation=fn_signature.return_annotation,
            )

            decorated_fn = tf.function(
                jax_fn,
                input_signature=input_signature,
                autograph=False,
            )
        else:
            # 1. Create a stateless wrapper for `fn`
            # 2. jax2tf the stateless wrapper
            # 3. Create a stateful function that binds the variables with
            #    the jax2tf converted stateless wrapper
            # 4. Make the signature of the stateful function the same as the
            #    original function
            # 5. Wrap in a `tf.function`
            def stateless_fn(variables, *args, **kwargs):
                state_mapping = zip(self._backend_variables, variables)
                with StatelessScope(state_mapping=state_mapping) as scope:
                    output = fn(*args, **kwargs)

                # Gather updated non-trainable variables
                non_trainable_variables = []
                for var in self._backend_non_trainable_variables:
                    new_value = scope.get_current_value(var)
                    non_trainable_variables.append(new_value)
                return output, non_trainable_variables

            jax2tf_stateless_fn = self._convert_jax2tf_function(
                stateless_fn, input_signature, jax2tf_kwargs=jax2tf_kwargs
            )

            def stateful_fn(*args, **kwargs):
                output, non_trainable_variables = jax2tf_stateless_fn(
                    # Change the trackable `ListWrapper` to a plain `list`
                    list(self._tf_trackable.variables),
                    *args,
                    **kwargs,
                )
                for var, new_value in zip(
                    self._tf_trackable.non_trainable_variables,
                    non_trainable_variables,
                ):
                    var.assign(tf.cast(new_value, var.dtype))
                return output

            stateful_fn.__signature__ = inspect.Signature(
                parameters=fn_parameters[0 : len(input_signature)],
                return_annotation=fn_signature.return_annotation,
            )

            decorated_fn = tf.function(
                stateful_fn,
                input_signature=input_signature,
                autograph=False,
            )
        return decorated_fn

    def _convert_jax2tf_function(self, fn, input_signature, jax2tf_kwargs=None):
        from jax.experimental import jax2tf

        variables_shapes = self._to_polymorphic_shape(
            self._backend_variables, allow_none=False
        )
        input_shapes = list(jax2tf_kwargs["polymorphic_shapes"])
        jax2tf_kwargs["polymorphic_shapes"] = [variables_shapes] + input_shapes
        return jax2tf.convert(fn, **jax2tf_kwargs)

    def _to_polymorphic_shape(self, struct, allow_none=True):
        if allow_none:
            # Generates unique names: a, b, ... z, aa, ab, ... az, ba, ... zz
            # for unknown non-batch dims. Defined here to be scope per endpoint.
            dim_names = itertools.chain(
                string.ascii_lowercase,
                itertools.starmap(
                    lambda a, b: a + b,
                    itertools.product(string.ascii_lowercase, repeat=2),
                ),
            )

        def convert_shape(x):
            poly_shape = []
            for index, dim in enumerate(list(x.shape)):
                if dim is not None:
                    poly_shape.append(str(dim))
                elif not allow_none:
                    raise ValueError(
                        f"Illegal None dimension in {x} with shape {x.shape}"
                    )
                elif index == 0:
                    poly_shape.append("batch")
                else:
                    poly_shape.append(next(dim_names))
            return f"({', '.join(poly_shape)})"

        return tree.map_structure(convert_shape, struct)

    def _check_device_compatible(self):
        from jax import default_backend as jax_device

        if (
            jax_device() == "gpu"
            and len(tf.config.list_physical_devices("GPU")) == 0
        ):
            warnings.warn(
                "JAX backend is using GPU for export, but installed "
                "TF package cannot access GPU, so reloading the model with "
                "the TF runtime in the same environment will not work. "
                "To use JAX-native serialization for high-performance export "
                "and serving, please install `tensorflow-gpu` and ensure "
                "CUDA version compatibility between your JAX and TF "
                "installations."
            )
            return False
        else:
            return True
