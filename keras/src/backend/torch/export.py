import copy
import warnings

import torch

from keras.src import layers
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import make_tf_tensor_spec
from keras.src.export.saved_model_export_archive import (
    BaseSavedModelExportArchive,
)
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.module_utils import torch_xla


class SavedModelExportArchive(BaseSavedModelExportArchive):
    """Torch backend implementation of SavedModel export archive."""

    def _backend_track_layer(self, layer):
        raise NotImplementedError(
            "`track` is not supported for `Layer`s and `Model`s in the torch "
            "backend. Use `track_and_add_endpoint` instead."
        )

    def _backend_add_endpoint(self, name, fn, input_signature, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not supported for `Layer`s and `Model`s in the "
            "torch backend. Use `track_and_add_endpoint` instead."
        )

    def track_and_add_endpoint(self, name, resource, input_signature, **kwargs):
        if name in self._endpoint_names:
            raise ValueError(f"Endpoint name '{name}' is already taken.")
        if not isinstance(resource, layers.Layer):
            raise ValueError(
                "Invalid resource type. Expected an instance of a Keras "
                "`Layer` or `Model`. "
                f"Received: resource={resource} (of type {type(resource)})"
            )
        if not resource.built:
            raise ValueError(
                "The layer provided has not yet been built. "
                "It must be built before export."
            )

        input_signature = tree.map_structure(
            make_tf_tensor_spec, input_signature
        )
        # Disable false alarms related to lifting parameters.
        warnings.filterwarnings("ignore", message=".*created when tracing.*")
        warnings.filterwarnings(
            "ignore", message=".*Unable to find the path of the module.*"
        )

        if not isinstance(resource, torch.nn.Module):
            raise TypeError(
                "`resource` must be an instance of `torch.nn.Module`. "
                f"Received: resource={resource} (of type {type(resource)})"
            )

        sample_inputs = tree.map_structure(
            lambda x: convert_spec_to_tensor(x, replace_none_number=2),
            input_signature,
        )
        sample_inputs = tuple(sample_inputs)

        # Build dynamic_shapes from input_signature where shape has None
        # Use a shared "batch" dim for dimension 0 across all inputs
        batch_dim = torch.export.Dim("batch", min=1)
        dynamic_shapes = []
        for spec in input_signature:
            dim_spec = {}
            for dim_idx, dim_val in enumerate(spec.shape):
                if dim_val is None:
                    if dim_idx == 0:
                        dim_spec[dim_idx] = batch_dim
                    else:
                        dim_spec[dim_idx] = torch.export.Dim(
                            f"dim_{len(dynamic_shapes)}_{dim_idx}", min=1
                        )
            dynamic_shapes.append(dim_spec if dim_spec else None)
        dynamic_shapes = tuple(dynamic_shapes) if any(dynamic_shapes) else None

        # Ref: torch_xla.tf_saved_model_integration
        exported = torch.export.export(
            resource, sample_inputs, dynamic_shapes=dynamic_shapes, strict=False
        )
        options = torch_xla.stablehlo.StableHLOExportOptions(
            override_tracing_arguments=sample_inputs
        )
        stablehlo_model = torch_xla.stablehlo.exported_program_to_stablehlo(
            exported, options
        )
        state_dict_keys = list(stablehlo_model._bundle.state_dict.keys())

        # Remove unused variables.
        for k in state_dict_keys:
            if "lifted" not in k:
                stablehlo_model._bundle.state_dict.pop(k)

        bundle = copy.deepcopy(stablehlo_model._bundle)
        bundle.state_dict = {
            k: tf.Variable(v, trainable=False, name=k)
            for k, v in bundle.state_dict.items()
        }
        bundle.additional_constants = [
            tf.Variable(v, trainable=False) for v in bundle.additional_constants
        ]

        # Track variables in `bundle` for `write_out`.
        self._tf_trackable.variables += (
            list(bundle.state_dict.values()) + bundle.additional_constants
        )

        # Ref: torch_xla.tf_saved_model_integration.save_stablehlo_graph_as_tf
        def make_tf_function(func, bundle):
            from tensorflow.compiler.tf2xla.python import xla as tfxla

            def _get_shape_with_dynamic(signature):
                shape = copy.copy(signature.shape)
                for i in signature.dynamic_dims:
                    shape[i] = None
                return shape

            def _extract_call_parameters(args, meta, bundle):
                call_args = []
                if meta.input_pytree_spec is not None:
                    args = tree.flatten(args)
                for loc in meta.input_locations:
                    if loc.type_ == torch_xla.stablehlo.VariableType.PARAMETER:
                        call_args.append(bundle.state_dict[loc.name])
                    elif loc.type_ == torch_xla.stablehlo.VariableType.CONSTANT:
                        call_args.append(
                            bundle.additional_constants[loc.position]
                        )
                    else:
                        call_args.append(args[loc.position])
                return call_args

            def inner(*args):
                Touts = [sig.dtype for sig in func.meta.output_signature]
                Souts = [
                    _get_shape_with_dynamic(sig)
                    for sig in func.meta.output_signature
                ]
                call_args = _extract_call_parameters(args, func.meta, bundle)
                results = tfxla.call_module(
                    tuple(call_args),
                    version=5,
                    Tout=Touts,  # dtype information
                    Sout=Souts,  # Shape information
                    function_list=[],
                    module=func.bytecode,
                )
                if len(Souts) == 1:
                    results = results[0]
                return results

            return inner

        decorated_fn = tf.function(
            make_tf_function(
                stablehlo_model._bundle.stablehlo_funcs[0], bundle
            ),
            input_signature=input_signature,
        )
        self._endpoint_signatures[name] = input_signature
        setattr(self._tf_trackable, name, decorated_fn)
        self._endpoint_names.append(name)
        return decorated_fn
