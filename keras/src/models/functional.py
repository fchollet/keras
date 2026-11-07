import copy
import inspect
import typing
import warnings

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.backend.common import get_keras_mask
from keras.src.backend.common import global_state
from keras.src.backend.common import set_keras_mask
from keras.src.layers.core.input_layer import Input
from keras.src.layers.core.input_layer import InputLayer
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.model import Model
from keras.src.ops.function import Function
from keras.src.ops.function import _build_map
from keras.src.ops.function import make_node_key
from keras.src.ops.node import KerasHistory
from keras.src.ops.node import Node
from keras.src.ops.operation import Operation
from keras.src.saving import serialization_lib
from keras.src.utils import tracking


def _match_expected_structure(expected, provided):
    """Map provided inputs onto the expected nested structure.

    Entries are matched by path rather than by flattening order, so extra
    dict keys cannot shift the alignment of the remaining inputs. Recursion
    is delegated to the `tree` API so that every registered structure type
    is handled, not just dicts, lists and tuples.

    Extra entries in `provided` are dropped with a warning. Raises
    `ValueError` if `provided` is missing anything `expected` requires.
    """
    err_msg = "The structure of `inputs` doesn't match the expected structure"
    provided_by_path = dict(tree.flatten_with_path(provided))
    expected_paths = [path for path, _ in tree.flatten_with_path(expected)]

    def format_paths(paths):
        return ", ".join(
            "'" + ".".join(str(key) for key in path) + "'" for path in paths
        )

    missing = [path for path in expected_paths if path not in provided_by_path]
    if missing:
        raise ValueError(f"{err_msg}. Missing fields: {format_paths(missing)}.")
    extra = [
        path for path in provided_by_path if path not in set(expected_paths)
    ]
    if extra:
        warnings.warn(
            f"{err_msg}. Ignored fields: {format_paths(extra)}.",
            UserWarning,
            stacklevel=4,
        )
    return tree.pack_sequence_as(
        expected, [provided_by_path[path] for path in expected_paths]
    )


class Functional(Function, Model):
    """A `Functional` model is a `Model` defined as a directed graph of layers.

    Three types of `Model` exist: subclassed `Model`, `Functional` model,
    and `Sequential` (a special case of `Functional`).

    A `Functional` model can be instantiated by passing two arguments to
    `__init__()`. The first argument is the `keras.Input` objects
    that represent the inputs to the model.
    The second argument specifies the output tensors that represent
    the outputs of this model. Both arguments can be a nested structure
    of tensors.

    Example:

    ```
    inputs = {'x1': keras.Input(shape=(10,), name='x1'),
              'x2': keras.Input(shape=(1,), name='x2')}
    t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
    outputs = keras.layers.Add()([t, inputs['x2']])
    model = keras.Model(inputs, outputs)
    ```

    A `Functional` model constructed using the Functional API can also
    include raw Keras 3 ops.

    Example:

    ```python
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(1)(inputs)
    outputs = ops.nn.relu(x)
    model = keras.Model(inputs, outputs)
    ```

    A new `Functional` model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.

    Example:

    ```python
    inputs = keras.Input(shape=(None, None, 3))
    processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
    conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
    pooling = keras.layers.GlobalAveragePooling2D()(conv)
    feature = keras.layers.Dense(10)(pooling)

    full_model = keras.Model(inputs, feature)
    backbone = keras.Model(processed, conv)
    activations = keras.Model(conv, feature)
    ```

    Note that the `backbone` and `activations` models are not
    created with `keras.Input` objects, but with the tensors
    that are originated from `keras.Input` objects.
    Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard `Functional` model that support
    all the existing API.

    Args:
        inputs: List of input tensors (must be created via `keras.Input()`
            or originated from `keras.Input()`).
        outputs: List of output tensors.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.
    """

    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, inputs, outputs, name=None, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, backend.KerasTensor) and k != v.name:
                    warnings.warn(
                        "When providing `inputs` as a dict, all keys in the "
                        "dict must match the names of the corresponding "
                        f"tensors. Received key '{k}' mapping to value {v} "
                        f"which has name '{v.name}'. Change the tensor name to "
                        f"'{k}' (via `Input(..., name='{k}')`)"
                    )

        trainable = kwargs.pop("trainable", None)
        flat_inputs = tree.flatten(inputs)
        flat_outputs = tree.flatten(outputs)
        for x in flat_inputs:
            if not isinstance(x, backend.KerasTensor):
                raise ValueError(
                    "All `inputs` values must be KerasTensors. Received: "
                    f"inputs={inputs} including invalid value {x} of "
                    f"type {type(x)}"
                )
        for x in flat_outputs:
            if not isinstance(x, backend.KerasTensor):
                raise ValueError(
                    "All `outputs` values must be KerasTensors. Received: "
                    f"outputs={outputs} including invalid value {x} of "
                    f"type {type(x)}"
                )

        if not all(is_input_keras_tensor(t) for t in flat_inputs):
            inputs, outputs = clone_graph_nodes(inputs, outputs)

        with tracking.DotNotTrackScope():
            Function.__init__(self, inputs, outputs, name=name)

        if trainable is not None:
            self.trainable = trainable

        for layer in self.layers:
            self._tracker.track(layer)
        self.build(None)
        # We will convert directly (to the correct dtype per input).
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        output_layers = [x._keras_history[0] for x in self.outputs]
        self.output_names = [x.name for x in output_layers]

    def _lock_state(self):
        # Unlike other layers, we allow Functional state to be mutable after
        # build. E.g. to attach a layer to a model that is not part of the
        # functional DAG.
        pass

    def _obj_type(self):
        return "Functional"

    @property
    def layers(self):
        layers = []
        for operation in self._operations:
            if isinstance(operation, Layer):
                layers.append(operation)
        return layers

    @layers.setter
    def layers(self, _):
        raise AttributeError(
            "`Model.layers` attribute is reserved and should not be used. "
            "Please use another name."
        )

    def call(self, inputs, training=None, mask=None):
        # Add support for training, masking
        inputs = self._standardize_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = tree.flatten(mask)
            for x, mask in zip(inputs, masks):
                if mask is not None:
                    set_keras_mask(x, mask)
        outputs = self._run_through_graph(inputs)
        return unpack_singleton(outputs)

    def compute_output_spec(self, inputs, training=None, mask=None):
        # From Function
        if mask is not None:
            has_input_masks = any(
                input_mask is not None for input_mask in tree.flatten(mask)
            )
        else:
            has_input_masks = any(
                x is not None and backend.get_keras_mask(x) is not None
                for x in tree.flatten(inputs)
            )
        has_output_masks = any(
            x is not None and backend.get_keras_mask(x) is not None
            for x in tree.flatten(self._outputs_struct)
        )
        if not has_input_masks and not has_output_masks:
            return super().compute_output_spec(inputs)

        flat_inputs = self._standardize_inputs(inputs)
        if mask is not None:
            for x, input_mask in zip(flat_inputs, tree.flatten(mask)):
                if x is not None and input_mask is not None:
                    backend.set_keras_mask(x, input_mask)

        def mask_propagation_call_fn(op, *args, **kwargs):
            outputs = op.compute_output_spec(*args, **kwargs)
            if "mask" in kwargs:
                previous_mask = kwargs["mask"]
            elif args:
                previous_mask = tree.map_structure(
                    backend.get_keras_mask, args[0]
                )
            else:
                previous_mask = None

            if getattr(op, "supports_masking", False):
                output_masks = op.compute_mask(
                    args[0] if args else None, previous_mask
                )
                if output_masks is not None:
                    flat_outputs = tree.flatten(outputs)
                    flat_output_masks = tree.flatten(output_masks)
                    for output, output_mask in zip(
                        flat_outputs, flat_output_masks
                    ):
                        if output is not None and output_mask is not None:
                            backend.set_keras_mask(output, output_mask)
            return outputs

        return self._run_through_graph(
            flat_inputs, call_fn=mask_propagation_call_fn
        )

    def compute_output_shape(self, input_shape):
        # From Function
        return super().compute_output_shape(input_shape)

    def build(self, input_shape):
        self.built = True

    @property
    def input_shape(self):
        input_shapes = tree.map_structure(lambda x: x.shape, self.inputs)
        if isinstance(input_shapes, list) and len(input_shapes) == 1:
            return input_shapes[0]
        return input_shapes

    @property
    def output_shape(self):
        output_shapes = tree.map_structure(lambda x: x.shape, self.outputs)
        if isinstance(output_shapes, list) and len(output_shapes) == 1:
            return output_shapes[0]
        return output_shapes

    def _assert_input_compatibility(self, *args):
        flat_inputs = tree.flatten(args[0])
        for x, input_tensor in zip(flat_inputs, self._inputs):
            if x is None:
                input_layer = input_tensor._keras_history.operation
                if (
                    isinstance(input_layer, InputLayer)
                    and not input_layer.optional
                ):
                    raise ValueError(
                        (
                            f"The input '{input_tensor.name}' is not optional, "
                            "but None was passed. "
                            "Please provide a valid tensor."
                        )
                    )
        return super(Model, self)._assert_input_compatibility(*args)

    def _maybe_warn_inputs_struct_mismatch(self, inputs, raise_exception=False):
        try:
            # We first normalize to tuples before performing the check to
            # suppress warnings when encountering mismatched tuples and lists.
            tree.assert_same_structure(
                tree.lists_to_tuples(inputs),
                tree.lists_to_tuples(self._inputs_struct),
            )
        except:
            model_inputs_struct = tree.map_structure(
                lambda x: "None" if x is None else x.name,
                self._inputs_struct,
            )
            inputs_struct = tree.map_structure(
                lambda x: "None" if x is None else f"Tensor(shape={x.shape})",
                inputs,
            )
            msg = (
                "The structure of `inputs` doesn't match the expected "
                f"structure.\nExpected: {model_inputs_struct}\n"
                f"Received: inputs={inputs_struct}"
            )
            if raise_exception:
                raise ValueError(msg)
            warnings.warn(msg)

    def _convert_inputs_to_tensors(self, flat_inputs):
        converted = []
        for x, input_tensor in zip(flat_inputs, self._inputs):
            if x is None:
                # Only enforce optional semantics when the input tensor was
                # created by an `InputLayer`. For other kinds of tensors
                # (e.g. tensors produced dynamically) preserve previous
                # behavior and allow None to pass through.
                input_layer = None
                if (
                    hasattr(input_tensor, "_keras_history")
                    and input_tensor._keras_history
                ):
                    input_layer = input_tensor._keras_history.operation

                if isinstance(input_layer, InputLayer):
                    if not input_layer.optional:
                        input_name = input_tensor.name
                        raise ValueError(
                            (
                                f"The input '{input_name}' is not optional, "
                                "but None was passed. "
                                "Please provide a valid tensor."
                            )
                        )
                # If input_layer is not an InputLayer, fall back to previous
                # behavior and accept None.
                converted.append(x)
            else:
                mask = backend.get_keras_mask(x)
                converted_input = ops.convert_to_tensor(
                    x,
                    dtype=input_tensor.dtype,
                    sparse=input_tensor.sparse,
                )
                if mask is not None:
                    backend.set_keras_mask(converted_input, mask)
                converted.append(converted_input)
        return converted

    def _adjust_input_rank(self, flat_inputs):
        adjusted = []
        for i, x in enumerate(flat_inputs):
            ref_shape = self._inputs[i].shape
            if x is None:
                adjusted.append(x)
                continue
            x_rank = len(x.shape)
            ref_rank = len(ref_shape)
            if x_rank == ref_rank:
                adjusted.append(x)
                continue
            if x_rank == ref_rank + 1:
                if x.shape[-1] == 1:
                    adjusted.append(ops.squeeze(x, axis=-1))
                    continue
            if x_rank == ref_rank - 1:
                if ref_shape[-1] == 1:
                    adjusted.append(ops.expand_dims(x, axis=-1))
                    continue
            flat_paths_and_inputs = tree.flatten_with_path(self._inputs_struct)
            path = ".".join(str(p) for p in flat_paths_and_inputs[i][0])
            raise ValueError(
                f"Invalid input shape for input {x} with name "
                f"'{self._inputs[i].name}' and path '{path}'. Expected shape "
                f"{ref_shape}, but input has incompatible shape {x.shape}"
            )
        # Add back metadata.
        for i in range(len(flat_inputs)):
            if hasattr(flat_inputs[i], "_keras_history"):
                adjusted[i]._keras_history = flat_inputs[i]._keras_history
            mask = get_keras_mask(flat_inputs[i])
            if mask is not None:
                set_keras_mask(adjusted[i], mask)
        return adjusted

    def _standardize_inputs(self, inputs):
        raise_exception = False
        if (
            isinstance(self._inputs_struct, list)
            and len(self._inputs_struct) == 1
            and ops.is_tensor(inputs)
        ):
            inputs = [inputs]
        elif isinstance(inputs, dict) and isinstance(self._inputs_struct, dict):
            inputs = _match_expected_structure(self._inputs_struct, inputs)
        elif isinstance(inputs, dict) and not isinstance(
            self._inputs_struct, dict
        ):
            # This is to avoid warning
            # when we have reconcilable dict/list structs
            if hasattr(self._inputs_struct, "__len__") and all(
                isinstance(i, backend.KerasTensor) for i in self._inputs_struct
            ):
                expected_keys = set(i.name for i in self._inputs_struct)
                keys = set(inputs.keys())
                if expected_keys.issubset(keys):
                    inputs = [inputs[i.name] for i in self._inputs_struct]
                else:
                    raise_exception = True
            elif isinstance(self._inputs_struct, backend.KerasTensor):
                if self._inputs_struct.name in inputs:
                    inputs = [inputs[self._inputs_struct.name]]
                else:
                    raise_exception = True
            else:
                raise_exception = True
        else:
            # Drop extra dict keys nested inside other structures, e.g.
            # `[{"a": x, "extra": y}]`. Entries are matched by path so extra
            # keys cannot misalign the remaining inputs. If the structures
            # are not otherwise reconcilable, leave `inputs` untouched and
            # let the existing mismatch handling report it, since it gives a
            # more precise message than a generic structure error.
            try:
                inputs = _match_expected_structure(self._inputs_struct, inputs)
            except ValueError:
                pass
        if (
            isinstance(self._inputs_struct, dict)
            and not isinstance(inputs, dict)
            and list(self._inputs_struct.keys())
            != sorted(self._inputs_struct.keys())
        ):
            raise_exception = True
        self._maybe_warn_inputs_struct_mismatch(
            inputs, raise_exception=raise_exception
        )

        flat_inputs = tree.flatten(inputs)
        flat_inputs = self._convert_inputs_to_tensors(flat_inputs)
        return self._adjust_input_rank(flat_inputs)

    @property
    def input(self):
        # For backwards compatibility,
        # override `input` to retrieve the used-provided
        # constructor inputs
        return self._inputs_struct

    @property
    def output(self):
        return self._outputs_struct

    def add_loss(self, loss):
        # Symbolic only. TODO
        raise NotImplementedError

    @property
    def input_spec(self):
        if hasattr(self, "_manual_input_spec"):
            return self._manual_input_spec

        def shape_with_no_batch_size(x):
            x = list(x)
            if x:
                x[0] = None
            return tuple(x)

        def make_spec_for_tensor(x, name=None):
            optional = False
            if isinstance(x._keras_history[0], InputLayer):
                if x._keras_history[0].optional:
                    optional = True
            return InputSpec(
                shape=shape_with_no_batch_size(x.shape),
                allow_last_axis_squeeze=True,
                name=x._keras_history[0].name if name is None else name,
                optional=optional,
            )

        if isinstance(self._inputs_struct, dict):
            if all(
                isinstance(x, backend.KerasTensor)
                for x in self._inputs_struct.values()
            ):
                # Case where `_nested_inputs` is a plain dict of Inputs.
                names = sorted(self._inputs_struct.keys())
                return [
                    make_spec_for_tensor(self._inputs_struct[name], name=name)
                    for name in names
                ]
            return None  # Deeply nested dict: skip checks.
        struct = self._inputs_struct
        if not isinstance(struct, (list, tuple)):
            struct = [struct]
        if not all(isinstance(x, backend.KerasTensor) for x in struct):
            # Anything nested inside the sequence, e.g. `[{"a": ...}]` or
            # `[[x, y], z]`. Flat specs cannot describe those, and comparing
            # them against the flattened inputs would reject extra dict keys
            # before `_standardize_inputs` gets a chance to drop them.
            return None
        return [make_spec_for_tensor(x) for x in self.inputs]

    @input_spec.setter
    def input_spec(self, value):
        self._manual_input_spec = value

    def get_config(self):
        if not functional_like_constructor(self.__class__):
            # Subclassed networks are not serializable
            # (unless serialization is implemented by
            # the author of the subclassed network).
            return Model.get_config(self)

        config = {
            "name": self.name,
            "trainable": self.trainable,
        }
        # Build a map from a layer unique name (make_node_key)
        # to the index of the nodes that are saved in the config.
        # Only nodes in network_nodes are saved.
        node_reindexing_map = {}
        for operation in self.operations:
            if issubclass(operation.__class__, Functional):
                # Functional models start with a pre-existing node
                # linking their input to output.
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(
                operation._inbound_nodes
            ):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # i.e. we mark it to be saved
                    node_reindexing_map[node_key] = kept_nodes
                    kept_nodes += 1

        # serialize and save the layers in layer_configs
        layer_configs = []
        for operation in self.operations:  # From the earliest layers on.
            filtered_inbound_nodes = []
            for original_node_index, node in enumerate(
                operation._inbound_nodes
            ):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # The node is relevant to the model:
                    # add to filtered_inbound_nodes.
                    node_data = serialize_node(node, own_nodes=self._nodes)
                    if node_data is not None:
                        filtered_inbound_nodes.append(node_data)

            serialize_obj_fn = serialization_lib.serialize_keras_object
            if global_state.get_global_attribute("use_legacy_config", False):
                # Legacy format serialization used for H5 and SavedModel
                serialize_obj_fn = legacy_serialization.serialize_keras_object
            layer_config = serialize_obj_fn(operation)
            layer_config["name"] = operation.name
            layer_config["inbound_nodes"] = filtered_inbound_nodes
            layer_configs.append(layer_config)
        config["layers"] = layer_configs

        # Gather info about inputs and outputs.
        def get_tensor_config(tensor):
            operation = tensor._keras_history[0]
            node_index = tensor._keras_history[1]
            tensor_index = tensor._keras_history[2]
            node_key = make_node_key(operation, node_index)
            if node_key not in self._nodes:
                raise RuntimeError(
                    f"Internal error: could not find node key {node_key}."
                )
            new_node_index = node_reindexing_map[node_key]
            return [operation.name, new_node_index, tensor_index]

        def map_tensors(tensors):
            return tree.map_structure(get_tensor_config, tensors)

        config["input_layers"] = map_tensors(self._inputs_struct)
        config["output_layers"] = map_tensors(self._outputs_struct)
        return copy.deepcopy(config)


def functional_from_config(cls, config, custom_objects=None):
    """Instantiates a Functional model from its config (from `get_config()`).

    Args:
        cls: Class of the model, e.g. a custom subclass of `Model`.
        config: Output of `get_config()` for the original model instance.
        custom_objects: Optional dict of custom objects.

    Returns:
        An instance of `cls`.
    """
    # Layer instances created during
    # the graph reconstruction process
    created_layers = {}

    # Dictionary mapping layer instances to
    # node data that specifies a layer call.
    # It acts as a queue that maintains any unprocessed
    # layer call until it becomes possible to process it
    # (i.e. until the input tensors to the call all exist).
    unprocessed_nodes = {}

    def add_unprocessed_node(layer, node_data):
        """Add node to layer list

        Arg:
            layer: layer object
            node_data: Node data specifying layer call
        """
        if layer not in unprocessed_nodes:
            unprocessed_nodes[layer] = [node_data]
        else:
            unprocessed_nodes[layer].append(node_data)

    def process_node(layer, node_data):
        """Reconstruct node by linking to inbound layers

        Args:
            layer: Layer to process
            node_data: List of layer configs
        """
        args, kwargs = deserialize_node(node_data, created_layers)
        # Call layer on its inputs, thus creating the node
        # and building the layer if needed.
        layer(*args, **kwargs)

    def process_layer(layer_data):
        """Deserializes a layer and index its inbound nodes.

        Args:
            layer_data: layer config dict.
        """
        layer_name = layer_data["name"]

        # Instantiate layer.
        if "module" not in layer_data:
            # Legacy format deserialization (no "module" key)
            # used for H5 and SavedModel formats
            layer = saving_utils.model_from_config(
                layer_data, custom_objects=custom_objects
            )
        else:
            layer = serialization_lib.deserialize_keras_object(
                layer_data, custom_objects=custom_objects
            )
        if not isinstance(layer, Operation):
            raise ValueError(
                "Unexpected object from deserialization, expected a layer or "
                f"operation, got a {type(layer)}"
            )
        created_layers[layer_name] = layer

        # Gather layer inputs.
        inbound_nodes_data = layer_data["inbound_nodes"]
        for node_data in inbound_nodes_data:
            # We don't process nodes (i.e. make layer calls)
            # on the fly because the inbound node may not yet exist,
            # in case of layer shared at different topological depths
            # (e.g. a model such as A(B(A(B(x)))))
            add_unprocessed_node(layer, node_data)

    # Extract config used to instantiate Functional model from the config. The
    # remaining config will be passed as keyword arguments to the Model
    # constructor.
    functional_config = {}
    for key in ["layers", "input_layers", "output_layers"]:
        functional_config[key] = config.pop(key)
    for key in ["name", "trainable"]:
        if key in config:
            functional_config[key] = config.pop(key)
        else:
            functional_config[key] = None

    # First, we create all layers and enqueue nodes to be processed
    for layer_data in functional_config["layers"]:
        process_layer(layer_data)

    # Then we process nodes in order of layer depth.
    # Nodes that cannot yet be processed (if the inbound node
    # does not yet exist) are re-enqueued, and the process
    # is repeated until all nodes are processed.
    while unprocessed_nodes:
        some_nodes_processed = False
        for layer_data in functional_config["layers"]:
            layer = created_layers[layer_data["name"]]

            # Process all nodes in layer, if not yet processed
            if layer in unprocessed_nodes:
                node_data_list = unprocessed_nodes[layer]

                # Process nodes in order
                node_index = 0
                while node_index < len(node_data_list):
                    node_data = node_data_list[node_index]
                    try:
                        process_node(layer, node_data)

                    # If the node does not have all inbound layers
                    # available, stop processing and continue later
                    except IndexError:
                        break

                    node_index += 1

                # If all nodes processed remove the layer
                if node_index >= len(node_data_list):
                    some_nodes_processed = True
                    del unprocessed_nodes[layer]
                # If not all nodes processed then store unprocessed nodes
                elif node_index > 0:
                    some_nodes_processed = True
                    unprocessed_nodes[layer] = node_data_list[node_index:]

        if not some_nodes_processed:
            raise ValueError(
                "Invalid Functional model configuration. The graph of the "
                "Functional model either has loops or disconnected nodes. "
                "The following nodes failed to be resolved: "
                f"{unprocessed_nodes}"
            )

    # Create list of input and output tensors and return new class
    name = functional_config["name"]
    trainable = functional_config["trainable"]

    def get_tensor(layer_name, node_index, tensor_index):
        if layer_name not in created_layers:
            raise RuntimeError(
                f"Internal error: could not find layer {layer_name}."
            )
        layer = created_layers[layer_name]
        if isinstance(layer, Functional):
            # Functional models start out with a built-in node.
            node_index -= 1
        layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
        return layer_output_tensors[tensor_index]

    def map_tensors(tensors):
        if (
            isinstance(tensors, list)
            and len(tensors) == 3
            and isinstance(tensors[0], str)
        ):
            # Leaf
            return get_tensor(*tensors)
        if isinstance(tensors, dict):
            return {k: map_tensors(v) for k, v in tensors.items()}
        if isinstance(tensors, tuple):
            return tuple([map_tensors(v) for v in tensors])
        return [map_tensors(v) for v in tensors]

    input_tensors = map_tensors(functional_config["input_layers"])
    output_tensors = map_tensors(functional_config["output_layers"])

    return cls(
        inputs=input_tensors,
        outputs=output_tensors,
        name=name,
        trainable=trainable,
        **config,
    )


def functional_like_constructor(cls):
    init_spec = inspect.getfullargspec(cls.__init__)
    init_args = init_spec.args[1:]
    functional_init_args = inspect.getfullargspec(Functional.__init__).args[1:]
    if init_args == functional_init_args:
        return True
    if init_spec.varargs is not None and init_spec.varkw is not None:
        return True
    return False


def unpack_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def serialize_node(node, own_nodes=()):
    if not node.input_tensors:
        # Does not need to be serialized.
        return

    def serialize_keras_tensor(x):
        # Serialize KerasTensor while converting
        # node indices to only include nodes relevant to `own_nodes`.
        if isinstance(x, backend.KerasTensor):
            operation, node_index, tensor_index = x._keras_history
            irrelevant_node_count = 0
            for i, node in enumerate(operation._inbound_nodes[:node_index]):
                node_key = make_node_key(operation, i)
                if node_key not in own_nodes:
                    irrelevant_node_count += 1
            x._keras_history = KerasHistory(
                operation, node_index - irrelevant_node_count, tensor_index
            )
            serialized = serialization_lib.serialize_keras_object(x)
            x._keras_history = KerasHistory(operation, node_index, tensor_index)
            return serialized
        return x

    args = node.arguments.args
    kwargs = node.arguments.kwargs

    args = tree.map_structure(serialize_keras_tensor, args)
    kwargs = tree.map_structure(serialize_keras_tensor, kwargs)
    return {
        "args": serialization_lib.serialize_keras_object(args),
        "kwargs": serialization_lib.serialize_keras_object(kwargs),
    }


def deserialize_node(node_data, created_layers):
    """Return (args, kwargs) for calling the node layer."""
    if not node_data:
        return [], {}

    if isinstance(node_data, list):
        # Legacy case.
        input_tensors = []
        for input_data in node_data:
            inbound_layer_name = input_data[0]
            inbound_node_index = input_data[1]
            inbound_tensor_index = input_data[2]
            if len(input_data) == 3:
                kwargs = {}
            elif len(input_data) == 4:
                kwargs = input_data[3]
            else:
                raise ValueError(
                    "Cannot deserialize the model (invalid config data?)"
                )

            if inbound_layer_name not in created_layers:
                raise ValueError(
                    "Invalid Functional model configuration. Missing node: "
                    f"{inbound_layer_name}"
                )
            inbound_layer = created_layers[inbound_layer_name]

            # Raise an error if the corresponding layer node
            # has not yet been created
            if len(inbound_layer._inbound_nodes) <= inbound_node_index:
                raise IndexError(
                    "Layer node index out of bounds.\n"
                    f"inbound_layer = {inbound_layer}\n"
                    "inbound_layer._inbound_nodes = "
                    f"{inbound_layer._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
            input_tensors.append(
                inbound_node.output_tensors[inbound_tensor_index]
            )
        return [unpack_singleton(input_tensors)], kwargs

    args = serialization_lib.deserialize_keras_object(node_data["args"])
    kwargs = serialization_lib.deserialize_keras_object(node_data["kwargs"])

    def convert_revived_tensor(x):
        if isinstance(x, backend.KerasTensor):
            history = x._pre_serialization_keras_history
            if history is None:
                return x
            layer = created_layers.get(history[0], None)
            if layer is None:
                raise ValueError(f"Unknown layer: {history[0]}")
            inbound_node_index = history[1]
            inbound_tensor_index = history[2]
            if len(layer._inbound_nodes) <= inbound_node_index:
                raise IndexError(
                    "Layer node index out of bounds.\n"
                    f"inbound_layer = {layer}\n"
                    f"inbound_layer._inbound_nodes = {layer._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = layer._inbound_nodes[inbound_node_index]
            return inbound_node.output_tensors[inbound_tensor_index]
        return x

    args = tree.map_structure(convert_revived_tensor, args)
    kwargs = tree.map_structure(convert_revived_tensor, kwargs)
    return args, kwargs


def is_input_keras_tensor(x):
    (
        operation,
        node_index,
        _,
    ) = x._keras_history
    node = operation._inbound_nodes[node_index]
    return node.is_input


def clone_single_keras_tensor(x):
    return backend.KerasTensor(
        shape=x.shape, dtype=x.dtype, sparse=x.sparse, name=f"{x.name}_clone"
    )


def clone_keras_tensors(tensors, kt_id_mapping):
    def swap(x):
        if not isinstance(x, backend.KerasTensor):
            return x
        if id(x) in kt_id_mapping:
            return kt_id_mapping[id(x)]
        new_x = clone_single_keras_tensor(x)
        kt_id_mapping[id(x)] = new_x
        return new_x

    return tree.map_structure(swap, tensors)


def find_nodes_by_inputs_and_outputs(inputs, outputs):
    nodes, _ = _build_map(inputs, outputs)
    return nodes


def clone_graph_nodes(inputs, outputs):
    """Clone the `Node` between the inputs and output tensors.

    This function is used to create a new functional model from any intermediate
    Keras tensors. The clone of the nodes mimic the behavior of reconstructing
    the functional graph network by re-executing all the `__call__()` methods.
    The cloned nodes will be appended to the layers.

    Note that a new `keras.Input` will be created for any items in the
    `inputs`

    Args:
    inputs: A nested structure of `KerasTensor` instances.
    outputs: A nested structure of `KerasTensor` instances.

    Returns:
        A pair of inputs and outputs, with cloned `KerasTensor` instances.
        They can be used to create a new functional model.
    """
    nodes_to_clone = find_nodes_by_inputs_and_outputs(inputs, outputs)
    cloned_inputs = []
    cloned_outputs = []
    # We not only need to create copies of Nodes (mimic the calls), also need to
    # clone Keras tensors to avoid the override of _keras_history attached on
    # the Keras tensor. The following dict is used to track any keras tensor we
    # cloned The key is the string ID of the original keras tensor, and value is
    # the cloned Keras tensor instance.
    kt_id_mapping = {}
    op_id_mapping = {}

    for kt_input in tree.flatten(inputs):
        if is_input_keras_tensor(kt_input):
            # For any existing Keras tensor from keras.Input, leave them as is.
            cloned_inputs.append(kt_input)
            kt_id_mapping[id(kt_input)] = kt_input
        else:
            # We need to create a new Keras tensor for any intermediate tensor
            original_op = kt_input._keras_history.operation
            optional = False
            if isinstance(original_op, InputLayer):
                optional = original_op.optional
            cloned_input = Input(
                batch_shape=kt_input.shape,
                dtype=kt_input.dtype,
                sparse=kt_input.sparse,
                name=f"{kt_input.name}CLONE",
                optional=optional,
            )
            cloned_inputs.append(cloned_input)
            kt_id_mapping[id(kt_input)] = cloned_input
            op_id_mapping[id(kt_input._keras_history[0])] = (
                cloned_input._keras_history[0]
            )
    cloned_inputs = tree.pack_sequence_as(inputs, cloned_inputs)

    for kt_output in tree.flatten(outputs):
        cpy = clone_single_keras_tensor(kt_output)
        # We reuse the _keras_history here, which contains the old information.
        cpy._keras_history = kt_output._keras_history
        cloned_outputs.append(cpy)
        kt_id_mapping[id(kt_output)] = cpy
    cloned_outputs = tree.pack_sequence_as(outputs, cloned_outputs)

    for node in nodes_to_clone:
        if id(node.operation) in op_id_mapping:
            operation = op_id_mapping[id(node.operation)]
        else:
            operation = node.operation
        # Clone any Keras tensor to avoid override of _keras_history
        # Or reuse an existing Keras tensor if it has already been cloned.
        output_copy = clone_keras_tensors(node.output_tensors, kt_id_mapping)
        if not isinstance(operation, InputLayer):
            call_args_copy = clone_keras_tensors(
                node.arguments.args, kt_id_mapping
            )
            call_kwargs_copy = clone_keras_tensors(
                node.arguments.kwargs, kt_id_mapping
            )
        else:
            call_args_copy = ()
            call_kwargs_copy = {}
        # Creating new nodes based on the existing node information.  Node wires
        # itself to inbound and outbound layers.  The Node constructor actually
        # updates this layer's self._inbound_nodes, sets _keras_history on the
        # outputs, and adds itself to the `_outbound_nodes` of the layers that
        # produced the inputs to this layer call.
        Node(
            operation,
            call_args=call_args_copy,
            call_kwargs=call_kwargs_copy,
            outputs=output_copy,
        )
    return cloned_inputs, cloned_outputs
