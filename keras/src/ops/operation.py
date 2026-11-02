import inspect
import math
import textwrap

from keras.src import backend
from keras.src import dtype_policies
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.common import remat
from keras.src.backend.common.keras_tensor import any_symbolic_tensors
from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.config import is_nnx_enabled
from keras.src.ops.node import Node
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils import python_utils
from keras.src.utils import traceback_utils
from keras.src.utils.naming import auto_name


@keras_export("keras.Operation")
class Operation(KerasSaveable):
    # A callable that receives the arguments of every forward pass, installed
    # by `keras.src.quantizers.capture.calibration_scope`; `None` otherwise.
    _calibration_capture = None

    def __init__(self, name=None):
        if name is None:
            name = auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                f"cannot contain character `/`. "
                f"Received: name={name} (of type {type(name)})"
            )
        self.name = name
        self._inbound_nodes = []
        self._outbound_nodes = []
        self._remat_mode = remat.get_current_remat_mode()

    @traceback_utils.filter_traceback
    def __call__(self, *args, **kwargs):
        if traceback_utils.is_traceback_filtering_enabled():
            try:
                if any_symbolic_tensors(args, kwargs):
                    return self.symbolic_call(*args, **kwargs)
                return self._dispatch_call(*args, **kwargs)
            except Exception as e:
                if not getattr(e, "_keras_call_info_injected", False):
                    raise traceback_utils.inject_argument_info_in_error(
                        e,
                        self.call,
                        args,
                        kwargs,
                        object_name=(f"{self.__class__.__name__}.call()"),
                    ) from None
                raise

        # Plain flow.
        if any_symbolic_tensors(args, kwargs):
            return self.symbolic_call(*args, **kwargs)
        return self._dispatch_call(*args, **kwargs)

    def _dispatch_call(self, *args, **kwargs):
        """Runs the forward pass this operation dispatches to.

        This is the one place that selects the forward: `quantized_call`
        under a quantization mode, `call` otherwise, wrapped for
        rematerialization when enabled. `__call__` and `stateless_call`
        both route through it. A calibration capture registered on the
        operation receives the forward's arguments first, so it sees every
        dispatched forward, whichever one that is. The capture runs outside
        the rematerialization wrapper, so it sees the arguments the dispatch
        receives rather than remat tracers. Only forwards that arrive
        through `__call__` or `stateless_call` are observed. Forwards
        traced for shape inference (`compute_output_spec`, which a symbolic
        call of an enclosing layer runs) are not.
        """
        capture = self._calibration_capture
        if capture is not None and not in_symbolic_scope():
            capture(*args, **kwargs)
        if getattr(self, "quantization_mode", None) is not None:
            call_fn = self.quantized_call
        else:
            call_fn = self.call
        if self._remat_mode:
            call_fn = self.rematerialized_call(call_fn, *args, **kwargs)
        return call_fn(*args, **kwargs)

    def symbolic_call(self, *args, **kwargs):
        # Perform shape/dtype inference.
        outputs = self.compute_output_spec(*args, **kwargs)
        # Record a new node in the operations graph.
        # The Node wires itself to inbound and outbound ops.  The
        # Node constructor updates this op's self._inbound_nodes,
        # sets _keras_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the ops that produced the inputs to this
        # call.
        Node(
            operation=self, call_args=args, call_kwargs=kwargs, outputs=outputs
        )
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def quantized_call(self, *args, **kwargs):
        raise NotImplementedError

    def rematerialized_call(self, fn, *args, **kwargs):
        """Enable rematerialization dynamically for an operation's call method.

        Args:
            fn: The original `call` or `quantized_call` method of an operation.

        Returns:
            Rematerialized method.
        """
        if not self._remat_mode:
            return fn

        def compute_size(x):
            return (
                math.prod([d or 1 for d in x.shape])
                if isinstance(x, KerasTensor)
                else 0
            )

        # Full rematerialization
        if self._remat_mode.mode == "full":
            return remat.remat(fn)

        # Apply rematerialization based on output size threshold
        elif self._remat_mode.mode == "larger_than":
            output_spec = self.compute_output_spec(*args, **kwargs)
            output_size = sum(
                tree.flatten(tree.map_structure(compute_size, output_spec))
            )
            if (
                output_size
                and output_size > self._remat_mode.output_size_threshold
            ):
                return remat.remat(fn)
        return fn

    def compute_output_spec(self, *args, **kwargs):
        try:
            return backend.compute_output_spec(self.call, *args, **kwargs)
        except Exception as e:
            new_e = e.__class__(
                "Could not automatically infer the output shape / dtype of "
                f"'{self.name}' (of type {self.__class__.__name__}). "
                f"Either the `{self.__class__.__name__}.call()` method "
                f"is incorrect, or you need to implement the "
                f"`{self.__class__.__name__}.compute_output_spec() / "
                "compute_output_shape()` method. "
                f"Error encountered:\n\n{e}"
            )
            raise new_e.with_traceback(e.__traceback__) from None

    def __new__(cls, *args, **kwargs):
        """We override __new__ to saving serializable constructor arguments.

        These arguments are used to auto-generate an object serialization
        config, which enables user-created subclasses to be serializable
        out of the box in most cases without forcing the user
        to manually implement `get_config()`.
        """
        instance = super(Operation, cls).__new__(cls)
        if backend.backend() == "jax" and is_nnx_enabled():
            from flax import nnx

            try:
                vars(instance)["_pytree__state"] = nnx.pytreelib.PytreeState()
            except AttributeError:
                vars(instance)["_object__state"] = nnx.object.ObjectState()

        # Generate a config to be returned by default by `get_config()`.
        auto_config = True

        signature = inspect.signature(cls.__init__)
        argspec = inspect.getfullargspec(cls.__init__)

        try:
            bound_parameters = signature.bind(None, *args, **kwargs)
        except TypeError:
            # Raised by signature.bind when the supplied args and kwargs
            # do not match the signature.
            auto_config = False

        if auto_config and any(
            [
                param.kind == inspect.Parameter.POSITIONAL_ONLY
                for name, param in signature.parameters.items()
                if name != argspec.args[0]
            ]
        ):
            # cls.__init__ takes positional only arguments, which
            # cannot be restored via cls(**config)
            auto_config = False
            # Create variable to show appropriate warning in get_config.
            instance._auto_config_error_args = True

        if auto_config:
            # Include default values in the config.
            bound_parameters.apply_defaults()
            # Extract all arguments as a dictionary.
            kwargs = bound_parameters.arguments
            # Expand variable kwargs argument.
            kwargs |= kwargs.pop(argspec.varkw, {})
            # Remove first positional argument, self.
            kwargs.pop(argspec.args[0])
            # Remove argument "name", as it is provided by get_config.
            kwargs.pop("name", None)
            if argspec.varargs is not None:
                # Varargs cannot be meaningfully converted to a dictionary.
                varargs = kwargs.pop(argspec.varargs)
                if len(varargs) > 0:
                    auto_config = False
                    # Store variable to show appropriate warning in get_config.
                    instance._auto_config_error_args = True

        # For safety, we only rely on auto-configs for a small set of
        # serializable types.
        supported_types = (str, int, float, bool, type(None))
        try:
            flat_arg_values = tree.flatten(kwargs)
            for value in flat_arg_values:
                if not isinstance(value, supported_types):
                    auto_config = False
                    break
        except TypeError:
            auto_config = False
        try:
            instance._lock = False
            if auto_config:
                from keras.src.saving import serialization_lib

                instance._auto_config = serialization_lib.SerializableDict(
                    **kwargs
                )
            else:
                instance._auto_config = None
            instance._lock = True
        except RecursionError:
            # Setting an instance attribute in __new__ has the potential
            # to trigger an infinite recursion if a subclass overrides
            # setattr in an unsafe way.
            pass
        return instance

    @python_utils.default
    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-instantiate it.
        """
        config = {
            "name": self.name,
        }

        if not python_utils.is_default(self.get_config):
            # In this case the subclass implements get_config()
            return config

        # In this case the subclass doesn't implement get_config():
        # Let's see if we can autogenerate it.
        if getattr(self, "_auto_config", None) is not None:
            config.update(self._auto_config.config)
            init_params = inspect.signature(self.__init__).parameters
            init_has_name = "name" in init_params
            init_has_kwargs = (
                "kwargs" in init_params
                and init_params["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
            )
            if not init_has_name and not init_has_kwargs:
                # We can't pass `name` back to `__init__`, remove it.
                config.pop("name", None)
            return config
        else:
            example_str = """
            class CustomLayer(keras.layers.Layer):
                def __init__(self, arg1, arg2, **kwargs):
                    super().__init__(**kwargs)
                    self.arg1 = arg1
                    self.arg2 = arg2

                def get_config(self):
                    config = super().get_config()
                    config.update({
                        "arg1": self.arg1,
                        "arg2": self.arg2,
                    })
                    return config
            """
            if getattr(self, "_auto_config_error_args", False):
                raise NotImplementedError(
                    textwrap.dedent(
                        f"""
            Object {self.__class__.__name__} was created by passing
            positional only or variadic positional arguments (e.g.,
            `*args`) to `__init__()`, which is not supported by the
            automatic config generation. Please remove all positional
            only and variadic arguments from `__init__()`
            or override `get_config()` and `from_config()` to make
            the object serializatble.

            Example:

            {example_str}"""
                    )
                )
            else:
                raise NotImplementedError(
                    textwrap.dedent(
                        f"""
            Object {self.__class__.__name__} was created by passing
            non-serializable argument values in `__init__()`,
            and therefore the object must override `get_config()` in
            order to be serializable. Please implement `get_config()`.

            Example:

            {example_str}"""
                    )
                )

    @classmethod
    def from_config(cls, config):
        """Creates an operation from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same operation from the config dictionary.

        Note: If you override this method, you might receive a serialized dtype
        config, which is a `dict`. You can deserialize it as follows:

        ```python
        if "dtype" in config and isinstance(config["dtype"], dict):
            policy = dtype_policies.deserialize(config["dtype"])
        ```

        Args:
            config: A Python dictionary, typically the output of `get_config`.

        Returns:
            An operation instance.
        """
        # Explicitly deserialize dtype config if needed. This enables users to
        # directly interact with the instance of `DTypePolicy`.
        if "dtype" in config and isinstance(config["dtype"], dict):
            config = config.copy()
            policy = dtype_policies.deserialize(config["dtype"])
            if (
                not isinstance(policy, dtype_policies.DTypePolicyMap)
                and policy.quantization_mode is None
            ):
                # For backward compatibility, we use a str (`name`) for
                # `DTypePolicy`
                policy = policy.name
            config["dtype"] = policy
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using "
                f"config={config}.\n\nException encountered: {e}"
            )

    def __repr__(self):
        return f"<Operation name={self.name}>"

    @property
    def input(self):
        """Retrieves the input tensor(s) of a symbolic operation.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Input tensor or list of input tensors.
        """
        return self._get_node_attribute_at_index(0, "input_tensors", "input")

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Output tensor or list of output tensors.
        """
        return self._get_node_attribute_at_index(0, "output_tensors", "output")

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the properties:
        - output
        - input

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The operation's attribute `attr` at the node of index `node_index`.
        """
        if not self._inbound_nodes:
            raise AttributeError(
                f"The layer {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the operation has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values

    def _obj_type(self):
        return "Operation"

    # Hooks for backend layer classes
    def _post_build(self):
        """Can be overridden for per backend post build actions."""
        pass

    def _setattr_hook(self, name, value):
        """Can be overridden for per backend post build actions."""
        return name, value

    def _post_track_variable(self, variable):
        """Can be overridden for per backend post track actions."""
        pass

    def _post_untrack_variable(self, variable):
        """Can be overridden for per backend post untrack actions."""
        pass
