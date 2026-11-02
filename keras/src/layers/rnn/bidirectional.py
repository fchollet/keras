import copy

from keras.src import backend
from keras.src import ops
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.gru import GRU
from keras.src.layers.rnn.lstm import LSTM
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Bidirectional")
class Bidirectional(Layer):
    """Bidirectional wrapper for RNNs.

    Args:
        layer: `keras.layers.RNN` instance, such as
            `keras.layers.LSTM` or `keras.layers.GRU`.
            It could also be a `keras.layers.Layer` instance
            that meets the following criteria:
            1. Be a sequence-processing layer (accepts 3D+ inputs).
            2. Have a `go_backwards`, `return_sequences` and `return_state`
            attribute (with the same semantics as for the `RNN` class).
            3. Have an `input_spec` attribute.
            4. Implement serialization via `get_config()` and `from_config()`.
            Note that the recommended way to create new RNN layers is to write a
            custom RNN cell and use it with `keras.layers.RNN`, instead of
            subclassing `keras.layers.Layer` directly.
            When `return_sequences` is `True`, the output of the masked
            timestep will be zero regardless of the layer's original
            `zero_output_for_mask` value.
        merge_mode: Mode by which outputs of the forward and backward RNNs
            will be combined. One of `{"sum", "mul", "concat", "ave", None}`.
            If `None`, the outputs will not be combined,
            they will be returned as a list. Defaults to `"concat"`.
        backward_layer: Optional `keras.layers.RNN`,
            or `keras.layers.Layer` instance to be used to handle
            backwards input processing.
            If `backward_layer` is not provided, the layer instance passed
            as the `layer` argument will be used to generate the backward layer
            automatically.
            Note that the provided `backward_layer` layer should have properties
            matching those of the `layer` argument, in particular
            it should have the same values for `stateful`, `return_states`,
            `return_sequences`, etc. In addition, `backward_layer`
            and `layer` should have different `go_backwards` argument values.
            A `ValueError` will be raised if these requirements are not met.

    Call arguments:
        The call arguments for this layer are the same as those of the
        wrapped RNN layer. Beware that when passing the `initial_state`
        argument during the call of this layer, the first half in the
        list of elements in the `initial_state` list will be passed to
        the forward RNN call and the last half in the list of elements
        will be passed to the backward RNN call.

    Note: instantiating a `Bidirectional` layer from an existing RNN layer
    instance will not reuse the weights state of the RNN layer instance -- the
    `Bidirectional` layer will have freshly initialized weights.

    Examples:

    ```python
    model = Sequential([
        Input(shape=(5, 10)),
        Bidirectional(LSTM(10, return_sequences=True),
        Bidirectional(LSTM(10)),
        Dense(5, activation="softmax"),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # With custom backward layer
    forward_layer = LSTM(10, return_sequences=True)
    backward_layer = LSTM(10, activation='relu', return_sequences=True,
                          go_backwards=True)
    model = Sequential([
        Input(shape=(5, 10)),
        Bidirectional(forward_layer, backward_layer=backward_layer),
        Dense(5, activation="softmax"),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    """

    def __init__(
        self,
        layer,
        merge_mode="concat",
        weights=None,
        backward_layer=None,
        **kwargs,
    ):
        if not isinstance(layer, Layer):
            raise ValueError(
                "Please initialize `Bidirectional` layer with a "
                f"`keras.layers.Layer` instance. Received: {layer}"
            )
        if backward_layer is not None and not isinstance(backward_layer, Layer):
            raise ValueError(
                "`backward_layer` need to be a `keras.layers.Layer` "
                f"instance. Received: {backward_layer}"
            )
        if merge_mode not in ["sum", "mul", "ave", "concat", None]:
            raise ValueError(
                f"Invalid merge mode. Received: {merge_mode}. "
                "Merge mode should be one of "
                '{"sum", "mul", "ave", "concat", None}'
            )
        super().__init__(**kwargs)

        # Recreate the forward layer from the original layer config, so that it
        # will not carry over any state from the layer.
        config = serialization_lib.serialize_keras_object(layer)
        config["config"]["name"] = (
            f"forward_{utils.removeprefix(layer.name, 'forward_')}"
        )
        self.forward_layer = serialization_lib.deserialize_keras_object(config)

        if backward_layer is None:
            config = serialization_lib.serialize_keras_object(layer)
            config["config"]["go_backwards"] = True
            config["config"]["name"] = (
                f"backward_{utils.removeprefix(layer.name, 'backward_')}"
            )
            self.backward_layer = serialization_lib.deserialize_keras_object(
                config
            )
        else:
            self.backward_layer = backward_layer
        # Keep the use_cudnn attribute if defined (not serialized).
        if hasattr(layer, "use_cudnn"):
            self.forward_layer.use_cudnn = layer.use_cudnn
            self.backward_layer.use_cudnn = layer.use_cudnn
        self._verify_layer_config()

        def force_zero_output_for_mask(layer):
            # Force the zero_output_for_mask to be True if returning sequences.
            if getattr(layer, "zero_output_for_mask", None) is not None:
                layer.zero_output_for_mask = layer.return_sequences

        force_zero_output_for_mask(self.forward_layer)
        force_zero_output_for_mask(self.backward_layer)

        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[: nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2 :]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state
        self.supports_masking = False
        self.input_spec = layer.input_spec

    def _verify_layer_config(self):
        """Ensure the forward and backward layers have valid common property."""
        if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
            raise ValueError(
                "Forward layer and backward layer should have different "
                "`go_backwards` value. Received: "
                "forward_layer.go_backwards "
                f"{self.forward_layer.go_backwards}, "
                "backward_layer.go_backwards="
                f"{self.backward_layer.go_backwards}"
            )

        common_attributes = ("stateful", "return_sequences", "return_state")
        for a in common_attributes:
            forward_value = getattr(self.forward_layer, a)
            backward_value = getattr(self.backward_layer, a)
            if forward_value != backward_value:
                raise ValueError(
                    "Forward layer and backward layer are expected to have "
                    f'the same value for attribute "{a}", got '
                    f'"{forward_value}" for forward layer and '
                    f'"{backward_value}" for backward layer'
                )

    def _check_merge_widths(self, forward_units, backward_units):
        # "sum", "mul" and "ave" combine the two directions elementwise, so
        # both directions must produce the same width.
        if self.merge_mode not in ("sum", "mul", "ave"):
            return
        if None in (forward_units, backward_units):
            return
        if forward_units != backward_units:
            raise ValueError(
                "The forward and backward layers must produce the same "
                "number of units when `merge_mode` combines them "
                "elementwise. Received: "
                f"merge_mode={self.merge_mode}, "
                f"forward units={forward_units}, "
                f"backward units={backward_units}. Use "
                'merge_mode="concat" or merge_mode=None to combine '
                "layers of different sizes."
            )

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        # An explicit `backward_layer` may have a different number of units
        # from the forward layer, so the result cannot be derived from the
        # forward layer alone.
        forward_output_shape = self.forward_layer.compute_output_shape(
            sequences_shape
        )
        backward_output_shape = self.backward_layer.compute_output_shape(
            sequences_shape
        )

        if self.return_state:
            forward_output_shape, forward_state_shape = (
                forward_output_shape[0],
                forward_output_shape[1:],
            )
            backward_output_shape, backward_state_shape = (
                backward_output_shape[0],
                backward_output_shape[1:],
            )

        if self.merge_mode == "concat":
            output_shape = list(forward_output_shape)
            output_shape[-1] += backward_output_shape[-1]
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [forward_output_shape, backward_output_shape]
        else:
            self._check_merge_widths(
                forward_output_shape[-1], backward_output_shape[-1]
            )
            output_shape = forward_output_shape

        if self.return_state:
            if self.merge_mode is None:
                return (
                    tuple(output_shape)
                    + forward_state_shape
                    + backward_state_shape
                )
            return (
                tuple([output_shape])
                + forward_state_shape
                + backward_state_shape
            )
        return tuple(output_shape)

    def call(
        self,
        sequences,
        initial_state=None,
        mask=None,
        training=None,
    ):
        if self._can_attempt_fused_lstm(mask, initial_state):
            try:
                return self._call_fused_lstm(sequences, initial_state)
            except NotImplementedError:
                pass
        if self._can_attempt_fused_gru(mask, initial_state):
            try:
                return self._call_fused_gru(sequences, initial_state)
            except NotImplementedError:
                pass

        kwargs = {}
        if self.forward_layer._call_has_training_arg:
            kwargs["training"] = training
        if self.forward_layer._call_has_mask_arg:
            kwargs["mask"] = mask

        if initial_state is not None:
            # initial_states are not keras tensors, eg eager tensor from np
            # array.  They are only passed in from kwarg initial_state, and
            # should be passed to forward/backward layer via kwarg
            # initial_state as well.
            forward_inputs, backward_inputs = sequences, sequences
            half = len(initial_state) // 2
            forward_state = initial_state[:half]
            backward_state = initial_state[half:]
        else:
            forward_inputs, backward_inputs = sequences, sequences
            forward_state, backward_state = None, None

        y = self.forward_layer(
            forward_inputs, initial_state=forward_state, **kwargs
        )
        y_rev = self.backward_layer(
            backward_inputs, initial_state=backward_state, **kwargs
        )

        if self.return_state:
            states = tuple(y[1:] + y_rev[1:])
            y = y[0]
            y_rev = y_rev[0]

        self._check_merge_widths(y.shape[-1], y_rev.shape[-1])
        y = ops.cast(y, self.compute_dtype)
        y_rev = ops.cast(y_rev, self.compute_dtype)

        if self.return_sequences:
            y_rev = ops.flip(y_rev, axis=1)
        if self.merge_mode == "concat":
            output = ops.concatenate([y, y_rev], axis=-1)
        elif self.merge_mode == "sum":
            output = y + y_rev
        elif self.merge_mode == "ave":
            output = (y + y_rev) / 2
        elif self.merge_mode == "mul":
            output = y * y_rev
        elif self.merge_mode is None:
            output = (y, y_rev)
        else:
            raise ValueError(
                "Unrecognized value for `merge_mode`. "
                f"Received: {self.merge_mode}"
                'Expected one of {"concat", "sum", "ave", "mul"}.'
            )
        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return (output,) + states
        return output

    def _can_attempt_fused_lstm(self, mask, initial_state=None):
        # Layer-level preconditions for dispatching to
        # `backend.rnn.bidirectional_lstm`. Backends that don't support a
        # fused path (or don't have the runtime resources for one, e.g.
        # no GPU) raise `NotImplementedError` from the call itself, and
        # the caller falls back to the two-layer path.
        if mask is not None or self.stateful:
            return False
        if initial_state is not None and len(initial_state) != 4:
            return False

        fwd, bwd = self.forward_layer, self.backward_layer
        if not isinstance(fwd, LSTM) or not isinstance(bwd, LSTM):
            return False
        if fwd.use_cudnn not in (True, "auto"):
            return False
        if bwd.use_cudnn not in (True, "auto"):
            return False
        if fwd.cell.dropout or fwd.cell.recurrent_dropout:
            return False
        if bwd.cell.dropout or bwd.cell.recurrent_dropout:
            return False
        if fwd.cell.units != bwd.cell.units:
            return False
        if fwd.cell.activation is not bwd.cell.activation:
            return False
        if fwd.cell.recurrent_activation is not bwd.cell.recurrent_activation:
            return False
        if not fwd.cell.use_bias or not bwd.cell.use_bias:
            return False
        if not fwd.built or not bwd.built:
            return False
        return True

    def _call_fused_lstm(self, sequences, initial_state):
        fwd_cell = self.forward_layer.cell
        bwd_cell = self.backward_layer.cell
        units = fwd_cell.units

        sequences = ops.convert_to_tensor(sequences)
        batch_size = ops.shape(sequences)[0]

        if initial_state is None:
            zeros = ops.zeros((batch_size, units), dtype=sequences.dtype)
            fwd_h0, fwd_c0, bwd_h0, bwd_c0 = zeros, zeros, zeros, zeros
        else:
            fwd_h0, fwd_c0, bwd_h0, bwd_c0 = initial_state

        fwd_out, bwd_out = backend.rnn.bidirectional_lstm(
            sequences,
            fwd_h0,
            fwd_c0,
            bwd_h0,
            bwd_c0,
            mask=None,
            fwd_kernel=fwd_cell.kernel,
            fwd_recurrent_kernel=fwd_cell.recurrent_kernel,
            fwd_bias=fwd_cell.bias,
            bwd_kernel=bwd_cell.kernel,
            bwd_recurrent_kernel=bwd_cell.recurrent_kernel,
            bwd_bias=bwd_cell.bias,
            activation=fwd_cell.activation,
            recurrent_activation=fwd_cell.recurrent_activation,
            return_sequences=self.return_sequences,
            unroll=self.forward_layer.unroll,
        )
        fwd_last, fwd_seq, fwd_states = fwd_out
        bwd_last, bwd_seq, bwd_states = bwd_out

        if self.return_sequences:
            y, y_rev = fwd_seq, bwd_seq
        else:
            y, y_rev = fwd_last, bwd_last

        y = ops.cast(y, self.compute_dtype)
        y_rev = ops.cast(y_rev, self.compute_dtype)

        # The fused backend helper already returns backward outputs in
        # original time order, so the per-layer `ops.flip(y_rev)` that the
        # non-fused path applies is unnecessary here.
        if self.merge_mode == "concat":
            output = ops.concatenate([y, y_rev], axis=-1)
        elif self.merge_mode == "sum":
            output = y + y_rev
        elif self.merge_mode == "ave":
            output = (y + y_rev) / 2
        elif self.merge_mode == "mul":
            output = y * y_rev
        elif self.merge_mode is None:
            output = (y, y_rev)
        else:
            raise ValueError(
                "Unrecognized value for `merge_mode`. "
                f"Received: {self.merge_mode}. "
                'Expected one of {"concat", "sum", "ave", "mul"}.'
            )

        if self.return_state:
            states = tuple(fwd_states + bwd_states)
            if self.merge_mode is None:
                return output + states
            return (output,) + states
        return output

    def _can_attempt_fused_gru(self, mask, initial_state=None):
        # Layer-level preconditions for dispatching to
        # `backend.rnn.bidirectional_gru`. The structure matches the LSTM gate
        # above. The only differences are the layer type, the
        # `reset_after=True` requirement (the cuDNN GRU formulation), and
        # the state count (one h per direction, so length 2).
        if mask is not None or self.stateful:
            return False
        if initial_state is not None and len(initial_state) != 2:
            return False

        fwd, bwd = self.forward_layer, self.backward_layer
        if not isinstance(fwd, GRU) or not isinstance(bwd, GRU):
            return False
        if fwd.use_cudnn not in (True, "auto"):
            return False
        if bwd.use_cudnn not in (True, "auto"):
            return False
        if fwd.cell.dropout or fwd.cell.recurrent_dropout:
            return False
        if bwd.cell.dropout or bwd.cell.recurrent_dropout:
            return False
        if fwd.cell.units != bwd.cell.units:
            return False
        if fwd.cell.activation is not bwd.cell.activation:
            return False
        if fwd.cell.recurrent_activation is not bwd.cell.recurrent_activation:
            return False
        if not fwd.cell.use_bias or not bwd.cell.use_bias:
            return False
        if not fwd.cell.reset_after or not bwd.cell.reset_after:
            return False
        if not fwd.built or not bwd.built:
            return False
        return True

    def _call_fused_gru(self, sequences, initial_state):
        fwd_cell = self.forward_layer.cell
        bwd_cell = self.backward_layer.cell
        units = fwd_cell.units

        sequences = ops.convert_to_tensor(sequences)
        batch_size = ops.shape(sequences)[0]

        if initial_state is None:
            zeros = ops.zeros((batch_size, units), dtype=sequences.dtype)
            fwd_h0, bwd_h0 = zeros, zeros
        else:
            fwd_h0, bwd_h0 = initial_state

        fwd_out, bwd_out = backend.rnn.bidirectional_gru(
            sequences,
            fwd_h0,
            bwd_h0,
            mask=None,
            fwd_kernel=fwd_cell.kernel,
            fwd_recurrent_kernel=fwd_cell.recurrent_kernel,
            fwd_bias=fwd_cell.bias,
            bwd_kernel=bwd_cell.kernel,
            bwd_recurrent_kernel=bwd_cell.recurrent_kernel,
            bwd_bias=bwd_cell.bias,
            activation=fwd_cell.activation,
            recurrent_activation=fwd_cell.recurrent_activation,
            return_sequences=self.return_sequences,
            unroll=self.forward_layer.unroll,
            reset_after=fwd_cell.reset_after,
        )
        fwd_last, fwd_seq, fwd_states = fwd_out
        bwd_last, bwd_seq, bwd_states = bwd_out

        if self.return_sequences:
            y, y_rev = fwd_seq, bwd_seq
        else:
            y, y_rev = fwd_last, bwd_last

        y = ops.cast(y, self.compute_dtype)
        y_rev = ops.cast(y_rev, self.compute_dtype)

        if self.merge_mode == "concat":
            output = ops.concatenate([y, y_rev], axis=-1)
        elif self.merge_mode == "sum":
            output = y + y_rev
        elif self.merge_mode == "ave":
            output = (y + y_rev) / 2
        elif self.merge_mode == "mul":
            output = y * y_rev
        elif self.merge_mode is None:
            output = (y, y_rev)
        else:
            raise ValueError(
                "Unrecognized value for `merge_mode`. "
                f"Received: {self.merge_mode}. "
                'Expected one of {"concat", "sum", "ave", "mul"}.'
            )

        if self.return_state:
            states = tuple(fwd_states + bwd_states)
            if self.merge_mode is None:
                return output + states
            return (output,) + states
        return output

    def reset_states(self):
        # Compatibility alias.
        self.reset_state()

    def reset_state(self):
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        self.forward_layer.reset_state()
        self.backward_layer.reset_state()

    @property
    def states(self):
        if self.forward_layer.states and self.backward_layer.states:
            return tuple(self.forward_layer.states + self.backward_layer.states)
        return None

    def build(self, sequences_shape, initial_state_shape=None):
        if not self.forward_layer.built:
            self.forward_layer.build(sequences_shape)
        if not self.backward_layer.built:
            self.backward_layer.build(sequences_shape)

    def compute_mask(self, _, mask):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_sequences:
            if not self.merge_mode:
                output_mask = (mask, mask)
            else:
                output_mask = mask
        else:
            output_mask = (None, None) if not self.merge_mode else None

        if self.return_state and self.states is not None:
            state_mask = (None for _ in self.states)
            if isinstance(output_mask, list):
                return output_mask + state_mask * 2
            return (output_mask,) + state_mask * 2
        return output_mask

    def get_config(self):
        config = {"merge_mode": self.merge_mode}
        config["layer"] = serialization_lib.serialize_keras_object(
            self.forward_layer
        )
        config["backward_layer"] = serialization_lib.serialize_keras_object(
            self.backward_layer
        )
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Instead of updating the input, create a copy and use that.
        config = copy.deepcopy(config)

        config["layer"] = serialization_lib.deserialize_keras_object(
            config["layer"], custom_objects=custom_objects
        )
        # Handle (optional) backward layer instantiation.
        backward_layer_config = config.pop("backward_layer", None)
        if backward_layer_config is not None:
            backward_layer = serialization_lib.deserialize_keras_object(
                backward_layer_config, custom_objects=custom_objects
            )
            config["backward_layer"] = backward_layer
        # Instantiate the wrapper, adjust it and return it.
        layer = cls(**config)
        return layer
