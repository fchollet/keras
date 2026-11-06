import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.remat import RematScope
from keras.src.quantizers.capture import calibration_scope
from keras.src.utils import traceback_utils


class _Block(layers.Layer):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def call(self, inputs):
        return self.inner(inputs)


class CalibrationScopeTest(testing.TestCase):
    """The dispatch machinery runs a capture before every forward pass."""

    def _layer(self, mode=None):
        layer = layers.Dense(3)
        layer.build((None, 4))
        if mode is not None:
            layer.quantize(mode)
        return layer

    def _record(self, seen):
        def capture(inputs):
            seen.append(inputs)

        return capture

    @parameterized.named_parameters(
        ("filtered_traceback", True), ("plain_traceback", False)
    )
    def test_capture_sees_the_input_of_every_forward(self, filtering):
        # Both branches of `Operation.__call__` dispatch the same way.
        was_enabled = traceback_utils.is_traceback_filtering_enabled()
        try:
            if filtering:
                traceback_utils.enable_traceback_filtering()
            else:
                traceback_utils.disable_traceback_filtering()
            for mode in (None, "int8"):
                layer = self._layer(mode)
                x = np.ones((2, 4), "float32")
                seen = []
                with calibration_scope({layer: self._record(seen)}):
                    layer(x)
                    layer(inputs=x)
                self.assertEqual(len(seen), 2)
                # The capture receives the input as the forward receives it.
                self.assertTrue(ops.is_tensor(seen[0]))
                self.assertAllClose(seen[0], x)
                self.assertIsNone(layer._calibration_capture)
                layer(x)
                self.assertEqual(len(seen), 2)
        finally:
            if was_enabled:
                traceback_utils.enable_traceback_filtering()
            else:
                traceback_utils.disable_traceback_filtering()

    def test_capture_sees_an_input_passed_by_its_own_name(self):
        # A layer whose `call` names its input `x` can be called with
        # `x=`, as `Layer.__call__` resolves the first argument by name.
        class XLayer(layers.Layer):
            def call(self, x):
                return x

        layer = XLayer()
        x = np.ones((2, 4), "float32")
        seen = []
        with calibration_scope({layer: self._record(seen)}):
            layer(x=x)
        self.assertEqual(len(seen), 1)
        self.assertAllClose(seen[0], x)

    def test_capture_runs_under_stateless_call(self):
        layer = self._layer("int8")
        x = np.ones((2, 4), "float32")
        seen = []
        with calibration_scope({layer: self._record(seen)}):
            layer.stateless_call(
                [v.value for v in layer.trainable_variables],
                [v.value for v in layer.non_trainable_variables],
                x,
            )
        self.assertEqual(len(seen), 1)

    def test_capture_runs_under_rematerialization(self):
        for mode in (None, "int8"):
            with RematScope(mode="full"):
                layer = layers.Dense(3)
            layer.build((None, 4))
            if mode is not None:
                layer.quantize(mode)
            seen = []
            with calibration_scope({layer: self._record(seen)}):
                layer(np.ones((2, 4), "float32"))
            self.assertEqual(len(seen), 1)
            # It sees the concrete input, not a rematerialization tracer
            # (a JAX tracer has no value to compare).
            self.assertAllClose(seen[0], np.ones((2, 4)))

    def test_symbolic_calls_run_no_capture(self):
        layer = self._layer()
        block = _Block(layer)
        seen = []
        with calibration_scope({layer: self._record(seen)}):
            inputs = layers.Input((4,))
            models.Model(inputs, layer(inputs))
            # A symbolic call of an enclosing layer runs this layer on
            # backend placeholders for shape inference.
            inputs = layers.Input((4,))
            models.Model(inputs, block(inputs))
            block.compute_output_spec(backend.KerasTensor((2, 4)))
        self.assertEqual(seen, [])

    def test_one_capture_per_layer(self):
        layer = self._layer()
        with calibration_scope({layer: self._record([])}):
            with self.assertRaisesRegex(ValueError, "already inside"):
                with calibration_scope({layer: self._record([])}):
                    pass
        self.assertIsNone(layer._calibration_capture)

    def test_scope_is_torn_down_on_errors(self):
        first, second = self._layer(), self._layer()
        # An error inside the scope removes every capture.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with calibration_scope(
                {first: self._record([]), second: self._record([])}
            ):
                raise RuntimeError("boom")
        self.assertIsNone(first._calibration_capture)
        self.assertIsNone(second._calibration_capture)
        # An error while installing removes the captures installed so far.
        with calibration_scope({second: self._record([])}):
            with self.assertRaisesRegex(ValueError, "already inside"):
                with calibration_scope(
                    {first: self._record([]), second: self._record([])}
                ):
                    pass
            self.assertIsNone(first._calibration_capture)
            self.assertIsNotNone(second._calibration_capture)
