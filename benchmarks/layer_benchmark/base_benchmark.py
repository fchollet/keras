import json
import time

import numpy as np
from absl import flags

import keras

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "benchmark_name",
    None,
    "The name of benchmark to run. If None, all benchmarks in the file will be "
    "run.",
)

flags.DEFINE_integer(
    "num_samples",
    1000,
    "Number of input data samples.",
)

flags.DEFINE_integer(
    "batch_size",
    20,
    "Batch size of data.",
)

flags.DEFINE_bool(
    "jit_compile",
    True,
    "If True, the benchmark will run with XLA compilation.",
)

flags.DEFINE_bool(
    "print_results_json",
    False,
    "Print a JSON line with each result.",
)


class BenchmarkMetricsCallback:
    def __init__(self, start_batch=1, stop_batch=None):
        self.start_batch = start_batch
        self.stop_batch = stop_batch

        self.state = {}

    def on_train_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput

    def on_predict_batch_begin(self, batch, logs=None):
        if batch == self.start_batch:
            self.state["benchmark_begin"] = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if batch == self.stop_batch:
            self.state["benchmark_end"] = time.time()
            throughput = (self.stop_batch - self.start_batch + 1) / (
                self.state["benchmark_end"] - self.state["benchmark_begin"]
            )
            self.state["throughput"] = throughput


class KerasCoreBenchmarkMetricsCallback(keras.callbacks.Callback):
    def __init__(self, start_batch=1, stop_batch=None):
        self._callback = BenchmarkMetricsCallback(start_batch, stop_batch)

    def on_train_batch_begin(self, batch, logs=None):
        self._callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._callback.on_train_batch_end(batch, logs)

    def on_predict_batch_begin(self, batch, logs=None):
        self._callback.on_predict_batch_begin(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        self._callback.on_predict_batch_end(batch, logs)


def _print_result_json(layer_name, mode, throughput):
    if FLAGS.print_results_json:
        result = {"layer": layer_name, "mode": mode, "throughput": throughput}
        print("BENCHMARK_RESULT:" + json.dumps(result))


class LayerBenchmark:
    def __init__(
        self,
        layer_name,
        init_args,
        input_shape,
        flat_call_inputs=True,
        jit_compile=True,
        keras_layer=None,
    ):
        self.layer_name = layer_name
        _keras_layer_class = getattr(keras.layers, layer_name)

        if keras_layer is None:
            self._keras_layer = _keras_layer_class(**init_args)
        else:
            self._keras_layer = keras_layer

        self.input_shape = input_shape
        self._keras_model = self._build_keras_model(
            input_shape, flat_call_inputs
        )

        self._keras_model.compile(
            loss="mse", optimizer="sgd", jit_compile=jit_compile
        )

        self.flat_call_inputs = flat_call_inputs
        self.jit_compile = jit_compile
        self.input_shape = input_shape

    def _build_keras_model(self, input_shape, flat_call_inputs=True):
        inputs = []
        if not isinstance(input_shape[0], (tuple, list)):
            input_shape = [input_shape]

        for shape in input_shape:
            inputs.append(keras.Input(shape=shape))

        if flat_call_inputs:
            outputs = self._keras_layer(*inputs)
        else:
            outputs = self._keras_layer(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def benchmark_predict(self, num_samples, batch_size, data=None):
        if data is None:
            # Generate default data if not provided.
            if isinstance(self.input_shape[0], (tuple, list)):
                # The layer has multiple inputs.
                data = []
                for data_shape in self.input_shape:
                    data_shape = [num_samples] + list(data_shape)
                    data.append(np.random.normal(size=data_shape))
            else:
                data_shape = [num_samples] + list(self.input_shape)
                data = np.random.normal(size=data_shape)

        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(stop_batch=num_iterations)

        self._keras_model.predict(
            data,
            batch_size=batch_size,
            callbacks=[callback],
        )

        keras_throughput = callback._callback.state["throughput"] * batch_size
        print(
            f"Keras 3 throughput of forward pass of {self.layer_name}: "
            f"{keras_throughput:.2f} samples/sec."
        )
        _print_result_json(self.layer_name, "predict", keras_throughput)

    def benchmark_train(self, num_samples, batch_size, data=None, label=None):
        if data is None:
            # Generate default data if not provided.
            if isinstance(self.input_shape[0], (tuple, list)):
                # The layer has multiple inputs.
                data = []
                for data_shape in self.input_shape:
                    data_shape = [num_samples] + list(data_shape)
                    data.append(np.random.normal(size=data_shape))
            else:
                data_shape = [num_samples] + list(self.input_shape)
                data = [np.random.normal(size=data_shape)]

        if label is None:
            # Generate default label if not provided.
            if self.flat_call_inputs:
                # Scale by a small factor to avoid zero gradients.
                label = (
                    keras.ops.convert_to_numpy(self._keras_layer(*data)) * 1.001
                )
            else:
                label = (
                    keras.ops.convert_to_numpy(self._keras_layer(data)) * 1.001
                )

        num_iterations = num_samples // batch_size - 1
        callback = KerasCoreBenchmarkMetricsCallback(stop_batch=num_iterations)

        self._keras_model.fit(
            data,
            label,
            batch_size=batch_size,
            callbacks=[callback],
        )

        keras_throughput = callback._callback.state["throughput"] * batch_size
        print(
            f"Keras 3 throughput of forward & backward pass of "
            f"{self.layer_name}: {keras_throughput:.2f} samples/sec."
        )
        _print_result_json(self.layer_name, "train", keras_throughput)
