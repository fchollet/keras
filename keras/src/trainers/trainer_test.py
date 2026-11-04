from unittest import mock

import jax
import numpy as np
import pytest
from absl.testing import parameterized

import keras
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import losses
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import testing
from keras.src.backend import config
from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.callbacks.callback import Callback
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.models.model import Trainer
from keras.src.optimizers.rmsprop import RMSprop
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product
from keras.src.testing.test_utils import x64_mode_scope
from keras.src.trainers.data_adapters import py_dataset_adapter


# A model is just a layer mixed in with a Trainer.
class ExampleModel(Trainer, layers.Dense):
    def __init__(self, units):
        layers.Dense.__init__(
            self,
            units=units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        Trainer.__init__(self)


class CustomTrainTestStepModel(ExampleModel):
    def train_step(self, data):
        logs = super().train_step(data)
        logs["my_custom_metric"] = 10.0
        return logs

    def test_step(self, data):
        logs = super().test_step(data)
        logs["my_custom_metric"] = 5.0
        return logs


class JaxCustomTrainTestStepModel(ExampleModel):
    def train_step(self, state, data):
        logs, state = super().train_step(state, data)
        logs["my_custom_metric"] = 10.0
        return logs, state

    def test_step(self, state, data):
        logs, state = super().test_step(state, data)
        logs["my_custom_metric"] = 5.0
        return logs, state


class StructModel(Trainer, layers.Layer):
    def __init__(self, units):
        layers.Layer.__init__(self)
        Trainer.__init__(self)
        self.dense_1 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        self.dense_2 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )

    def call(self, x):
        return {
            "y_one": self.dense_1(x["x_one"]),
            "y_two": self.dense_2(x["x_two"]),
        }


class ListInputModel(Trainer, layers.Layer):
    def __init__(self, units):
        layers.Layer.__init__(self)
        Trainer.__init__(self)
        self.dense_1 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        self.dense_2 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )

    def call(self, x):
        if not isinstance(x, (list, tuple)):
            raise ValueError("x must be a list or tuple")
        return self.dense_1(x[0]) + self.dense_2(x[1])


class ListOutputModel(Trainer, layers.Layer):
    def __init__(self, units):
        layers.Layer.__init__(self)
        Trainer.__init__(self)
        self.dense_1 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )
        self.dense_2 = layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=initializers.Ones(),
        )

    def call(self, x):
        return [self.dense_1(x), self.dense_2(x)]


class TrainingTestingLayer(Trainer, layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self, **kwargs)
        Trainer.__init__(self)

    def call(self, x, training=False):
        if training:
            return x
        return x * 0


class TestPyDataset(py_dataset_adapter.PyDataset):
    def __init__(self, infinite=False, **kwargs):
        super().__init__(**kwargs)
        self.infinite = infinite

    @property
    def num_batches(self):
        return None if self.infinite else 20

    def __getitem__(self, idx):
        return np.ones((5, 4)), np.zeros((5, 3))


def create_dataset(dataset_type, dataset_kwargs):
    if dataset_type == "np_array":
        return np.ones((100, 4)), np.zeros((100, 3))
    elif dataset_type == "native_array":
        return ops.ones((100, 4)), ops.zeros((100, 3))
    elif dataset_type == "py_dataset":
        return TestPyDataset(**dataset_kwargs), None
    elif dataset_type == "tf_dataset":
        import tensorflow as tf

        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.ones((100, 4)), tf.zeros((100, 3)))
        ).batch(5)
        if dataset_kwargs.get("infinite", False):
            dataset = dataset.repeat()
        return dataset, None
    elif dataset_type == "torch_dataloader":
        import torch

        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for _ in range(20):
                    yield torch.ones((5, 4)), torch.zeros((5, 3))

        class TestIterableDatasetWithLen(TestIterableDataset):
            def __len__(self):
                return 20

        if dataset_kwargs.get("iterable", False):
            if dataset_kwargs.get("has_len", False):
                dataset = TestIterableDatasetWithLen()
            else:
                dataset = TestIterableDataset()
            return torch.utils.data.DataLoader(dataset), None
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.ones((100, 4)), torch.zeros((100, 3))
            )
            return torch.utils.data.DataLoader(dataset, batch_size=5), None
    elif dataset_type == "generator":

        def generate_finite():
            for _ in range(20):
                yield ops.ones((5, 4)), ops.zeros((5, 3))

        def generate_infinite():
            while True:
                yield ops.ones((5, 4)), ops.zeros((5, 3))

        if dataset_kwargs.get("infinite", False):
            return generate_infinite(), None
        else:
            return generate_finite(), None
    elif dataset_type == "grain_dataset":
        import grain

        class TestIterableDataset(grain.sources.RandomAccessDataSource):
            def __init__(self):
                super().__init__()
                self.x = np.ones((100, 4)).astype("float32")
                self.y = np.zeros((100, 3)).astype("float32")

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        if dataset_kwargs.get("use_dataloader", False):
            source = TestIterableDataset()
            dataloader = grain.DataLoader(
                data_source=source,
                sampler=grain.samplers.IndexSampler(len(source), num_epochs=1),
                operations=[grain.transforms.Batch(batch_size=5)],
            )
            return dataloader, None
        else:
            dataset = grain.MapDataset.source(TestIterableDataset())
            if dataset_kwargs.get("has_len", False):
                dataset = dataset.to_iter_dataset()
            dataset = dataset.batch(5)
            return dataset, None
    else:
        raise ValueError(f"Invalid dataset type {dataset_type}")


def sparse_generator(generator_type):
    if generator_type == "scipy":
        import scipy

        for _ in range(4):
            x = scipy.sparse.random(2, 4, density=0.25, dtype="float32")
            y = np.random.rand(2, 3).astype("float32")
            yield x, y
    elif generator_type == "tf":
        import tensorflow as tf

        for _ in range(4):
            x = tf.random.uniform((2, 4), dtype="float32")
            x = tf.sparse.from_dense(tf.nn.dropout(x, 0.25))
            y = tf.random.uniform((2, 3), dtype="float32")
            yield x, y
    elif generator_type == "jax":
        import jax
        import jax.experimental.sparse as jax_sparse

        for _ in range(4):
            seed = jax.random.PRNGKey(0)
            x = jax_sparse.random_bcoo(seed, (2, 4), dtype="float32", nse=0.25)
            y = jax.random.uniform(seed, (2, 3), dtype="float32")
            yield x, y
    else:
        raise ValueError(f"Invalid generator type {generator_type}")


class EpochAgnosticMeanSquaredError(metrics.MeanSquaredError):
    def __init__(self):
        super().__init__(name="mse")
        super().reset_state()

    def reset_state(self):
        # prevent reset at each starting epoch
        pass


class StepObserver(Callback):
    def __init__(self):
        super().__init__()
        self.begin_count = 0
        self.end_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0
        self.batch_loss_history = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin_count += 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end_count += 1

    def on_batch_begin(self, batch, logs=None):
        self.begin_count += 1

    def on_batch_end(self, batch, logs=None):
        self.end_count += 1
        self.batch_loss_history.append(logs["mse"])


class StepCount(Callback):
    def __init__(self, steps_per_execution=1):
        super().__init__()
        self.begin_count = 0
        self.end_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0
        self.steps_per_execution = steps_per_execution

    def on_epoch_begin(self, epoch, logs=None):
        self.begin_count = 0
        self.end_count = 0
        self.epoch_begin_count += 1

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end_count += 1

    def on_batch_begin(self, batch, logs=None):
        if batch != self.begin_count * self.steps_per_execution:
            raise ValueError("Batch index is not correct")
        self.begin_count += 1

    def on_batch_end(self, batch, logs=None):
        self.end_count += 1
        if batch != self.end_count * self.steps_per_execution - 1:
            raise ValueError("Batch index is not correct")


class TestTrainer(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_metric_tracking(self):
        class ModelWithMetric(Trainer, layers.Dense):
            def __init__(self, units):
                layers.Dense.__init__(
                    self,
                    units=units,
                    use_bias=False,
                    kernel_initializer=initializers.Ones(),
                )
                Trainer.__init__(self)
                self.my_metric = metrics.MeanSquaredError(name="my_metric")

        model = ModelWithMetric(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        x = np.ones((2, 4))
        y = np.zeros((2, 3))
        # Fit the model to make sure compile_metrics are built
        model.fit(x, y, batch_size=2, epochs=1)

        # The model should have 3 metrics: loss_tracker, compile_metrics,
        # my_metric.
        self.assertEqual(len(model.metrics), 3)
        self.assertEqual(model.metrics[0], model._loss_tracker)
        self.assertEqual(model.metrics[1], model._compile_metrics)
        self.assertEqual(model.metrics[2], model.my_metric)

        # All metrics should have their weights created
        self.assertEqual(len(model._loss_tracker.variables), 2)
        self.assertEqual(len(model._compile_metrics.variables), 2)
        self.assertEqual(len(model.my_metric.variables), 2)

        # And those weights are tracked at the model level
        self.assertEqual(len(model.metrics_variables), 6)
        self.assertLen(model.non_trainable_variables, 0)

        # Models with only weighted_metrics should have the same 3 metrics
        model_weighted = ModelWithMetric(units=3)
        model_weighted.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            weighted_metrics=[metrics.MeanSquaredError()],
        )
        model_weighted.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            sample_weight=np.ones(2),
        )
        self.assertEqual(len(model_weighted.metrics), 3)

    def test_nested_trainer_metrics(self):
        # https://github.com/keras-team/keras/issues/20188
        model = ExampleModel(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        self.assertLen(model.metrics, 2)
        self.assertEqual(model.metrics[0], model._loss_tracker)
        self.assertEqual(model.metrics[1], model._compile_metrics)

        inputs = keras.Input((4,))
        outputs = model(inputs)
        outputs = layers.Dense(8)(outputs)
        new_model = models.Model(inputs, outputs)
        new_model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        self.assertLen(new_model.metrics, 2)
        self.assertEqual(new_model.metrics[0], new_model._loss_tracker)
        self.assertEqual(new_model.metrics[1], new_model._compile_metrics)

    def test_nested_trainer_metrics_without_compile(self):
        model = ExampleModel(units=3)
        self.assertLen(model.metrics, 0)

        inputs = keras.Input((4,))
        outputs = model(inputs)
        outputs = layers.Dense(8)(outputs)
        new_model = models.Model(inputs, outputs)
        new_model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        self.assertLen(new_model.metrics, 2)
        self.assertEqual(new_model.metrics[0], new_model._loss_tracker)
        self.assertEqual(new_model.metrics[1], new_model._compile_metrics)

    def test_multiple_compiles(self):
        # https://github.com/keras-team/keras/issues/20474
        model1 = ExampleModel(units=3)
        model2 = ExampleModel(units=3)
        model1.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        # Combine these 2 models into `combined`.
        inputs = keras.Input(shape=(4,))
        x = model1(inputs)
        outputs = model2(x)
        combined = models.Model(inputs, outputs)
        combined.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        self.assertLen(model1.metrics, 2)
        self.assertIsNotNone(model1._loss_tracker)
        self.assertEqual(model1.metrics[0], model1._loss_tracker)
        self.assertEqual(model1.metrics[1], model1._compile_metrics)

        # `combined.metrics` will not include `model1.metrics`.
        self.assertLen(combined.metrics, 2)
        self.assertIsNotNone(combined._loss_tracker)
        self.assertEqual(combined.metrics[0], combined._loss_tracker)
        self.assertEqual(combined.metrics[1], combined._compile_metrics)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="torch backend runs in eager mode for jit_compile='auto'",
    )
    def test_compile_eager_vs_jit_torch(self):
        model = ExampleModel(units=3)
        model.compile(jit_compile="auto")
        # torch trainer en/disables torch.compile only based on the value of
        # model.jit_compile (not model.run_eagerly)
        self.assertFalse(model.run_eagerly)
        self.assertFalse(model.jit_compile)

    @parameterized.named_parameters(
        [
            ("eager", True, False, False),
            ("graph_fn", False, False, False),
            ("jit", False, True, False),
            ("steps_per_epoch_eager", True, False, True),
            ("steps_per_epoch_graph_fn", False, False, True),
            ("steps_per_epoch_jit", False, True, True),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_fit_flow(self, run_eagerly, jit_compile, use_steps_per_epoch):
        model = ExampleModel(units=3)
        epochs = 3
        batch_size = 20
        steps_per_epoch = 7
        dataset_size = batch_size * (steps_per_epoch - 2)
        x = np.ones((dataset_size, 4))
        y = np.zeros((dataset_size, 3))

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch if use_steps_per_epoch else None,
            epochs=epochs,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("mean_squared_error", history)
        self.assertAllClose(
            history["mean_squared_error"],
            [14.5, 11.5, 8.5],
            atol=1.0,  # TODO: results vary across backends
        )

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "np_array",
                "dataset_type": "np_array",
                "fit_kwargs": {"batch_size": 5},
            },
            {
                "testcase_name": "native_array",
                "dataset_type": "native_array",
                "fit_kwargs": {"batch_size": 5},
            },
            {
                "testcase_name": "py_dataset",
                "dataset_type": "py_dataset",
            },
            {
                "testcase_name": "py_dataset_cw",
                "dataset_type": "py_dataset",
                "fit_kwargs": {"class_weight": {0: 1, 1: 2}},
            },
            {
                "testcase_name": "py_dataset_infinite",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"infinite": True},
                "fit_kwargs": {"steps_per_epoch": 20},
            },
            {
                "testcase_name": "py_dataset_infinite_cw",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"infinite": True},
                "fit_kwargs": {
                    "steps_per_epoch": 20,
                    "class_weight": {0: 1, 1: 2},
                },
            },
            {
                "testcase_name": "py_dataset_multithreading",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"workers": 2},
            },
            {
                "testcase_name": "py_dataset_multithreading_cw",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"workers": 2},
                "fit_kwargs": {"class_weight": {0: 1, 1: 2}},
            },
            {
                "testcase_name": "py_dataset_multithreading_infinite",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"infinite": True, "workers": 2},
                "fit_kwargs": {"steps_per_epoch": 20},
            },
            {
                "testcase_name": "py_dataset_multiprocessing",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"workers": 2, "use_multiprocessing": True},
            },
            {
                "testcase_name": "py_dataset_multiprocessing_cw",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {"workers": 2, "use_multiprocessing": True},
                "fit_kwargs": {"class_weight": {0: 1, 1: 2}},
            },
            {
                "testcase_name": "py_dataset_multiprocessing_infinite",
                "dataset_type": "py_dataset",
                "dataset_kwargs": {
                    "infinite": True,
                    "workers": 2,
                    "use_multiprocessing": True,
                },
                "fit_kwargs": {"steps_per_epoch": 20},
            },
            {
                "testcase_name": "tf_dataset",
                "dataset_type": "tf_dataset",
            },
            {
                "testcase_name": "tf_dataset_infinite",
                "dataset_type": "tf_dataset",
                "dataset_kwargs": {"infinite": True},
                "fit_kwargs": {"steps_per_epoch": 20},
            },
            {
                "testcase_name": "torch_dataloader_tensor",
                "dataset_type": "torch_dataloader",
            },
            {
                "testcase_name": "torch_dataloader_iterable",
                "dataset_type": "torch_dataloader",
                "dataset_kwargs": {"iterable": True, "has_len": False},
            },
            {
                "testcase_name": "torch_dataloader_iterable_with_len",
                "dataset_type": "torch_dataloader",
                "dataset_kwargs": {"iterable": True, "has_len": True},
            },
            {
                "testcase_name": "generator",
                "dataset_type": "generator",
            },
            {
                "testcase_name": "generator_infinite",
                "dataset_type": "generator",
                "dataset_kwargs": {"infinite": True},
                "fit_kwargs": {"steps_per_epoch": 20},
            },
            {
                "testcase_name": "grain_dataset",
                "dataset_type": "grain_dataset",
                "dataset_kwargs": {"has_len": False},
            },
            {
                "testcase_name": "grain_dataset_with_len",
                "dataset_type": "grain_dataset",
                "dataset_kwargs": {"has_len": True},
            },
            {
                "testcase_name": "grain_dataloader",
                "dataset_type": "grain_dataset",
                "dataset_kwargs": {"use_dataloader": True},
            },
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_fit_with_data_adapter(
        self, dataset_type, dataset_kwargs={}, fit_kwargs={}
    ):
        jit_compile = True
        if dataset_kwargs.get("use_multiprocessing", False):
            if backend.backend() == "jax":
                pytest.skip("Multiprocessing not supported with JAX backend")
            elif backend.backend() == "tensorflow":
                pytest.skip("Multiprocessing hangs on Tensorflow")
        if dataset_type == "grain_dataset" and backend.backend() == "torch":
            # Grain datasets are not supported with torch + jit_compile.
            jit_compile = False

        model = ExampleModel(units=3)
        optimizer = optimizers.Adagrad()
        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            jit_compile=jit_compile,
        )
        x, y = create_dataset(dataset_type, dataset_kwargs)
        model.fit(x, y, epochs=3, **fit_kwargs)

    @parameterized.named_parameters(
        [
            ("eager", True, False, False),
            ("graph_fn", False, False, False),
            ("jit", False, True, False),
            ("steps_per_epoch_eager", True, False, True),
            ("steps_per_epoch_graph_fn", False, False, True),
            ("steps_per_epoch_jit", False, True, True),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_fit_with_val_split(
        self, run_eagerly, jit_compile, use_steps_per_epoch
    ):
        model = ExampleModel(units=3)
        epochs = 3
        batch_size = 20
        steps_per_epoch = 7
        dataset_size = batch_size * (steps_per_epoch - 2)
        x = np.ones((dataset_size, 4))
        y = np.zeros((dataset_size, 3))

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch if use_steps_per_epoch else None,
            epochs=epochs,
            validation_split=0.2,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("val_loss", history)

        # Test with backend-native tensors.
        x = ops.ones((dataset_size, 4))
        y = ops.zeros((dataset_size, 3))
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch if use_steps_per_epoch else None,
            epochs=epochs,
            validation_split=0.2,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("val_loss", history)

    @pytest.mark.requires_trainable_backend
    def test_fit_with_custom_train_step(self):
        if backend.backend() == "jax":
            model = JaxCustomTrainTestStepModel(units=3)
        else:
            model = CustomTrainTestStepModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        history = model.fit(x, y, batch_size=batch_size)
        history = history.history
        self.assertIn("loss", history)
        self.assertIn("mean_squared_error", history)
        self.assertAllClose(history["my_custom_metric"], [10.0])

    @parameterized.named_parameters(
        named_product(
            generator_type=["tf", "jax", "scipy"], mode=["eager", "graph"]
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_fit_sparse(self, generator_type, mode):
        model = ExampleModel(units=3)
        optimizer = optimizers.Adagrad()
        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=False,
        )
        dataset = sparse_generator(generator_type)

        sparse_variable_updates = False

        def mock_optimizer_assign(variable, value):
            nonlocal sparse_variable_updates
            if value.__class__.__name__ == "IndexedSlices":
                sparse_variable_updates = True

        with mock.patch.object(
            optimizer, "assign_sub", autospec=True
        ) as optimizer_assign_sub:
            optimizer_assign_sub.side_effect = mock_optimizer_assign
            model.fit(dataset)

        # JAX does not produce sparse gradients the way we use it.
        if backend.backend() != "jax":
            # Verify tensors did not get densified along the way.
            self.assertTrue(sparse_variable_updates)

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_evaluate_flow(self, run_eagerly, jit_compile):
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        output = model.evaluate(x, y, batch_size=batch_size)
        self.assertAllClose(output, [16.0, 16.0])
        output = model.evaluate(x, y, batch_size=batch_size, return_dict=True)
        self.assertIsInstance(output, dict)
        self.assertIn("loss", output)
        self.assertIn("mean_squared_error", output)
        self.assertAllClose(output["mean_squared_error"], 16.0)

    @parameterized.named_parameters([("flat", False), ("dict", True)])
    @pytest.mark.requires_trainable_backend
    def test_evaluate_with_custom_test_step(self, return_dict):
        if backend.backend() == "jax":
            model = JaxCustomTrainTestStepModel(units=3)
        else:
            model = CustomTrainTestStepModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        batch_size = 16

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )
        output = model.evaluate(
            x, y, batch_size=batch_size, return_dict=return_dict
        )
        self.assertLen(output, 3)
        if return_dict:
            self.assertAllClose(output["my_custom_metric"], 5.0)
        else:
            self.assertAllClose(output[-1], 5.0)  # Custom metrics go last.

    @parameterized.named_parameters(
        named_product(
            generator_type=["tf", "jax", "scipy"], mode=["eager", "graph"]
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_evaluate_sparse(self, generator_type, mode):
        model = ExampleModel(units=3)
        model.compile(
            optimizer=optimizers.Adagrad(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=False,
        )
        dataset = sparse_generator(generator_type)
        model.evaluate(dataset)

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_predict_flow(self, run_eagerly, jit_compile):
        # Test basic example
        model = ExampleModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = np.ones((100, 4))
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertAllClose(outputs, 4 * np.ones((100, 3)))

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_predict_flow_struct(self, run_eagerly, jit_compile):
        # Test with input/output structs
        model = StructModel(units=3)
        model.run_eagerly = run_eagerly
        model.jit_compile = jit_compile

        x = {
            "x_one": np.ones((100, 4)),
            "x_two": np.ones((100, 4)),
        }
        batch_size = 16
        outputs = model.predict(x, batch_size=batch_size)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(len(outputs), 2)
        self.assertAllClose(outputs["y_one"], 4 * np.ones((100, 3)))
        self.assertAllClose(outputs["y_two"], 4 * np.ones((100, 3)))

    @parameterized.named_parameters(
        named_product(
            generator_type=["tf", "jax", "scipy"], mode=["eager", "graph"]
        )
    )
    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS,
        reason="Backend does not support sparse tensors.",
    )
    def test_predict_sparse(self, generator_type, mode):
        model = ExampleModel(units=3)
        model.compile(
            optimizer=optimizers.Adagrad(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=False,
        )
        dataset = sparse_generator(generator_type)
        dataset_size = sum(
            [batch[1].shape[0] for batch in sparse_generator(generator_type)]
        )
        y = model.predict(dataset)
        self.assertEqual(len(y), dataset_size)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Memory optimization is only implemented in JAX",
    )
    def test_fit_eval_flow_for_jax_model_weights(self):
        test_obj = self

        model = ExampleModel(units=3)
        epochs = 3
        batch_size = 20
        steps_per_epoch = 7
        dataset_size = batch_size * (steps_per_epoch - 2)
        x = np.ones((dataset_size, 4))
        y = np.zeros((dataset_size, 3))

        class ModelWeightCheck(Callback):
            def __init__(self):
                super().__init__()

            # Note that we access model via self._model since self.model
            # will trigger a sync of the jax training state back to the model.
            def on_train_batch_end(self, batch, logs=None):
                for v in self._model.trainable_variables:
                    test_obj.assertIsNone(v._value)
                for v in self._model.non_trainable_variables:
                    test_obj.assertIsNone(v._value)
                for v in self._model.optimizer.variables:
                    test_obj.assertIsNone(v._value)
                for v in self._model.metrics_variables:
                    test_obj.assertIsNone(v._value)

            def on_test_batch_end(self, batch, logs=None):
                for v in self._model.non_trainable_variables:
                    test_obj.assertIsNone(v._value)
                for v in self._model.metrics_variables:
                    test_obj.assertIsNone(v._value)

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        model.fit(
            x,
            y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[ModelWeightCheck()],
        )

        model.evaluate(
            x,
            y,
            batch_size=batch_size,
            callbacks=[ModelWeightCheck()],
        )

    @parameterized.named_parameters(
        named_product(
            steps_per_execution=[3, 101], mode=["eager", "non_jit", "jit"]
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_steps_per_execution_steps_count(self, steps_per_execution, mode):
        data_size = 100
        batch_size = 16
        epochs = 2

        x = np.ones((data_size, 4))
        y = np.ones((data_size, 1))

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        step_count = StepCount(steps_per_execution)

        history = model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[step_count],
            verbose=0,
        )

        self.assertEqual(
            step_count.begin_count,
            1 + (data_size - 1) // (steps_per_execution * batch_size),
        )
        self.assertEqual(step_count.end_count, step_count.begin_count)
        self.assertEqual(step_count.epoch_begin_count, epochs)
        self.assertEqual(
            step_count.epoch_end_count, step_count.epoch_begin_count
        )

        model_2 = ExampleModel(units=1)
        model_2.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=1,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        history_2 = model_2.fit(
            x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=0
        )

        self.assertAllClose(history.history["loss"], history_2.history["loss"])
        self.assertAllClose(model.get_weights(), model_2.get_weights())
        self.assertAllClose(
            model.predict(x, batch_size=batch_size),
            model_2.predict(x, batch_size=batch_size),
        )
        self.assertAllClose(model.evaluate(x, y), model_2.evaluate(x, y))

    @parameterized.named_parameters(
        named_product(steps_per_execution=[3, 8, 32])
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="`unrolled_steps_per_execution` is only "
        "available with the tensorflow backend.",
    )
    def test_steps_per_execution_unrolled_steps_steps_count(
        self, steps_per_execution
    ):
        data_size = 100
        batch_size = 16
        epochs = 2
        unrolled_steps_per_execution = 8

        x = np.ones((data_size, 4))
        y = np.ones((data_size, 1))

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            jit_compile=True,
        )
        step_count = StepCount(steps_per_execution)
        model.unrolled_steps_per_execution = unrolled_steps_per_execution
        history = model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[step_count],
            verbose=0,
        )

        self.assertEqual(
            step_count.begin_count,
            1 + (data_size - 1) // (steps_per_execution * batch_size),
        )
        self.assertEqual(step_count.end_count, step_count.begin_count)
        self.assertEqual(step_count.epoch_begin_count, epochs)
        self.assertEqual(
            step_count.epoch_end_count, step_count.epoch_begin_count
        )

        model_2 = ExampleModel(units=1)
        model_2.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            jit_compile=True,
        )
        model_2.unrolled_steps_per_execution = 1
        history_2 = model_2.fit(
            x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=0
        )

        self.assertAllClose(history.history["loss"], history_2.history["loss"])
        self.assertAllClose(model.get_weights(), model_2.get_weights())
        self.assertAllClose(
            model.predict(x, batch_size=batch_size),
            model_2.predict(x, batch_size=batch_size),
        )
        self.assertAllClose(model.evaluate(x, y), model_2.evaluate(x, y))

    @parameterized.named_parameters(
        named_product(
            steps_per_execution=[1, 50], mode=["eager", "non_jit", "jit"]
        )
    )
    def test_predict_preserve_order(self, steps_per_execution, mode):
        def generate_uneven_batches():
            batch_sizes = [2, 3, 4]

            def gen_i():
                for i in range(100):
                    yield i

            iterator = iter(gen_i())
            j = 0
            while True:
                batch_size = batch_sizes[j % len(batch_sizes)]
                try:
                    batch = np.array(
                        [next(iterator) for _ in range(batch_size)]
                    )
                except StopIteration:
                    break
                j += 1
                yield batch

        from keras.src.utils.module_utils import tensorflow as tf

        dataset = tf.data.Dataset.from_generator(
            generate_uneven_batches,
            output_signature=tf.TensorSpec((None,), dtype=tf.int32),
        )
        x = keras.layers.Input(shape=())
        y = keras.layers.Identity()(x)
        model = keras.Model(x, y)
        model.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )

        preds = model.predict(x=dataset, verbose=0)

        self.assertAllEqual(preds, np.arange(len(preds), dtype=np.float32))

    @parameterized.named_parameters(
        named_product(
            steps_per_execution=[1, 50], mode=["eager", "non_jit", "jit"]
        )
    )
    def test_predict_generator(self, steps_per_execution, mode):
        batch_size = 2

        def generate_batches():
            def gen_i():
                for i in range(10):
                    yield i

            iterator = iter(gen_i())
            j = 0
            while True:
                try:
                    batch = np.array(
                        [next(iterator) for _ in range(batch_size)]
                    )
                except StopIteration:
                    break
                j += 1
                yield (batch,)

        model = keras.Sequential(
            [keras.layers.InputLayer(shape=()), keras.layers.Identity()]
        )
        model.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )

        preds = model.predict(x=generate_batches(), verbose=0)
        self.assertAllEqual(
            preds, np.concatenate(list(generate_batches()), axis=1)[0]
        )

    @parameterized.named_parameters(
        named_product(
            steps_per_execution=[3, 101], mode=["eager", "non_jit", "jit"]
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_steps_per_execution_steps_count_unknown_dataset_size(
        self, steps_per_execution, mode
    ):
        data_size = 100
        batch_size = 16
        epochs = 2

        def data_generator():
            x = np.ones((data_size, 4), dtype=np.float32)
            y = np.ones((data_size, 1), dtype=np.float32)
            for _x, _y in zip(x, y):
                yield _x, _y

        import tensorflow as tf

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(4,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            ),
        )
        dataset = dataset.batch(batch_size)

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        step_count = StepCount(steps_per_execution)

        history = model.fit(
            dataset,
            epochs=epochs,
            callbacks=[step_count],
            verbose=0,
        )

        batch_count = 1 + (data_size - 1) // (steps_per_execution * batch_size)
        self.assertGreaterEqual(step_count.begin_count, batch_count)
        self.assertEqual(step_count.end_count, batch_count)
        self.assertEqual(step_count.epoch_begin_count, epochs)
        self.assertEqual(
            step_count.epoch_end_count, step_count.epoch_begin_count
        )

        model_2 = ExampleModel(units=1)
        model_2.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=1,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        history_2 = model_2.fit(dataset, epochs=epochs, verbose=0)

        self.assertAllClose(history.history["loss"], history_2.history["loss"])
        self.assertAllClose(model.get_weights(), model_2.get_weights())
        self.assertAllClose(
            model.predict(dataset),
            model_2.predict(dataset),
        )
        self.assertAllClose(model.evaluate(dataset), model_2.evaluate(dataset))

    @parameterized.named_parameters(
        named_product(
            steps_per_epoch_test=[
                "match_one_epoch",
                "match_multi_epoch",
                "not_match_too_low",
                "not_match_but_high_enough",
            ],
            mode=["eager", "non_jit", "jit"],
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_steps_per_execution_steps_per_epoch(
        self, steps_per_epoch_test, mode
    ):
        batch_size = 8
        epochs = 2
        steps_per_execution = 2
        num_batches = 5 * steps_per_execution
        data_size = num_batches * batch_size

        if steps_per_epoch_test == "match_one_epoch":
            steps_per_epoch = num_batches
        elif steps_per_epoch_test == "match_multi_epoch":
            steps_per_epoch = num_batches // steps_per_execution
        elif steps_per_epoch_test == "not_match_too_low":
            steps_per_epoch = num_batches - steps_per_execution
        elif steps_per_epoch_test == "not_match_but_high_enough":
            steps_per_epoch = num_batches + steps_per_execution

        x = np.ones((data_size, 4))
        y = np.ones((data_size, 1))

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            metrics=[EpochAgnosticMeanSquaredError()],
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        step_observer = StepObserver()

        model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[step_observer],
            verbose=0,
        )
        if steps_per_epoch_test != "not_match_too_low":
            training_batch_count = (
                epochs
                * min(steps_per_epoch, num_batches)
                // steps_per_execution
            )
        else:
            complete_epochs = (num_batches // steps_per_execution) // (
                steps_per_epoch // steps_per_execution
            )
            remaining_steps = (num_batches // steps_per_execution) % (
                steps_per_epoch // steps_per_execution
            )
            steps_cycles = [
                complete_epochs * steps_per_epoch // steps_per_execution,
                remaining_steps,
            ] * epochs
            steps_per_epochs = steps_cycles[:epochs]
            training_batch_count = sum(steps_per_epochs)

        self.assertEqual(step_observer.begin_count, training_batch_count)
        self.assertEqual(step_observer.end_count, step_observer.begin_count)
        self.assertEqual(step_observer.epoch_begin_count, epochs)
        self.assertEqual(
            step_observer.epoch_end_count, step_observer.epoch_begin_count
        )

        if steps_per_epoch_test != "not_match_too_low":
            model_2 = ExampleModel(units=1)
            model_2.compile(
                loss="mse",
                optimizer="sgd",
                metrics=[EpochAgnosticMeanSquaredError()],
                steps_per_execution=1,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode == "jit"),
            )
            step_observer_2 = StepObserver()

            if steps_per_epoch_test in (
                "not_match_but_high_enough",
                "match_one_epoch",
            ):
                model_2_epochs = epochs
            else:
                model_2_epochs = 1

            model_2.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=model_2_epochs,
                callbacks=[step_observer_2],
                verbose=0,
            )

            losses = step_observer.batch_loss_history
            losses_2 = step_observer_2.batch_loss_history[
                steps_per_execution - 1 :: steps_per_execution
            ]
            self.assertAllClose(losses, losses_2)
            self.assertAllClose(model.get_weights(), model_2.get_weights())
            self.assertAllClose(
                model.predict(x, batch_size=batch_size),
                model_2.predict(x, batch_size=batch_size),
            )
            self.assertAllClose(model.evaluate(x, y), model_2.evaluate(x, y))

    @parameterized.named_parameters(
        named_product(
            steps_per_epoch_test=[
                "match_one_epoch",
                "match_multi_epoch",
                "not_match_too_low",
                "not_match_but_high_enough",
            ],
            mode=["eager", "non_jit", "jit"],
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_steps_per_epoch(self, steps_per_epoch_test, mode):
        batch_size = 8
        epochs = 4
        num_batches = 10
        data_size = num_batches * batch_size

        if steps_per_epoch_test == "match_one_epoch":
            steps_per_epoch = num_batches
        elif steps_per_epoch_test == "match_multi_epoch":
            steps_per_epoch = num_batches // (epochs // 2)
        elif steps_per_epoch_test == "not_match_too_low":
            steps_per_epoch = num_batches - 1
        elif steps_per_epoch_test == "not_match_but_high_enough":
            steps_per_epoch = num_batches + 1

        x = np.ones((data_size, 4))
        y = np.ones((data_size, 1))

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            metrics=[EpochAgnosticMeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        step_observer = StepObserver()

        model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[step_observer],
            verbose=0,
        )
        if steps_per_epoch_test != "not_match_too_low":
            training_batch_count = epochs * min(steps_per_epoch, num_batches)
        else:
            complete_epochs = num_batches // steps_per_epoch
            remaining_steps = num_batches % steps_per_epoch
            steps_cycles = [
                complete_epochs * steps_per_epoch,
                remaining_steps,
            ] * epochs
            steps_per_epochs = steps_cycles[:epochs]
            training_batch_count = sum(steps_per_epochs)

        self.assertEqual(step_observer.begin_count, training_batch_count)
        self.assertEqual(step_observer.end_count, step_observer.begin_count)
        self.assertEqual(step_observer.epoch_begin_count, epochs)
        self.assertEqual(
            step_observer.epoch_end_count, step_observer.epoch_begin_count
        )

        if steps_per_epoch_test != "not_match_too_low":
            model_2 = ExampleModel(units=1)
            model_2.compile(
                loss="mse",
                optimizer="sgd",
                metrics=[EpochAgnosticMeanSquaredError()],
                steps_per_execution=1,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode == "jit"),
            )
            step_observer_2 = StepObserver()

            if steps_per_epoch_test in (
                "not_match_but_high_enough",
                "match_one_epoch",
            ):
                model_2_epochs = epochs
            elif steps_per_epoch_test == "match_multi_epoch":
                model_2_epochs = epochs // (num_batches // steps_per_epoch)
            else:
                model_2_epochs = 1

            model_2.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=model_2_epochs,
                callbacks=[step_observer_2],
                verbose=0,
            )

            losses = step_observer.batch_loss_history
            losses_2 = step_observer_2.batch_loss_history

            self.assertAllClose(losses, losses_2)
            self.assertAllClose(model.get_weights(), model_2.get_weights())
            self.assertAllClose(
                model.predict(x, batch_size=batch_size),
                model_2.predict(x, batch_size=batch_size),
            )
            self.assertAllClose(model.evaluate(x, y), model_2.evaluate(x, y))

    @pytest.mark.requires_trainable_backend
    def test_max_epochs_and_steps(self):
        batch_size = 8
        epochs = 4
        num_batches = 10
        data_size = num_batches * batch_size
        x, y = np.ones((data_size, 4)), np.ones((data_size, 1))
        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            metrics=[EpochAgnosticMeanSquaredError()],
        )
        step_observer = StepObserver()
        model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[step_observer],
            verbose=0,
        )
        self.assertEqual(step_observer.epoch_begin_count, epochs)
        self.assertEqual(step_observer.begin_count, num_batches * epochs)
        try:
            config.set_max_epochs(2)
            config.set_max_steps_per_epoch(3)
            step_observer = StepObserver()
            model.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[step_observer],
                verbose=0,
            )
            self.assertEqual(step_observer.epoch_begin_count, 2)
            self.assertEqual(step_observer.begin_count, 6)
        finally:
            config.set_max_epochs(None)
            config.set_max_steps_per_epoch(None)

    @parameterized.named_parameters(
        named_product(
            steps_per_epoch_test=[
                "match",
                "not_match_too_low",
                "not_match_but_high_enough",
            ],
            mode=["eager", "non_jit", "jit"],
        )
    )
    @pytest.mark.requires_trainable_backend
    def test_steps_per_execution_steps_per_epoch_unknown_data_size(
        self, steps_per_epoch_test, mode
    ):
        batch_size = 8
        epochs = 2
        steps_per_execution = 2
        num_batches = 5 * epochs * steps_per_execution
        data_size = num_batches * batch_size

        if steps_per_epoch_test == "match":
            steps_per_epoch = num_batches // epochs
        elif steps_per_epoch_test == "not_match_too_low":
            steps_per_epoch = num_batches - steps_per_execution
        elif steps_per_epoch_test == "not_match_but_high_enough":
            steps_per_epoch = num_batches + steps_per_execution

        def data_generator():
            x = np.ones((data_size, 4), dtype=np.float32)
            y = np.ones((data_size, 1), dtype=np.float32)
            for _x, _y in zip(x, y):
                yield _x, _y

        import tensorflow as tf

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(4,), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=tf.float32),
            ),
        )
        dataset = dataset.batch(batch_size)

        model = ExampleModel(units=1)
        model.compile(
            loss="mse",
            optimizer="sgd",
            metrics=[EpochAgnosticMeanSquaredError()],
            steps_per_execution=steps_per_execution,
            run_eagerly=(mode == "eager"),
            jit_compile=(mode == "jit"),
        )
        step_observer = StepObserver()

        model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[step_observer],
            verbose=0,
        )
        if steps_per_epoch_test != "not_match_too_low":
            training_batch_count = (
                epochs
                * min(steps_per_epoch, num_batches)
                // steps_per_execution
            )
        else:
            complete_epochs = (num_batches // steps_per_execution) // (
                steps_per_epoch // steps_per_execution
            )
            remaining_steps = (num_batches // steps_per_execution) % (
                steps_per_epoch // steps_per_execution
            )
            steps_cycles = [
                complete_epochs * steps_per_epoch // steps_per_execution,
                remaining_steps,
            ] * epochs
            steps_per_epochs = steps_cycles[:epochs]
            training_batch_count = sum(steps_per_epochs)

        self.assertGreaterEqual(step_observer.begin_count, training_batch_count)
        self.assertEqual(step_observer.end_count, training_batch_count)
        self.assertEqual(step_observer.epoch_begin_count, epochs)
        self.assertEqual(
            step_observer.epoch_end_count, step_observer.epoch_begin_count
        )

        if steps_per_epoch_test != "not_match_too_low":
            model_2 = ExampleModel(units=1)
            model_2.compile(
                loss="mse",
                optimizer="sgd",
                metrics=[EpochAgnosticMeanSquaredError()],
                steps_per_execution=1,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode == "jit"),
            )
            step_observer_2 = StepObserver()

            if steps_per_epoch_test == "not_match_but_high_enough":
                model_2_epochs = epochs
            else:
                model_2_epochs = 1

            model_2.fit(
                dataset,
                epochs=model_2_epochs,
                callbacks=[step_observer_2],
                verbose=0,
            )

            losses = step_observer.batch_loss_history
            losses_2 = step_observer_2.batch_loss_history[
                steps_per_execution - 1 :: steps_per_execution
            ]
            self.assertAllClose(losses, losses_2)
            self.assertAllClose(model.get_weights(), model_2.get_weights())
            self.assertAllClose(
                model.predict(dataset), model_2.predict(dataset)
            )
            self.assertAllClose(
                model.evaluate(dataset), model_2.evaluate(dataset)
            )

    def test_steps_per_execution_steps_count_without_training(self):
        test_obj = self

        class StepCount(Callback):
            def __init__(self):
                super().__init__()
                self.test_count = 0
                self.predict_count = 0
                self.batches = [0, 3, 6]

            def on_test_batch_begin(self, batch, logs=None):
                test_obj.assertEqual(batch, self.batches[self.test_count])
                self.test_count += 1

            def on_predict_batch_begin(self, batch, logs=None):
                test_obj.assertEqual(batch, self.batches[self.predict_count])
                self.predict_count += 1

        x = np.ones((100, 4))
        y = np.ones((100, 1))
        batch_size = 16
        model = ExampleModel(units=1)
        model.compile(loss="mse", steps_per_execution=3)
        step_count = StepCount()
        model.predict(x, batch_size=batch_size, callbacks=[step_count])
        self.assertEqual(step_count.predict_count, 3)
        model.evaluate(x, y, batch_size=batch_size, callbacks=[step_count])
        self.assertEqual(step_count.test_count, 3)

    @pytest.mark.requires_trainable_backend
    def test_fit_with_different_batch_size_same_loss(self):
        x = np.random.rand(100, 4)
        y = np.ones((100, 1))
        model = ExampleModel(units=1)
        model.trainable = False
        model.compile(loss="mse")
        loss1 = model.fit(x, y, batch_size=80).history["loss"]
        loss2 = model.fit(x, y, batch_size=100).history["loss"]
        self.assertAllClose(loss1, loss2)

    def test_evaluate_with_different_batch_size_same_loss(self):
        x = np.random.rand(100, 4)
        y = np.ones((100, 1))
        model = ExampleModel(units=1)
        model.compile(loss="mse")
        loss1 = model.evaluate(x, y, batch_size=80)
        loss2 = model.evaluate(x, y, batch_size=100)
        self.assertAllClose(loss1, loss2)

    def test_evaluate_fixed_batch_dim_graph_matches_eager(self):
        import tensorflow as tf

        batch_size = 4
        num_samples = 32

        x_data = np.random.randn(num_samples, 4).astype(np.float32)
        y_data = np.random.randn(num_samples, 1).astype(np.float32)

        def make_dataset():
            def gen():
                for i in range(0, num_samples, batch_size):
                    yield (
                        x_data[i : i + batch_size],
                        y_data[i : i + batch_size],
                    )

            return tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec((batch_size, 4), tf.float32),
                    tf.TensorSpec((batch_size, 1), tf.float32),
                ),
            )

        # Graph mode
        model_graph = ExampleModel(units=1)
        model_graph.compile(loss="mse", metrics=["mae"], run_eagerly=False)
        graph_result = model_graph.evaluate(
            make_dataset(), verbose=0, return_dict=True
        )

        # Eager mode
        model_eager = ExampleModel(units=1)
        model_eager.compile(loss="mse", metrics=["mae"], run_eagerly=True)
        eager_result = model_eager.evaluate(
            make_dataset(), verbose=0, return_dict=True
        )

        self.assertAllClose(
            graph_result["mae"],
            eager_result["mae"],
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.requires_trainable_backend
    def test_adds_loss_scaling_optimizer(self):
        model = TrainingTestingLayer(dtype="mixed_float16")
        model.compile(optimizer="rmsprop", loss="mse")
        x = np.ones((128, 1))
        y = np.zeros((128, 1))
        model.fit(x, y, batch_size=32)
        self.assertIsInstance(model.optimizer, optimizers.LossScaleOptimizer)

        model = TrainingTestingLayer(dtype="mixed_float16")
        model.compile(optimizer="rmsprop", loss="mse", auto_scale_loss=False)
        x = np.ones((128, 1))
        y = np.zeros((128, 1))
        model.fit(x, y, batch_size=32)
        self.assertIsInstance(model.optimizer, RMSprop)

        model = TrainingTestingLayer(dtype="mixed_bfloat16")
        model.compile(optimizer="rmsprop", loss="mse")
        x = np.ones((128, 1))
        y = np.zeros((128, 1))
        model.fit(x, y, batch_size=32)
        self.assertIsInstance(model.optimizer, RMSprop)

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="half precision unsupported on torch CPU.",
    )
    def test_loss_scaling_prevents_underflow(self):
        class DeepModel(Trainer, layers.Layer):
            def __init__(self):
                layers.Layer.__init__(self, dtype="mixed_float16")
                Trainer.__init__(self)
                self.layers = []
                for _ in range(15):
                    # Sigmoid has a small gradient, will eventually underflow.
                    self.layers.append(
                        layers.Dense(
                            1,
                            use_bias=False,
                            kernel_initializer="ones",
                            activation="sigmoid",
                            dtype="mixed_float16",
                        )
                    )

            def call(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        loss = losses.MeanSquaredError()
        # Blow up any gradient updates, so underflow is obvious.
        optimizer = optimizers.SGD(learning_rate=1e9)
        model = DeepModel()
        model.compile(optimizer, loss=loss, auto_scale_loss=False)
        model.fit(np.ones((1, 1)), np.ones((1, 1)), batch_size=1)
        first_kernel = model.layers[0].kernel
        # Without autoscaling, the first dense will not update.
        self.assertEqual(first_kernel, np.ones_like(first_kernel))

        # Blow up any gradient updates, so underflow is obvious.
        optimizer = optimizers.SGD(learning_rate=1e9)
        model = DeepModel()
        model.compile(optimizer, loss=loss, auto_scale_loss=True)
        model.fit(np.ones((1, 1)), np.ones((1, 1)), batch_size=1)
        first_kernel = model.layers[0].kernel
        # With autoscaling, the first dense will update.
        self.assertNotEqual(first_kernel, np.ones_like(first_kernel))

    @pytest.mark.requires_trainable_backend
    def test_training_arg(self):
        model = TrainingTestingLayer()
        model.compile(optimizer="rmsprop", loss="mse")
        x = np.ones((128, 1))
        y = np.zeros((128, 1))
        history = model.fit(x, y, batch_size=32)
        self.assertAllClose(history.history["loss"], [1.0])
        val_loss = model.evaluate(x, y, batch_size=32)
        self.assertAllClose(val_loss, 0.0)
        preds = model.predict(x)
        self.assertAllClose(preds, np.zeros((128, 1)))

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_on_batch_methods(self, run_eagerly, jit_compile):
        if backend.backend() == "torch" and jit_compile:
            self.skipTest(
                "test_on_batch with jit_compile=True not supported in torch "
                "backend yet."
            )
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))
        sw = np.arange(100).reshape((100,)).astype("float32") / 50.0

        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        logs = model.train_on_batch(x, y)
        self.assertIsInstance(logs, list)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 16.0)

        logs = model.train_on_batch(x, y, return_dict=True)
        self.assertIsInstance(logs, dict)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 15.158, tpu_decimal=1)

        logs = model.test_on_batch(x, y)
        self.assertIsInstance(logs, list)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 14.360, tpu_decimal=1)

        logs = model.test_on_batch(x, y, return_dict=True)
        self.assertIsInstance(logs, dict)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 14.360, tpu_decimal=1)

        output = model.predict_on_batch(x)
        self.assertIsInstance(output, np.ndarray)
        self.assertAllClose(
            output[0],
            np.array([3.789511, 3.789511, 3.789511]),
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

        # With sample weights
        logs = model.train_on_batch(x, y, sw)
        self.assertAlmostEqual(logs[0], 14.217, tpu_decimal=1)
        logs = model.test_on_batch(x, y, sw)
        self.assertAlmostEqual(logs[0], 13.476, tpu_decimal=1)
        output = model.predict_on_batch(x)
        self.assertAllClose(
            output[0],
            np.array([3.689468, 3.689468, 3.689468]),
            tpu_atol=1e-2,
            tpu_rtol=1e-2,
        )

        # With class weights
        logs = model.train_on_batch(x, y, class_weight={1: 0.3, 0: 0.2})
        self.assertAlmostEqual(logs[0], 2.722, tpu_decimal=1)

    @parameterized.named_parameters(
        [
            ("eager", True, False),
            ("graph_fn", False, False),
            ("jit", False, True),
        ]
    )
    def test_on_batch_methods_without_training(self, run_eagerly, jit_compile):
        if backend.backend() == "torch" and jit_compile:
            self.skipTest(
                "test_on_batch with jit_compile=True not supported in torch "
                "backend yet."
            )
        model = ExampleModel(units=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 3))

        model.compile(
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        output = model.predict_on_batch(x)
        self.assertIsInstance(output, np.ndarray)
        self.assertAllClose(output[0], np.array([4.0, 4.0, 4.0]))

        logs = model.test_on_batch(x, y)
        self.assertIsInstance(logs, list)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs[0], 16.0)

        logs = model.test_on_batch(x, y, return_dict=True)
        self.assertIsInstance(logs, dict)
        self.assertEqual(len(logs), 2)
        self.assertAlmostEqual(logs["loss"], 16.0)

    def test_nested_input_predict(self):
        # https://github.com/keras-team/keras/issues/325

        class TupleInputModel(keras.Model):
            def call(self, inputs):
                a, b = inputs
                return a + b

        model = TupleInputModel()
        x1, x2 = np.random.rand(2, 3, 4)
        out = model.predict((x1, x2))
        self.assertEqual(out.shape, (3, 4))

        class DictInputModel(keras.Model):
            def call(self, inputs):
                return inputs["a"] + inputs["b"]

        model = DictInputModel()
        x1, x2 = np.random.rand(2, 3, 4)
        out = model.predict({"a": x1, "b": x2})
        self.assertEqual(out.shape, (3, 4))

    @pytest.mark.requires_trainable_backend
    def test_for_eval_epoch_iterator(self):
        model = ExampleModel(units=3)
        model.compile(
            optimizer="adam", loss="mse", metrics=["mean_absolute_error"]
        )
        x = np.ones((16, 4))
        y = np.zeros((16, 3))
        x_test = np.ones((16, 4))
        y_test = np.zeros((16, 3))
        model.fit(
            x,
            y,
            batch_size=4,
            validation_data=(x_test, y_test),
        )
        self.assertIsNone(getattr(model, "_eval_epoch_iterator", None))

        # Try model.fit with reshaped validation_data
        # This will throw an exception which is intended
        try:
            model.fit(
                x,
                y,
                batch_size=4,
                validation_data=(
                    x_test.reshape((-1, 16, 4)),
                    y_test.reshape((-1, 16, 3)),
                ),
            )
        except:
            pass

        # Try model.fit with correct validation_data this should work.
        # After successful training `_eval_epoch_iterator` should be None
        model.fit(
            x,
            y,
            batch_size=4,
            validation_data=(x_test, y_test),
        )
        self.assertIsNone(getattr(model, "_eval_epoch_iterator", None))

    @pytest.mark.requires_trainable_backend
    def test_callback_methods_keys(self):
        test_obj = self

        class CustomCallback(Callback):
            def on_train_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_train_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(
                    keys,
                    [
                        "loss",
                        "mean_absolute_error",
                        "val_loss",
                        "val_mean_absolute_error",
                    ],
                )

            def on_epoch_begin(self, epoch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_epoch_end(self, epoch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(
                    keys,
                    [
                        "loss",
                        "mean_absolute_error",
                        "val_loss",
                        "val_mean_absolute_error",
                    ],
                )

            def on_test_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_test_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, ["loss", "mean_absolute_error"])

            def on_predict_begin(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_predict_end(self, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_train_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_train_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, ["loss", "mean_absolute_error"])

            def on_test_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_test_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, ["loss", "mean_absolute_error"])

            def on_predict_batch_begin(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, [])

            def on_predict_batch_end(self, batch, logs=None):
                keys = sorted(list(logs.keys()))
                test_obj.assertEqual(keys, ["outputs"])

        model = ExampleModel(units=3)
        model.compile(
            optimizer="adam", loss="mse", metrics=["mean_absolute_error"]
        )
        x = np.ones((16, 4))
        y = np.zeros((16, 3))
        x_test = np.ones((16, 4))
        y_test = np.zeros((16, 3))
        model.fit(
            x,
            y,
            callbacks=[CustomCallback()],
            batch_size=4,
            validation_data=(x_test, y_test),
        )
        model.evaluate(x_test, y_test, batch_size=4)
        model.predict(x_test, batch_size=4)

    @pytest.mark.requires_trainable_backend
    def test_internal_only_loss(self):
        class LossLayer(layers.Layer):
            def call(self, x):
                self.add_loss(ops.sum(x))
                return x

        model = keras.Sequential(
            [
                layers.Dense(2),
                LossLayer(),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam")
        x = np.ones((16, 2))
        y = np.zeros((16, 1))
        model.fit(x, y, batch_size=4)

    def get_layer(self):
        class ExampleLayer(keras.Layer):
            def call(self, x):
                return x * 2

        return ExampleLayer

    def get_model(self):
        class ExampleModel(keras.Model):
            def call(self, x):
                return x * 2

        return ExampleModel

    def get_functional(self):
        ExampleLayer = self.get_layer()

        class ExampleFunctional(keras.src.Functional):
            def __init__(self, input_shape=(None,)):
                inputs = keras.Input(input_shape)
                outputs = ExampleLayer()(inputs)
                super().__init__(inputs=inputs, outputs=outputs)

        return ExampleFunctional

    @parameterized.named_parameters(
        [
            {
                "testcase_name": "model",
                "model_class": "get_model",
            },
            {
                "testcase_name": "layer",
                "model_class": "get_layer",
            },
            {
                "testcase_name": "functional",
                "model_class": "get_functional",
            },
        ]
    )
    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        keras.backend.backend() != "tensorflow",
        reason="Only tensorflow supports raggeds",
    )
    def test_trainer_with_raggeds(self, model_class):
        from keras.src.utils.module_utils import tensorflow as tf

        def loss_fn(y, y_pred, sample_weight=None):
            return 0

        model = getattr(self, model_class)()()
        x = tf.ragged.constant([[1], [2, 3]])

        # test forward pass
        y = model(x)
        self.assertEqual(type(y), tf.RaggedTensor)

        # test training
        if model_class in ["get_model", "get_functional"]:
            model.compile(optimizer="adam", loss=loss_fn)
            model.fit(x, x)
            y = model.predict(x)
            self.assertEqual(type(y), tf.RaggedTensor)

        # test if everything works with the sequential model
        model = keras.Sequential([model])
        model.compile(optimizer="adam", loss=loss_fn)
        model.fit(x, x)
        y = model.predict(x)
        self.assertEqual(type(y), tf.RaggedTensor)

    def test_functional_layer_training_overrides_model_training_argument(self):
        # Test that `predict` with a dropout op
        # has nondeterministic behavior across batches.

        inputs = layers.Input((20,))
        # Passing `training=True` during construction will force dropout to be
        # enabled even during predict.
        outputs = layers.Dropout(0.5, seed=1337)(inputs, training=True)
        model = keras.Model(inputs, outputs)
        out1 = model.predict(np.ones((4, 20)), batch_size=2)
        self.assertNotAllClose(out1[:2, :], out1[2:4, :])

        out2 = model.predict_on_batch(np.ones((2, 20)))
        out3 = model.predict_on_batch(np.ones((2, 20)))
        self.assertNotAllClose(out2, out3)

    @pytest.mark.requires_trainable_backend
    def test_recompile(self):
        model = ExampleModel(units=3)
        model.compile(
            optimizer="sgd", loss="mse", metrics=["mean_squared_error"]
        )
        history_1 = model.fit(np.ones((3, 2)), np.ones((3, 3))).history
        eval_out_1 = model.evaluate(
            np.ones((3, 2)), np.ones((3, 3)), return_dict=True
        )
        model.compile(
            optimizer="sgd", loss="mse", metrics=["mean_absolute_error"]
        )
        history_2 = model.fit(np.ones((3, 2)), np.ones((3, 3))).history
        eval_out_2 = model.evaluate(
            np.ones((3, 2)), np.ones((3, 3)), return_dict=True
        )
        self.assertEqual(
            sorted(list(history_1.keys())), ["loss", "mean_squared_error"]
        )
        self.assertEqual(
            sorted(list(eval_out_1.keys())), ["loss", "mean_squared_error"]
        )
        self.assertEqual(
            sorted(list(history_2.keys())), ["loss", "mean_absolute_error"]
        )
        self.assertEqual(
            sorted(list(eval_out_2.keys())), ["loss", "mean_absolute_error"]
        )

    def test_evaluate_return_list_respect_metrics_order(self):
        def metrics_zero(y_true, y_pred):
            return 0.0

        def metrics_one(y_true, y_pred):
            return 1.0

        model = ExampleModel(units=3)
        model.compile(
            optimizer="sgd", loss="mse", metrics=[metrics_zero, metrics_one]
        )
        eval_out = model.evaluate(np.ones((3, 2)), np.ones((3, 3)))
        self.assertLen(eval_out, 3)
        self.assertEqual(eval_out[1], 0.0)
        self.assertEqual(eval_out[2], 1.0)

        model.compile(
            optimizer="sgd", loss="mse", metrics=[metrics_one, metrics_zero]
        )
        eval_out = model.evaluate(np.ones((3, 2)), np.ones((3, 3)))
        self.assertLen(eval_out, 3)
        self.assertEqual(eval_out[1], 1.0)
        self.assertEqual(eval_out[2], 0.0)

    @pytest.mark.requires_trainable_backend
    def test_nested_inputs(self):
        model = ListInputModel(units=2)
        out = model([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(tuple(out.shape), (3, 2))
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
        history = model.fit(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        ).history
        self.assertAllClose(
            history["loss"], [16.0], tpu_atol=1e-4, tpu_rtol=1e-4
        )
        train_out = model.train_on_batch(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(train_out[0], 14.44, tpu_atol=1e-1, tpu_rtol=1e-1)
        eval_out = model.evaluate(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(eval_out[0], 13.0321, tpu_atol=1e-2, tpu_rtol=1e-2)
        eval_out = model.test_on_batch(
            [np.ones((3, 2)), np.ones((3, 3))], np.ones((3, 2))
        )
        self.assertAllClose(eval_out[0], 13.0321, tpu_atol=1e-2, tpu_rtol=1e-2)
        predict_out = model.predict([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(predict_out.shape, (3, 2))
        predict_out = model.predict_on_batch([np.ones((3, 2)), np.ones((3, 3))])
        self.assertEqual(predict_out.shape, (3, 2))

    @pytest.mark.requires_trainable_backend
    def test_validation_data_infinite_generator(self):
        # Test that you can pass an infinite generator to `validation_data`
        # arg of fit() as well as a `validation_steps` argument and that
        # validation only runs for the correct number of steps.
        model = ExampleModel(units=3)
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])

        class Recorder(keras.callbacks.Callback):
            def __init__(self):
                self.train_counter = 0
                self.val_counter = 0

            def on_train_batch_end(self, *args, **kwargs):
                self.train_counter += 1

            def on_test_batch_end(self, *args, **kwargs):
                self.val_counter += 1

        def infinite_gen():
            while True:
                yield np.ones((2, 2)), np.ones((2, 3))

        recorder = Recorder()

        model.fit(
            infinite_gen(),
            validation_data=infinite_gen(),
            steps_per_epoch=3,
            validation_steps=4,
            epochs=1,
            shuffle=False,
            callbacks=[recorder],
        )
        self.assertEqual(recorder.train_counter, 3)
        self.assertEqual(recorder.val_counter, 4)

    @pytest.mark.skipif(
        backend.backend() != "jax",
        reason="Only JAX backend uses _maybe_symbolic_build.",
    )
    @pytest.mark.requires_trainable_backend
    def test_symbolic_build_empty_dataset(self):
        val_x = np.empty((0, 2), dtype="float32")
        val_y = np.empty((0, 3), dtype="float32")
        unbuilt_model = ExampleModel(units=3)
        unbuilt_model.compile(optimizer="sgd", loss="mse")
        with self.assertRaisesRegex(
            ValueError,
            "The symbolic build failed because train or validation dataset "
            "is empty",
        ):
            unbuilt_model.evaluate(val_x, val_y)

    @pytest.mark.requires_trainable_backend
    def test_fit_empty_validation_data(self):
        model = ExampleModel(units=3)
        model.compile(optimizer="sgd", loss="mse")
        x = np.ones((4, 2), dtype="float32")
        y = np.ones((4, 3), dtype="float32")
        val_x = np.empty((0, 2), dtype="float32")
        val_y = np.empty((0, 3), dtype="float32")
        model.fit(x, y, epochs=1, batch_size=2, validation_data=(val_x, val_y))

    @parameterized.named_parameters(
        [
            ("fit", "fit", "training", "train"),
            ("evaluate", "evaluate", "evaluating", "test"),
            ("predict", "predict", "predicting", "predict"),
        ]
    )
    @pytest.mark.requires_trainable_backend
    def test_stop_loop(self, method, method_gerund, on_end_name):
        model = ExampleModel(units=3)
        model.compile(optimizer="sgd", loss="mse", metrics=["mse"])

        class Stopper(keras.callbacks.Callback):
            def __init__(self, stop_count):
                self.stop_count = stop_count
                self.counter = 0
                setattr(self, f"on_{on_end_name}_batch_end", self.batch_end)

            def batch_end(self, *args, **kwargs):
                self.counter += 1
                if self.counter == self.stop_count:
                    setattr(self.model, f"stop_{method_gerund}", True)

        def infinite_gen():
            while True:
                x = np.ones((2, 2))
                y = np.ones((2, 3))
                yield (x,) if method == "predict" else (x, y)

        stop_count = 5
        stopper = Stopper(stop_count)

        getattr(model, method)(
            infinite_gen(),
            callbacks=[stopper],
        )
        self.assertEqual(stopper.counter, stop_count)

    @pytest.mark.requires_trainable_backend
    def test_constraints_are_applied(self):
        model = models.Sequential(
            [layers.Dense(2, kernel_constraint="non_neg")]
        )
        x = np.ones((2, 3))
        y = np.ones((2, 2))
        model.compile(optimizer="rmsprop", loss="mse")
        model.fit(x, y)
        self.assertGreaterEqual(
            np.min(backend.ops.convert_to_numpy(model.layers[0].kernel)), 0.0
        )

    @pytest.mark.requires_trainable_backend
    def test_rng_updated_during_predict(self):
        class TestTimeDropout(layers.Layer):
            def __init__(self):
                super().__init__()
                self.random_generator = keras.random.SeedGenerator()

            def call(self, x):
                return keras.random.dropout(
                    x, rate=0.5, seed=self.random_generator
                )

        inputs = layers.Input((20,))
        outputs = TestTimeDropout()(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="mse")

        x = np.ones((32, 20))
        out_1 = model.predict(x)
        out_2 = model.predict(x)
        self.assertGreater(np.mean(np.abs(out_1 - out_2)), 0.01)

    @pytest.mark.requires_trainable_backend
    def test_callbacks_can_update_state_at_batch_boundary(self):
        class CounterModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.train_counter = self.add_weight(
                    shape=(),
                    initializer="zeros",
                )
                self.test_counter = self.add_weight(
                    shape=(),
                    initializer="zeros",
                )
                self.predict_counter = self.add_weight(
                    shape=(),
                    initializer="zeros",
                )
                self.dense = layers.Dense(3)

            def call(self, x):
                return self.dense(x)

        class CounterCallback(keras.callbacks.Callback):
            def __init__(self):
                self.eager_call_counter_train = 0
                self.eager_call_counter_test = 0
                self.eager_call_counter_predict = 0

            def on_train_batch_end(self, *args, **kwargs):
                self.model.train_counter.assign_add(1)
                self.eager_call_counter_train += 1

            def on_test_batch_end(self, *args, **kwargs):
                self.model.test_counter.assign_add(1)
                self.eager_call_counter_test += 1

            def on_predict_batch_end(self, *args, **kwargs):
                self.model.predict_counter.assign_add(1)
                self.eager_call_counter_predict += 1

        model = CounterModel()
        model.compile(
            optimizer="sgd", loss="mse", metrics=["mse"], run_eagerly=True
        )
        cbk = CounterCallback()
        model.fit(
            np.ones((4, 3)),
            np.ones((4, 3)),
            callbacks=[cbk],
            epochs=3,
            batch_size=1,
            verbose=0,
            validation_data=(np.ones((2, 3)), np.ones((2, 3))),
        )
        self.assertAlmostEqual(cbk.eager_call_counter_train, 12)
        self.assertAlmostEqual(model.train_counter.numpy(), 12)
        self.assertAlmostEqual(cbk.eager_call_counter_test, 6)
        self.assertAlmostEqual(model.test_counter.numpy(), 6)
        model.predict(
            np.ones((4, 3)),
            callbacks=[cbk],
            batch_size=1,
        )
        self.assertAlmostEqual(cbk.eager_call_counter_predict, 4)
        self.assertAlmostEqual(model.predict_counter.numpy(), 4)

    @pytest.mark.requires_trainable_backend
    def test_metric_update_in_compute_loss(self):
        test_self = self

        class MyModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.custom_metric = keras.metrics.Mean(name="custom")
                self.dense = keras.layers.Dense(2)

            def call(self, x):
                return self.dense(x)

            def compute_loss(
                self,
                x=None,
                y=None,
                y_pred=None,
                sample_weight=None,
                training=True,
            ):
                if not in_symbolic_scope():
                    test_self.assertTrue(training)
                loss = super().compute_loss(
                    x, y, y_pred, sample_weight, training
                )
                self.custom_metric.update_state(loss * 4)
                return loss

        model = MyModel()
        model.compile(optimizer="sgd", loss="mse")
        x = np.ones((32, 4))
        y = np.ones((32, 2)) * 2
        history = model.fit(x, y)
        self.assertAlmostEqual(
            history.history["custom"][0], history.history["loss"][0] * 4
        )

    @pytest.mark.requires_trainable_backend
    def test_fwd_pass_loss_presence_in_compute_loss(self):
        test_self = self

        class MyModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.custom_metric = keras.metrics.Mean(name="custom")
                self.dense = keras.layers.Dense(2, activity_regularizer="l2")

            def call(self, x):
                return self.dense(x)

            def compute_loss(
                self,
                x=None,
                y=None,
                y_pred=None,
                sample_weight=None,
                training=True,
            ):
                if not in_symbolic_scope():
                    test_self.assertTrue(training)
                loss = super().compute_loss(
                    x, y, y_pred, sample_weight, training
                )
                self.custom_metric.update_state(sum(self.losses))
                return loss

        model = MyModel()
        model.compile(optimizer="sgd", loss="mse")
        x = np.ones((32, 4))
        y = np.ones((32, 2)) * 2
        history = model.fit(x, y)
        self.assertGreater(history.history["custom"][0], 0.0)

    @pytest.mark.requires_trainable_backend
    def test_evaluate_with_custom_compute_loss(self):
        test_self = self

        class MyModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.custom_metric = keras.metrics.Mean(name="custom")
                self.dense = keras.layers.Dense(2, activity_regularizer="l2")

            def call(self, x):
                return self.dense(x)

            def compute_loss(
                self,
                x=None,
                y=None,
                y_pred=None,
                sample_weight=None,
                training=True,
            ):
                if not in_symbolic_scope():
                    test_self.assertFalse(training)
                loss = super().compute_loss(
                    x, y, y_pred, sample_weight, training
                )
                self.custom_metric.update_state(loss * 4)
                return loss

        model = MyModel()
        model.compile(optimizer="sgd", loss="mse")
        x = np.ones((32, 4))
        y = np.ones((32, 2)) * 2
        logs = model.evaluate(x, y, return_dict=True)
        self.assertAlmostEqual(logs["custom"], logs["loss"] * 4)

    @pytest.mark.requires_trainable_backend
    def test_compute_loss_no_training_backwards_compatibility(self):
        class MyModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.custom_metric = keras.metrics.Mean(name="custom")
                self.dense = keras.layers.Dense(2, activity_regularizer="l2")

            def call(self, x):
                return self.dense(x)

            def compute_loss(
                self,
                x=None,
                y=None,
                y_pred=None,
                sample_weight=None,
            ):
                loss = super().compute_loss(x, y, y_pred, sample_weight)
                self.custom_metric.update_state(loss * 4)
                return loss

        model = MyModel()
        model.compile(optimizer="sgd", loss="mse")
        x = np.ones((32, 4))
        y = np.ones((32, 2)) * 2
        logs = model.evaluate(x, y, return_dict=True)
        self.assertAlmostEqual(logs["custom"], logs["loss"] * 4)
        history = model.fit(x, y)
        self.assertAlmostEqual(
            history.history["custom"][0], history.history["loss"][0] * 4
        )

    @pytest.mark.requires_trainable_backend
    def test_loss_weights(self):
        epochs = 3
        batch_size = 20
        dataset_size = batch_size * 2

        # Single output case.
        model = ExampleModel(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            loss_weights=0.2,
        )
        x = np.ones((dataset_size, 4))
        y = np.zeros((dataset_size, 3))
        history = model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertAllClose(
            history["loss"],
            [3.182979, 3.115617, 3.049681],
            atol=1e-3,
            tpu_atol=1e-2,
        )

        # Dict output case.
        model = StructModel(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss={
                "y_one": losses.MeanSquaredError(),
                "y_two": losses.MeanSquaredError(),
            },
            metrics={
                "y_one": metrics.MeanSquaredError(),
                "y_two": metrics.MeanSquaredError(),
            },
            loss_weights={"y_one": 0.1, "y_two": 0.2},
        )
        x1 = np.ones((dataset_size, 4))
        x2 = np.ones((dataset_size, 4))
        y1 = np.zeros((dataset_size, 3))
        y2 = np.zeros((dataset_size, 3))
        history = model.fit(
            {"x_one": x1, "x_two": x2},
            {"y_one": y1, "y_two": y2},
            batch_size=batch_size,
            epochs=epochs,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertAllClose(
            history["loss"],
            [4.778718, 4.694403, 4.611693],
            atol=1e-3,
            tpu_atol=1e-2,
        )

        # List output case.
        model = ListOutputModel(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=[losses.MeanSquaredError(), losses.MeanSquaredError()],
            metrics=[metrics.MeanSquaredError(), metrics.MeanSquaredError()],
            loss_weights=[0.1, 0.2],
        )
        x = np.ones((dataset_size, 4))
        y1 = np.zeros((dataset_size, 3))
        y2 = np.zeros((dataset_size, 3))
        history = model.fit(
            x,
            [y1, y2],
            batch_size=batch_size,
            epochs=epochs,
        )
        history = history.history
        self.assertIn("loss", history)
        self.assertAllClose(
            history["loss"],
            [4.778718, 4.694403, 4.611693],
            atol=1e-3,
            tpu_atol=1e-2,
        )

    @pytest.mark.requires_trainable_backend
    def test_partial_loss_partial_label(self):
        inputs = keras.Input((2,))
        x = keras.layers.Dense(3, kernel_initializer="ones")(inputs)
        partial_model = keras.Model(inputs, [x, x, x])
        partial_model.compile(loss=["mse", None, None])
        full_model = keras.Model(inputs, [x, x, x])
        full_model.compile(loss=["mse", "mse", "mse"])

        eval_x = np.ones((32, 2))
        eval_y = np.ones((32, 3))

        partial_logs = partial_model.evaluate(eval_x, eval_y, return_dict=True)
        logs = full_model.evaluate(eval_x, [eval_y] * 3, return_dict=True)

        self.assertAlmostEqual(partial_logs["loss"] * 3, logs["loss"])

    def test_symbolic_build(self):
        class ExampleModelWithTrainingArgs(Trainer, layers.Layer):
            def __init__(self, units):
                layers.Layer.__init__(self)
                Trainer.__init__(self)
                self.dense = layers.Dense(units)
                self.bn = layers.BatchNormalization(axis=-1)

            def build(self, input_shape):
                self.dense.build(input_shape)
                input_shape = self.dense.compute_output_shape(input_shape)
                self.bn.build(input_shape)

            def call(self, x, training=None):
                outputs = self.bn(self.dense(x), training=training)
                return [outputs, outputs]

        model = ExampleModelWithTrainingArgs(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=[losses.MeanSquaredError(), losses.MeanSquaredError()],
            metrics=[metrics.MeanSquaredError(), metrics.MeanSquaredError()],
        )
        x = np.ones((4, 4))
        y = np.zeros((4, 3))
        model(x)  # Eager call to build model weights
        ref_weights = model.get_weights()

        # Before `_symbolic_build`
        self.assertTrue(model.built)
        self.assertFalse(model._compile_metrics.built)
        self.assertFalse(model._compile_loss.built)
        self.assertLen(model._compile_loss.metrics, 0)
        self.assertLen(model.metrics, 2)

        model._symbolic_build(data_batch=(x, (y, y)))
        weights = model.get_weights()

        # Ensure weights are intact
        self.assertEqual(len(weights), len(ref_weights))
        for w, ref_w in zip(weights, ref_weights):
            self.assertAllClose(w, ref_w)

        # Ensure `built`
        self.assertTrue(model.built)
        self.assertTrue(model._compile_metrics.built)
        self.assertTrue(model._compile_loss.built)

        # Ensure the len of metrics (original metrics + loss trackers)
        self.assertLen(model._compile_metrics.metrics, 2)
        self.assertLen(model._compile_loss.metrics, 2)
        self.assertLen(model.metrics, 4)

        # Ensure no values in metrics
        for v in model._compile_metrics.variables:
            self.assertAllClose(v, 0.0)
        for v in model._compile_loss.variables:
            self.assertAllClose(v, 0.0)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="This test is only applicable to TensorFlow.",
    )
    @pytest.mark.requires_trainable_backend
    def test_jit_compile_with_tf_determinism(self):
        from tensorflow.python.framework.config import disable_op_determinism
        from tensorflow.python.framework.config import enable_op_determinism

        enable_op_determinism()

        model = ExampleModel(units=3)
        model.compile(
            optimizer=optimizers.SGD(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        self.assertFalse(model.jit_compile)
        disable_op_determinism()

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch uses a Python bundling loop; side-effect counters "
        "will increment for every batch even after compilation.",
    )
    def test_retracing(self):
        x = np.ones((100, 4))
        y = np.ones((100, 1))

        input = keras.Input(shape=[4])
        output = keras.layers.Dense(1, activation="relu")(input)

        tracing_count = [0]

        class TracingCounterModel(keras.Model):
            def train_step(self, *args):
                tracing_count[0] = tracing_count[0] + 1
                return super().train_step(*args)

        model = TracingCounterModel(inputs=input, outputs=output)
        model.compile(
            loss="mse",
            optimizer="adam",
            steps_per_execution=20,
        )

        epochs = 1
        model.fit(
            x=x,
            y=y,
            batch_size=1,
            epochs=epochs,
            verbose=0,
        )
        self.assertLessEqual(tracing_count[0], 2)

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch uses a Python bundling loop; side-effect counters "
        "will increment for every batch even after compilation.",
    )
    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="`predict_function` with `steps_per_execution` is not "
        "optimized for tensorflow yet",
    )
    def test_retracing_predict(self):
        x = np.ones((100, 4))

        input = keras.Input(shape=[4])
        output = keras.layers.Dense(1, activation="relu")(input)

        tracing_count = [0]

        class TracingCounterModel(keras.Model):
            def predict_step(self, *args):
                tracing_count[0] = tracing_count[0] + 1
                return super().predict_step(*args)

        model = TracingCounterModel(inputs=input, outputs=output)
        model.compile(
            loss="mse",
            optimizer="adam",
            steps_per_execution=20,
        )

        model.predict(
            x=x,
            batch_size=1,
            verbose=0,
        )
        self.assertLessEqual(tracing_count[0], 2)


class JAXTrainerCorrectnessTest(test_case.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("single_device", False),
        ("distributed", True),
    )
    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
    def test_jit_fit_with_out_shardings_logic(self, distributed):
        x = np.random.rand(64, 8).astype("float32")
        y = np.random.rand(64, 1).astype("float32")

        distribution = None
        if distributed:
            if len(jax.local_devices()) < 2:
                self.skipTest(
                    "Distributed test requires at least 2 JAX devices."
                )

            devices = jax.local_devices()
            mesh = DeviceMesh(
                shape=(len(devices),), axis_names=("batch",), devices=devices
            )
            distribution = DataParallel(mesh)

        scope = distribution.scope() if distribution else mock.MagicMock()

        with scope:
            model = models.Sequential(
                [
                    layers.Dense(4, activation="relu", input_shape=(8,)),
                    layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse", jit_compile=True)

            if distribution:
                expected_shardings = [
                    v.value.sharding for v in model.trainable_variables
                ]
                self.assertNotEqual(len(set(expected_shardings)), 1)

            model.fit(x, y, epochs=2, batch_size=32, verbose=0)

            if distribution:
                actual_shardings = [
                    v.value.sharding for v in model.trainable_variables
                ]
                self.assertListEqual(actual_shardings, expected_shardings)

    @parameterized.named_parameters(("one_step", 1), ("two_steps", 2))
    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
    def test_fit_evaluate_predict_keep_int64_inputs(self, steps_per_execution):
        x = np.array([[2**40 + 1], [2**40 + 2]], dtype="int64")
        y = np.full((2, 1), 2**10, dtype="float32")
        with x64_mode_scope("explicit_x64"):
            inputs = layers.Input((1,), dtype="int64")
            # Keeps only the high bits, which a cut to 32 bits would lose.
            high_bits = layers.Lambda(lambda t: ops.cast(t // 2**30, "float32"))
            model = models.Model(inputs, high_bits(inputs))
            model.compile(loss="mse", steps_per_execution=steps_per_execution)

            history = model.fit(x, y, batch_size=1, verbose=0)
            self.assertEqual(history.history["loss"], [0.0])
            self.assertEqual(model.evaluate(x, y, batch_size=1, verbose=0), 0.0)
            self.assertAllEqual(model.predict(x, batch_size=1, verbose=0), y)
            self.assertAllEqual(model.predict_on_batch(x), y)
