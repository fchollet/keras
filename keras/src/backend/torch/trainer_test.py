import os

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from absl.testing import parameterized
from torch._dynamo.utils import counters

from keras.src import backend
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch.distributed_test_utils import (
    TorchDistributedTestMixin,
)
from keras.src.backend.torch.trainer import TorchEpochIterator
from keras.src.backend.torch.trainer import _distribute_data
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel


class SimpleModel(models.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1)

    def call(self, x):
        return self.dense(x)


class TrainingAwareModel(models.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1)
        self.dropout = layers.Dropout(0.5)

    def call(self, x, training=False):
        x = self.dense(x)
        return self.dropout(x, training=training)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
@pytest.mark.no_pytest_xdist
class TorchTrainerDistributionTest(TorchDistributedTestMixin, testing.TestCase):
    _master_port = "29515"

    @parameterized.named_parameters(
        ("base", SimpleModel, False),
        ("with_distribution", SimpleModel, True),
        ("with_training_arg", TrainingAwareModel, False),
    )
    def test_torch_trainer_ddp(self, model_class, use_distribution):
        if use_distribution:
            mesh = DeviceMesh(
                shape=(1,),
                axis_names=["batch"],
                devices=np.array([dist_lib.list_devices()[0]]),
            )
            distribution = DataParallel(device_mesh=mesh)
            dist_lib.set_distribution(distribution)
            self.addCleanup(lambda: dist_lib.set_distribution(None))

        model = model_class()
        model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

        x = np.ones((10, 10), dtype="float32")
        y = np.ones((10, 1), dtype="float32")

        # Fit should trigger DDP wrapping
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        # Verify ddp_model is created
        self.assertTrue(hasattr(model, "ddp_model"))

        # Evaluate should also work
        model.evaluate(x, y, verbose=0)

        # Predict should also work
        y_pred = model.predict(x, verbose=0)
        self.assertEqual(y_pred.shape, (10, 1))

        # Test on_batch methods with numpy inputs
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)
        y_pred_batch = model.predict_on_batch(x)
        self.assertEqual(y_pred_batch.shape, (10, 1))

    def test_jit_compile_with_data_parallel(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        distribution = DataParallel(device_mesh=mesh)
        x = np.random.rand(16, 10).astype("float32")
        y = np.random.rand(16, 1).astype("float32")
        reference = SimpleModel()
        reference(x)
        initial_weights = reference.get_weights()

        trained_weights = []
        for jit_compile in (False, True):
            torch._dynamo.reset()
            counters.clear()
            with distribution.scope():
                model = SimpleModel()
                model(x)
                model.set_weights(initial_weights)
                model.compile(
                    optimizer=optimizers.SGD(0.1),
                    loss="mse",
                    metrics=["mae"],
                    jit_compile=jit_compile,
                )
                model.fit(
                    x, y, epochs=2, batch_size=4, shuffle=False, verbose=0
                )
            trained_weights.append(model.get_weights())

        # TorchDynamo compiled the steps instead of skipping every frame.
        self.assertGreater(counters["stats"]["unique_graphs"], 0)
        for eager_weight, jit_weight in zip(*trained_weights):
            self.assertAllClose(eager_weight, jit_weight, atol=1e-6)

    def test_jit_compile_with_model_parallel_runs_eagerly(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        layout_map = LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = (None, "batch")
        distribution = ModelParallel(layout_map=layout_map)
        x = np.ones((8, 10), dtype="float32")
        y = np.ones((8, 1), dtype="float32")

        with distribution.scope():
            model = SimpleModel()
            model.compile(optimizer="sgd", loss="mse", jit_compile=True)
            with pytest.warns(UserWarning, match="`ModelParallel`"):
                history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
        self.assertLen(history.history["loss"], 1)

    def test_metrics_distributed_multi_process(self):
        # Use real multi-process distribution to hit the metrics logic
        mp.spawn(
            _distributed_metrics_worker,
            args=(),
            nprocs=2,
            join=True,
        )

    def test_get_metrics_result_model_parallel(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        distribution = ModelParallel(layout_map=LayoutMap(mesh))
        dist_lib.set_distribution(distribution)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        model = SimpleModel()
        mean_metric = metrics.MeanAbsoluteError()
        model.compile(optimizer="sgd", loss="mse", metrics=[mean_metric])

        # Initialize metrics
        x = np.ones((2, 10), dtype="float32")
        y = np.ones((2, 1), dtype="float32")
        model.train_on_batch(x, y)

        results = model.get_metrics_result()

        self.assertIn("mean_absolute_error", results)

    def test_model_parallel_data_distribution(self):
        from torch.distributed.tensor import DTensor

        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        distribution = ModelParallel(layout_map=LayoutMap(mesh))
        dist_lib.set_distribution(distribution)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        x = np.ones((4, 2), dtype="float32")
        y = np.ones((4, 1), dtype="float32")
        distributed_data = _distribute_data((x, y, None))

        self.assertIsInstance(distributed_data[0], DTensor)
        self.assertIsInstance(distributed_data[1], DTensor)
        self.assertIsNone(distributed_data[2])

        iterator = TorchEpochIterator(x=x, y=y, batch_size=2)
        batches = list(iterator._get_iterator())
        self.assertLen(batches, 2)
        for batch in batches:
            self.assertIsInstance(batch[0], DTensor)
            self.assertIsInstance(batch[1], DTensor)


def _distributed_metrics_worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29519"
    # Use gloo for CPU-based multi-process testing
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=2
    )

    model = SimpleModel()
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

    x = np.ones((2, 10), dtype="float32")
    y = np.ones((2, 1), dtype="float32")

    # Run one step
    model.train_on_batch(x, y)

    # Trigger metrics aggregation
    results = model.get_metrics_result()

    # Basic verification
    if "mae" not in results or "loss" not in results:
        raise RuntimeError(f"Metrics missing from results: {results}")

    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchTrainerJitCompileTest(testing.TestCase):
    def test_jit_compile_with_frozen_weights(self):
        model = SimpleModel()
        x = np.ones((8, 10), dtype="float32")
        y = np.ones((8, 1), dtype="float32")
        model(x)
        # Manually set requires_grad = False on one of the trainable weight
        # tensors (kernel is frozen, but bias is still trainable)
        model.dense.kernel.value.requires_grad = False

        model.compile(
            optimizer=optimizers.Adam(),
            loss="mse",
            jit_compile=True,
        )
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
        self.assertIn("loss", history.history)

    def test_jit_compile_with_all_weights_frozen(self):
        model = SimpleModel()
        x = np.ones((8, 10), dtype="float32")
        y = np.ones((8, 1), dtype="float32")
        model(x)
        model.dense.kernel.value.requires_grad = False
        model.dense.bias.value.requires_grad = False

        model.compile(
            optimizer=optimizers.Adam(),
            loss="mse",
            jit_compile=True,
        )
        with self.assertRaises(ValueError):
            model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    def test_jit_compile_with_detached_loss(self):
        class DetachedLossModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(1)

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
                return torch.tensor(1.0)

        model = DetachedLossModel()
        x = np.ones((8, 10), dtype="float32")
        y = np.ones((8, 1), dtype="float32")
        model.compile(
            optimizer=optimizers.Adam(),
            jit_compile=True,
        )
        with self.assertRaises(ValueError):
            model.fit(x, y, epochs=1, batch_size=4, verbose=0)

    def test_jit_compile_with_train_step_calling_backward(self):
        class BackwardModel(SimpleModel):
            def train_step(self, data):
                x, y = data
                self.zero_grad()
                y_pred = self(x, training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
                loss.backward()
                trainable_weights = self.trainable_weights[:]
                gradients = [v.value.grad for v in trainable_weights]
                with torch.no_grad():
                    self.optimizer.apply(gradients, trainable_weights)
                return {"loss": loss.detach()}

        model = BackwardModel()
        x = np.ones((8, 10), dtype="float32")
        y = np.ones((8, 1), dtype="float32")
        model.compile(
            optimizer=optimizers.Adam(),
            loss="mse",
            jit_compile=True,
        )
        history = model.fit(x, y, epochs=2, batch_size=4, verbose=0)
        self.assertLen(history.history["loss"], 2)

    @parameterized.named_parameters(
        ("sgd", optimizers.SGD, {"learning_rate": 0.1}),
        ("sgd_momentum", optimizers.SGD, {"momentum": 0.9}),
        ("sgd_nesterov", optimizers.SGD, {"momentum": 0.9, "nesterov": True}),
        ("adadelta", optimizers.Adadelta, {"learning_rate": 1.0}),
    )
    def test_jit_compile_matches_eager_training(self, optimizer_cls, kwargs):
        x = np.random.rand(16, 10).astype("float32")
        y = np.random.rand(16, 1).astype("float32")
        reference = SimpleModel()
        reference(x)
        initial_weights = reference.get_weights()

        trained_weights = []
        for jit_compile in (False, True):
            model = SimpleModel()
            model(x)
            model.set_weights(initial_weights)
            model.compile(
                optimizer=optimizer_cls(**kwargs),
                loss="mse",
                jit_compile=jit_compile,
            )
            torch._dynamo.reset()
            counters.clear()
            model.fit(x, y, epochs=2, batch_size=4, shuffle=False, verbose=0)
            trained_weights.append(model.get_weights())

        if hasattr(torch._dynamo.config, "trace_autograd_ops"):
            # The `jit_compile=True` fit (the last one) compiles the whole
            # train step, optimizer update included, into one graph.
            self.assertEqual(dict(counters["graph_break"]), {})
            self.assertEqual(counters["stats"]["unique_graphs"], 1)
        for eager_weight, jit_weight in zip(*trained_weights):
            self.assertAllClose(eager_weight, jit_weight, atol=1e-6)

    @parameterized.named_parameters(
        ("adamw", optimizers.AdamW, {"learning_rate": 0.01}, None),
        (
            "adamw_bias_excluded",
            optimizers.AdamW,
            {"learning_rate": 0.01},
            ["bias"],
        ),
        # `Muon` applies `adam_weight_decay` when `weight_decay` is None.
        (
            "muon_adam_weight_decay_only",
            optimizers.Muon,
            {"weight_decay": None},
            None,
        ),
    )
    def test_jit_compile_with_weight_decay_compiles_once(
        self, optimizer_cls, kwargs, excluded_names
    ):
        x = np.random.rand(16, 10).astype("float32")
        y = np.random.rand(16, 1).astype("float32")
        reference = SimpleModel()
        reference(x)
        initial_weights = reference.get_weights()

        trained_weights = []
        for jit_compile in (False, True):
            model = SimpleModel()
            model(x)
            model.set_weights(initial_weights)
            optimizer = optimizer_cls(**kwargs)
            if excluded_names:
                optimizer.exclude_from_weight_decay(var_names=excluded_names)
            model.compile(
                optimizer=optimizer, loss="mse", jit_compile=jit_compile
            )
            torch._dynamo.reset()
            counters.clear()
            model.fit(x, y, epochs=2, batch_size=4, shuffle=False, verbose=0)
            trained_weights.append(model.get_weights())

        if hasattr(torch._dynamo.config, "trace_autograd_ops"):
            # The weight decay lookup must not make the second step recompile.
            self.assertEqual(counters["stats"]["unique_graphs"], 1)
        for eager_weight, jit_weight in zip(*trained_weights):
            self.assertAllClose(eager_weight, jit_weight, atol=1e-6)
