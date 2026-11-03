from unittest import mock

import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import optimizers
from keras.src import testing


class ScatterUpdateOptimizer(optimizers.Optimizer):
    def __init__(self):
        super().__init__(learning_rate=0.001)

    def build(self, variables):
        if self.built:
            return
        super().build(variables)
        self.momentums = [
            self.add_variable_from_reference(v, name="momentum")
            for v in variables
        ]

    def update_step(self, grad, variable, learning_rate):
        momentum = self.momentums[self._get_variable_index(variable)]
        self.assign(momentum, ops.cast(grad, momentum.dtype))
        self.assign(variable, ops.cast(grad, variable.dtype))


TEST_CASES = [
    {
        "testcase_name": "adadelta",
        "optimizer_class": optimizers.Adadelta,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adafactor",
        "optimizer_class": optimizers.Adafactor,
        "init_kwargs": {"clip_threshold": 0.5},
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adagrad",
        "optimizer_class": optimizers.Adagrad,
        "expect_model_sparse_variable_updates": True,
        "expect_optimizer_sparse_variable_updates": True,
    },
    {
        "testcase_name": "adam",
        "optimizer_class": optimizers.Adam,
    },
    {
        "testcase_name": "adam_amsgrad",
        "optimizer_class": optimizers.Adam,
        "init_kwargs": {"amsgrad": True},
    },
    {
        "testcase_name": "adamax",
        "optimizer_class": optimizers.Adamax,
    },
    {
        "testcase_name": "adamw",
        "optimizer_class": optimizers.AdamW,
    },
    {
        "testcase_name": "adamw_amsgrad",
        "optimizer_class": optimizers.AdamW,
        "init_kwargs": {"amsgrad": True},
    },
    {
        "testcase_name": "ftrl",
        "optimizer_class": optimizers.Ftrl,
    },
    {
        "testcase_name": "lion",
        "optimizer_class": optimizers.Lion,
    },
    {
        "testcase_name": "loss_scale_optimizer_sgd",
        "optimizer_class": lambda: optimizers.LossScaleOptimizer(
            optimizers.SGD(learning_rate=0.5)
        ),
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "nadam",
        "optimizer_class": optimizers.Nadam,
    },
    {
        "testcase_name": "rmsprop",
        "optimizer_class": optimizers.RMSprop,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "rmsprop_momentum",
        "optimizer_class": optimizers.RMSprop,
        "init_kwargs": {"momentum": 0.05},
    },
    {
        "testcase_name": "rmsprop_momentum_centered",
        "optimizer_class": optimizers.RMSprop,
        "init_kwargs": {"momentum": 0.05, "centered": True},
    },
    {
        "testcase_name": "sgd",
        "optimizer_class": optimizers.SGD,
        "expect_model_sparse_variable_updates": True,
    },
    {
        "testcase_name": "sgd_momentum",
        "optimizer_class": optimizers.SGD,
        "init_kwargs": {"momentum": 0.05},
    },
    {
        "testcase_name": "sgd_momentum_nesterov",
        "optimizer_class": optimizers.SGD,
        "init_kwargs": {"momentum": 0.05, "nesterov": True},
    },
    {
        "testcase_name": "scatter_update",
        "optimizer_class": ScatterUpdateOptimizer,
        "expect_model_sparse_variable_updates": True,
        "expect_optimizer_sparse_variable_updates": True,
    },
]


@pytest.mark.skipif(
    not backend.SUPPORTS_SPARSE_TENSORS,
    reason="Backend does not support sparse tensors.",
)
class OptimizerSparseTest(testing.TestCase):
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="IndexedSlices are specific to TensorFlow.",
    )
    @parameterized.product(
        optimizer_class=(
            optimizers.Adam,
            optimizers.Adagrad,
            optimizers.RMSprop,
        ),
        cancel_duplicates=(False, True),
        run_eagerly=(False, True),
    )
    def test_duplicate_indices_correctness(
        self, optimizer_class, cancel_duplicates, run_eagerly
    ):
        import tensorflow as tf

        var_sparse = backend.Variable(initializer="ones", shape=(3, 2))
        var_dense = backend.Variable(initializer="ones", shape=(3, 2))
        optimizer_sparse = optimizer_class(learning_rate=0.1)
        optimizer_dense = optimizer_class(learning_rate=0.1)
        optimizer_sparse.build([var_sparse])
        optimizer_dense.build([var_dense])
        duplicate = [-1.0, -2.0] if cancel_duplicates else [2.0, 1.0]
        grad_sparse = tf.IndexedSlices(
            values=tf.constant([[1.0, 2.0], [3.0, 4.0], duplicate]),
            indices=tf.constant([2, 0, 2]),
            dense_shape=tf.constant([3, 2]),
        )

        def update(gradient):
            optimizer_sparse.apply([gradient], [var_sparse])
            optimizer_dense.apply([tf.convert_to_tensor(gradient)], [var_dense])

        if not run_eagerly:
            update = tf.function(update)

        # Cancellation can leave weights unchanged while corrupting slots.
        # A subsequent dense update also checks the effect of saved state.
        for gradient in (grad_sparse, tf.ones((3, 2))):
            update(gradient)
            self.assertAllClose(var_sparse.value, var_dense.value)
            for sparse_state, dense_state in zip(
                optimizer_sparse.variables, optimizer_dense.variables
            ):
                self.assertAllClose(sparse_state.value, dense_state.value)

    @parameterized.named_parameters(TEST_CASES)
    def test_sparse_gradients(
        self,
        optimizer_class,
        init_kwargs={},
        expect_model_sparse_variable_updates=False,
        expect_optimizer_sparse_variable_updates=False,
    ):
        # This test verifies that:
        # - Optimizers use Keras ops everywhere instead of native operators
        #   (e.g. `ops.add()` instead of `+`) where sparse gradients are handled
        # - The used ops handle sparse gradients
        # - Optimizers use `self.assign/assign_add/assign_sub` instead of
        #   calling the method on the variable directly. Otherwise, the sparse
        #   updates are densified before being applied.
        # - For some optimizers, a sparse gradient actually results in a sparse
        #   variable update as per `expect_model_sparse_variable_updates` and
        #   `expect_optimizer_sparse_variable_updates`

        model_variable = backend.Variable(initializer="ones", shape=(5, 10))
        optimizer = optimizer_class(**init_kwargs)

        # Mocking "tensorflow.Variable" won't work as it gets substituted with
        # the resource variable class.

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            grad = tf.IndexedSlices(0.5 * ops.ones((3, 10)), (0, 2, 4), (5, 10))
            sparse_class = tf.IndexedSlices
            variable_class = model_variable._value.__class__
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            grad = jax_sparse.BCOO(
                (0.5 * ops.ones((3, 10)), ((0,), (2,), (4,))), shape=(5, 10)
            )
            sparse_class = jax_sparse.JAXSparse
            variable_class = model_variable.__class__
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        optimizer_to_patch = (
            optimizer.inner_optimizer
            if isinstance(optimizer, optimizers.LossScaleOptimizer)
            else optimizer
        )

        model_sparse_variable_updates = False
        optimizer_sparse_variable_updates = False

        def mock_optimizer_assign(variable, value):
            nonlocal model_sparse_variable_updates
            nonlocal optimizer_sparse_variable_updates
            if isinstance(variable, backend.Variable):
                variable = variable._value
            if isinstance(value, sparse_class):
                if variable is model_variable._value:
                    model_sparse_variable_updates = True
                elif any(variable is v._value for v in optimizer.variables):
                    optimizer_sparse_variable_updates = True

        def mock_variable_assign(variable, value):
            # Make an exception for scalar variables
            if len(variable.shape):
                pytest.fail(
                    "Optimizer is calling `assign`, `assign_add` or "
                    "`assign_sub` directly on a variable. Use "
                    "`self.assign/assign_add/assign_sub(variable, value)` "
                    "instead to support sparse updates."
                )

        # patch "_apply_weight_decay" to exclude this special case.
        # patch the optimizer "assign" methods to detect sparse updates.
        # patch the tf.Variable "assign" methods to detect direct assign calls.
        with (
            mock.patch.object(
                optimizer_to_patch, "_apply_weight_decay", autospec=True
            ),
            mock.patch.object(
                optimizer_to_patch, "assign", autospec=True
            ) as optimizer_assign,
            mock.patch.object(
                optimizer_to_patch, "assign_add", autospec=True
            ) as optimizer_assign_add,
            mock.patch.object(
                optimizer_to_patch, "assign_sub", autospec=True
            ) as optimizer_assign_sub,
            mock.patch.object(
                variable_class, "assign", autospec=True
            ) as variable_assign,
            mock.patch.object(
                variable_class, "assign_add", autospec=True
            ) as variable_assign_add,
            mock.patch.object(
                variable_class, "assign_sub", autospec=True
            ) as variable_assign_sub,
        ):
            optimizer_assign.side_effect = mock_optimizer_assign
            optimizer_assign_add.side_effect = mock_optimizer_assign
            optimizer_assign_sub.side_effect = mock_optimizer_assign
            variable_assign.side_effect = mock_variable_assign
            variable_assign_add.side_effect = mock_variable_assign
            variable_assign_sub.side_effect = mock_variable_assign

            optimizer.apply([grad], [model_variable])

        self.assertEqual(
            model_sparse_variable_updates, expect_model_sparse_variable_updates
        )
        self.assertEqual(
            optimizer_sparse_variable_updates,
            expect_optimizer_sparse_variable_updates,
        )

    @parameterized.named_parameters(TEST_CASES)
    def test_sparse_correctness(
        self, optimizer_class, init_kwargs={}, **kwargs
    ):
        # This test verifies that applying a sparse gradient gives the same
        # numerical results as the same dense gradient.

        optimizer_sparse = optimizer_class(**init_kwargs)
        optimizer_dense = optimizer_class(**init_kwargs)
        var_sparse = backend.Variable(initializer="ones", shape=(5, 3, 2))
        var_dense = backend.Variable(initializer="ones", shape=(5, 3, 2))
        stateless = backend.backend() == "jax"
        if stateless:
            optimizer_sparse.build([var_sparse])
            optimizer_dense.build([var_dense])

        optimizer_sparse_vars = optimizer_sparse.variables
        optimizer_dense_vars = optimizer_dense.variables
        var_sparse_values = [var_sparse.value]
        var_dense_values = [var_dense.value]

        for i in range(5):
            if backend.backend() == "tensorflow":
                import tensorflow as tf

                grad_sparse = tf.IndexedSlices(
                    values=ops.ones((3, 3, 2)) * (10.0 - i),
                    indices=(0, 2, 4),
                    dense_shape=(5, 3, 2),
                )
            elif backend.backend() == "jax":
                import jax.experimental.sparse as jax_sparse

                grad_sparse = jax_sparse.BCOO(
                    (ops.ones((3, 3, 2)) * (10.0 - i), ((0,), (2,), (4,))),
                    shape=(5, 3, 2),
                )
            else:
                self.fail(
                    f"Sparse is unsupported with backend {backend.backend()}"
                )

            grad_dense = ops.convert_to_tensor(grad_sparse, sparse=False)
            if stateless:
                (
                    var_sparse_values,
                    optimizer_sparse_vars,
                ) = optimizer_sparse.stateless_apply(
                    optimizer_sparse_vars, [grad_sparse], var_sparse_values
                )
                (
                    var_dense_values,
                    optimizer_dense_vars,
                ) = optimizer_dense.stateless_apply(
                    optimizer_dense_vars, [grad_dense], var_dense_values
                )
                self.assertAllClose(var_sparse_values[0], var_dense_values[0])

            else:
                optimizer_sparse.apply([grad_sparse], [var_sparse])
                optimizer_dense.apply([grad_dense], [var_dense])
                self.assertAllClose(var_sparse.value, var_dense.value)
