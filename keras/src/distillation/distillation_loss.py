import keras
from keras.src import backend
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.losses.losses import LossFunctionWrapper
from keras.src.losses.losses import binary_crossentropy
from keras.src.losses.losses import binary_focal_crossentropy
from keras.src.losses.losses import categorical_crossentropy
from keras.src.losses.losses import categorical_focal_crossentropy
from keras.src.losses.losses import kl_divergence
from keras.src.losses.losses import sparse_categorical_crossentropy
from keras.src.saving import serialization_lib
from keras.src.utils import tracking

# Losses that need targets `LogitsDistillation` cannot derive from the
# teacher logits (class indices, or independent per-class probabilities).
_UNSUPPORTED_LOGITS_LOSSES = (
    sparse_categorical_crossentropy,
    binary_crossentropy,
    binary_focal_crossentropy,
    categorical_focal_crossentropy,
)


def _convert_loss_to_function(loss_item):
    """Convert a loss string identifier to a loss function.

    Arguments:
        loss_item: Either a string identifier, a loss function instance,
            or `None`.

    Returns:
        A loss function instance, or `None`.

    Raises:
        ValueError: If the loss string identifier is unknown.
    """
    if loss_item is None:
        return None
    elif isinstance(loss_item, str):
        loss_fn = keras.losses.get(loss_item)
        if loss_fn is None:
            raise ValueError(f"Unknown loss function: '{loss_item}'.")
        return loss_fn
    else:
        return loss_item


def _unwrap_loss_fn(loss_fn):
    """Return the function a built-in loss instance wraps, or `loss_fn`."""
    if isinstance(loss_fn, LossFunctionWrapper):
        return loss_fn.fn
    return loss_fn


@keras_export("keras.distillation.DistillationLoss")
class DistillationLoss:
    """Base class for distillation loss computation.

    Distillation losses define how to compute the distillation loss
    between teacher and student outputs. Each loss implements a specific
    approach to knowledge transfer, from simple logits matching to feature-based
    distillation.

    To create custom distillation losses, subclass this class and
    override the `compute_loss` method.
    """

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss between teacher and student outputs.

        This method should implement the specific distillation logic for
        transferring knowledge from teacher to student.

        Arguments:
            teacher_outputs: Outputs from the teacher model. Can be a single
                tensor or a list/tuple of tensors for multi-output models.
            student_outputs: Outputs from the student model. Can be a single
                tensor or a list/tuple of tensors for multi-output models.
            **kwargs: Additional arguments for custom distillation_loss.
        Returns:
            Distillation loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that teacher and student outputs are compatible.

        Arguments:
            teacher_outputs: Outputs from the teacher model.
            student_outputs: Outputs from the student model.
        Raises:
            ValueError: If outputs are not compatible.
        """
        keras.tree.assert_same_structure(teacher_outputs, student_outputs)

    def validate_model_compatibility(self, teacher, student):
        """Validate that teacher and student models are compatible.

        Arguments:
            teacher: The teacher model.
            student: The student model.
        Raises:
            ValueError: If models are not compatible with this distillation
                loss.
        """
        pass


@keras_export("keras.distillation.FeatureDistillation")
class FeatureDistillation(DistillationLoss):
    """Feature distillation loss.

    Feature distillation transfers knowledge from intermediate layers of the
    teacher model to corresponding layers of the student model. This approach
    helps the student learn better internal representations and often leads
    to better performance compared to logits-only distillation.

    Arguments:
        loss: Loss function to use for feature distillation. Can be:
            - String identifier (e.g., 'mse', 'cosine_similarity', 'mae')
            - Keras loss instance
            - Nested structure of losses matching the layer output structure
            - `None` to skip distillation for that output (useful for
              multi-output models where you only want to distill some outputs)
            At least one loss must be non-`None`. Defaults to 'mse'.
        teacher_layer_name: Name of the teacher layer to extract features from.
            If `None`, uses the final output. Defaults to `None`.
        student_layer_name: Name of the student layer to extract features from.
            If `None`, uses the final output. Defaults to `None`.

    Examlpe(s):

    ```python
    # Basic feature distillation from final outputs
    distillation_loss = FeatureDistillation(loss="mse")

    # Distill from specific intermediate layers
    distillation_loss = FeatureDistillation(
        loss="mse",
        teacher_layer_name="dense_1",
        student_layer_name="dense_1"
    )

    # Use cosine similarity for different feature sizes
    distillation_loss = FeatureDistillation(
        loss="cosine_similarity",
        teacher_layer_name="conv2d_2",
        student_layer_name="conv2d_1"
    )

    # With custom loss instance
    distillation_loss = FeatureDistillation(
        loss=keras.losses.MeanAbsoluteError()
    )

    # For multi-output models
    distillation_loss = FeatureDistillation(
        loss=["mse", "cosine_similarity"]
    )

    # For multi-output models, only distill some outputs
    distillation_loss = FeatureDistillation(
        loss=["mse", None, "cosine_similarity"]  # Skip middle output
    )
    ```
    """

    @tracking.no_automatic_dependency_tracking
    def __init__(
        self, loss="mse", teacher_layer_name=None, student_layer_name=None
    ):
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_name = student_layer_name
        self.loss = tree.map_structure(_convert_loss_to_function, loss)

        flat_losses = tree.flatten(self.loss)
        if all(l is None for l in flat_losses):
            raise ValueError(
                "The `loss` argument in `FeatureDistillation` must "
                "contain at least one non-`None` value."
            )

    def validate_model_compatibility(self, teacher, student):
        """Validate that teacher and student models are compatible for feature
        distillation."""
        if (
            self.teacher_layer_name is not None
            or self.student_layer_name is not None
        ):
            teacher_is_subclassed = (
                not hasattr(teacher, "inputs") or teacher.inputs is None
            )
            student_is_subclassed = (
                not hasattr(student, "inputs") or student.inputs is None
            )

            if teacher_is_subclassed or student_is_subclassed:
                subclassed_models = []
                if teacher_is_subclassed:
                    subclassed_models.append("teacher")
                if student_is_subclassed:
                    subclassed_models.append("student")

                models_str = " and ".join(subclassed_models)
                raise ValueError(
                    f"FeatureDistillation with specific layer names requires "
                    f"Functional or Sequential models. The {models_str} "
                    f"model(s) appear to be subclassed (no symbolic "
                    f"inputs/outputs). Either use Functional/Sequential "
                    f"models, or use FeatureDistillation without layer names "
                    f"(to distill final outputs only), or use "
                    f"LogitsDistillation instead."
                )

        if self.teacher_layer_name is not None:
            try:
                teacher.get_layer(name=self.teacher_layer_name)
            except ValueError as e:
                raise ValueError(f"In teacher model: {e}")

        if self.student_layer_name is not None:
            try:
                student.get_layer(name=self.student_layer_name)
            except ValueError as e:
                raise ValueError(f"In student model: {e}")

    def validate_outputs(self, teacher_outputs, student_outputs):
        """Validate that outputs are compatible for feature distillation."""
        super().validate_outputs(teacher_outputs, student_outputs)

        try:
            tree.assert_same_structure(self.loss, teacher_outputs)
        except ValueError as e:
            raise ValueError(
                f"Loss structure mismatch. "
                f"Loss structure: {tree.structure(self.loss)}, "
                f"Output structure: {tree.structure(teacher_outputs)}. "
                f"Error: {e}"
            )

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute feature distillation loss using extracted features.

        Arguments:
            teacher_outputs: Extracted features from teacher layer.
            student_outputs: Extracted features from student layer.
            **kwargs: Additional arguments (ignored).
        Returns:
            Scalar distillation loss tensor.
        """

        def apply_loss(loss_fn, teacher_features, student_features):
            if loss_fn is None:
                return 0.0

            loss = backend.ops.numpy.mean(
                loss_fn(teacher_features, student_features)
            )

            return loss

        loss_values = tree.map_structure(
            apply_loss, self.loss, teacher_outputs, student_outputs
        )

        flat_losses = tree.flatten(loss_values)
        return backend.ops.numpy.sum(backend.ops.numpy.stack(flat_losses))

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "loss": keras.losses.serialize(self.loss),
            "teacher_layer_name": self.teacher_layer_name,
            "student_layer_name": self.student_layer_name,
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)


@keras_export("keras.distillation.LogitsDistillation")
class LogitsDistillation(DistillationLoss):
    """Distillation loss that transfers knowledge from final model outputs.

    This distillation loss applies temperature scaling to the teacher's logits
    before computing the loss between teacher and student predictions. It's the
    most common approach for knowledge distillation.

    Arguments:
        temperature: Temperature for softmax scaling. Higher values produce
            softer probability distributions that are easier for the student to
            learn. Typical values range from 3-5. Defaults to 3.0.
        loss: Loss function to use for distillation. Can be:
            - String identifier (e.g., 'kl_divergence',
              'categorical_crossentropy')
            - Keras loss instance
            - Nested structure of losses matching the model output structure
            - `None` to skip distillation for that output (useful for
              multi-output models where you only want to distill some outputs)
            At least one loss must be non-`None`. Defaults to 'kl_divergence'.

    KL divergence and categorical crossentropy losses take the teacher
    probabilities `softmax(teacher_logits / temperature)` as the target and
    the student predictions at the same temperature. Their result is scaled
    by `temperature**2`. All other losses take the temperature-scaled logits
    of both models. Sparse categorical, binary and focal crossentropy losses
    are not supported.

    Examlpe(s):

    ```python
    # Basic logits distillation with KL divergence
    distillation_loss = LogitsDistillation(temperature=3.0)

    # With categorical crossentropy loss
    distillation_loss = LogitsDistillation(
        temperature=4.0,
        loss="categorical_crossentropy"
    )

    # With custom loss instance
    distillation_loss = LogitsDistillation(
        temperature=4.0,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True)
    )

    # For multi-output models
    distillation_loss = LogitsDistillation(
        temperature=3.0,
        loss=["kl_divergence", "categorical_crossentropy"]
    )

    # For multi-output models, only distill some outputs
    distillation_loss = LogitsDistillation(
        temperature=3.0,
        loss=["kl_divergence", None]  # Skip second output
    )
    ```
    """

    @tracking.no_automatic_dependency_tracking
    def __init__(
        self,
        temperature=3.0,
        loss="kl_divergence",
    ):
        self.temperature = temperature
        self.loss = tree.map_structure(_convert_loss_to_function, loss)

        flat_losses = tree.flatten(self.loss)
        if all(l is None for l in flat_losses):
            raise ValueError("At least one loss must be non-`None`.")
        for loss_fn in flat_losses:
            if _unwrap_loss_fn(loss_fn) in _UNSUPPORTED_LOGITS_LOSSES:
                loss_name = getattr(
                    loss_fn, "name", getattr(loss_fn, "__name__", loss_fn)
                )
                raise ValueError(
                    f"`LogitsDistillation` does not support the loss "
                    f"'{loss_name}'. Its targets cannot be derived from the "
                    f"teacher logits. Use 'kl_divergence' or "
                    f"'categorical_crossentropy' instead."
                )

        if not isinstance(self.temperature, (int, float)):
            raise ValueError(
                f"temperature must be a number, got {type(self.temperature)}"
            )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")

    def compute_loss(self, teacher_outputs, student_outputs, **kwargs):
        """Compute distillation loss using the configured loss function.

        Arguments:
            teacher_outputs: Logits from teacher model. Can be a single tensor,
                list/tuple of tensors, or dict of tensors.
            student_outputs: Logits from student model. Can be a single tensor,
                list/tuple of tensors, or dict of tensors.
            **kwargs: Additional arguments (ignored).
        Returns:
            Distillation loss tensor.
        """
        # Apply temperature scaling using tree.map_structure
        teacher_scaled = tree.map_structure(
            lambda x: backend.ops.numpy.divide(x, self.temperature),
            teacher_outputs,
        )
        student_scaled = tree.map_structure(
            lambda x: backend.ops.numpy.divide(x, self.temperature),
            student_outputs,
        )

        # Apply loss function(s) to corresponding outputs
        def apply_loss(loss_fn, teacher_logits, student_logits):
            if loss_fn is None:
                return 0.0

            fn = _unwrap_loss_fn(loss_fn)
            if fn is kl_divergence:
                # KL divergence needs probabilities for both models.
                teacher_probs = backend.ops.nn.softmax(teacher_logits, axis=-1)
                student_probs = backend.ops.nn.softmax(student_logits, axis=-1)
                loss = loss_fn(teacher_probs, student_probs)
            elif fn is categorical_crossentropy:
                # Crossentropy needs teacher probabilities as the target.
                teacher_probs = backend.ops.nn.softmax(teacher_logits, axis=-1)
                if loss_fn is fn:
                    loss = loss_fn(
                        teacher_probs, student_logits, from_logits=True
                    )
                elif getattr(loss_fn, "from_logits", False):
                    loss = loss_fn(teacher_probs, student_logits)
                else:
                    student_probs = backend.ops.nn.softmax(
                        student_logits, axis=-1
                    )
                    loss = loss_fn(teacher_probs, student_probs)
            else:
                # For other losses, use logits directly
                return backend.ops.numpy.mean(
                    loss_fn(teacher_logits, student_logits)
                )
            # Scale by temperature^2 to keep the gradient magnitude
            # independent of the temperature (per literature)
            return backend.ops.numpy.mean(loss) * (self.temperature**2)

        # Apply losses using tree.map_structure
        loss_values = tree.map_structure(
            apply_loss, self.loss, teacher_scaled, student_scaled
        )

        # Sum all losses and return scalar
        flat_losses = tree.flatten(loss_values)
        return backend.ops.numpy.sum(backend.ops.numpy.stack(flat_losses))

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "temperature": self.temperature,
            "loss": serialization_lib.serialize_keras_object(self.loss),
        }

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        config = config.copy()
        config["loss"] = keras.losses.deserialize(config["loss"])
        return cls(**config)
