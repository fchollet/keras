import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils import dataset_utils
from keras.src.utils import file_utils
from keras.src.utils.grain_utils import make_batch
from keras.src.utils.module_utils import grain
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.module_utils import tensorflow_io as tfio

try:
    from scipy import signal
    from scipy.io import wavfile
except ImportError:
    signal = None
    wavfile = None

ALLOWED_FORMATS = (".wav",)

# When resampling, the target length is `len(audio) * sampling_rate /
# file_sample_rate`, and `file_sample_rate` is read straight from the
# (untrusted) WAV header. A file that declares an absurdly small rate turns a
# small clip into an enormous resample target, which the resampler then
# allocates -- e.g. a 2 MB file claiming 1 Hz resampled to 16 kHz asks for
# ~1.6e10 samples (~128 GB). A target is rejected as such a bomb only when it
# exceeds BOTH this absolute floor AND this multiple of the stored samples,
# mirroring the dual condition used by `saving_lib._reject_zip_bomb` (CWE-409).
_AUDIO_RESAMPLE_BOMB_FLOOR = 1 << 28  # 268M samples (~93 min @ 48 kHz)
_AUDIO_RESAMPLE_MAX_EXPANSION = 1000


def _as_control_inputs(assert_op):
    """Returns `assert_op` as a `tf.control_dependencies` input list.

    `tf.debugging` asserts return `None` when they run eagerly, and
    `tf.control_dependencies` rejects a `None` entry.
    """
    return [] if assert_op is None else [assert_op]


def _reject_resample_bomb(num_samples, input_samples):
    if (
        num_samples > _AUDIO_RESAMPLE_BOMB_FLOOR
        and num_samples > _AUDIO_RESAMPLE_MAX_EXPANSION * max(1, input_samples)
    ):
        raise ValueError(
            f"Refusing to resample audio to {num_samples} samples from "
            f"{input_samples} stored samples; the requested expansion looks "
            "like a decompression bomb driven by an implausible file sample "
            "rate."
        )


@keras_export("keras.utils.audio_dataset_from_directory")
def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
    format="tf",
    verbose=True,
):
    """Generates a dataset from audio files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav
    ```

    Then calling `audio_dataset_from_directory(main_directory,
    labels='inferred')`
    will return a dataset that yields batches of audio files from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.wav` files are supported at this time.

    By default, this function will return a `tf.data.Dataset` object. You can
    set `format="grain"` to return a `grain.IterDataset` object instead, which
    removes the TensorFlow dependency.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain subdirectories,
            each containing audio files for a class. Otherwise, the directory
            structure is ignored.
        labels: Either "inferred" (labels are generated from the directory
            structure), `None` (no labels), or a list/tuple of integer labels
            of the same size as the number of audio files found in
            the directory. Labels should be sorted according to the
            alphanumeric order of the audio file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers (e.g. for
              `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are encoded as a categorical
              vector (e.g. for `categorical_crossentropy` loss)
            - `"binary"` means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0
              or 1 (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if "labels" is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Default: 32. If `None`,
            the data will not be batched
            (the dataset will yield individual samples).
        sampling_rate: Audio sampling rate (in samples per second).
        output_sequence_length: Maximum length of an audio sequence. Audio files
            longer than this will be truncated to `output_sequence_length`,
            after resampling if `sampling_rate` is set.
            If set to `None`, then all sequences in the same batch will
            be padded to the
            length of the longest sequence in the batch.
        ragged: Whether to return a Ragged dataset (where each sequence has its
            own length). Defaults to `False`.
        shuffle: Whether to shuffle the data.
            If set to `False`, sorts the data in alphanumeric order.
            Defaults to `True`.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1, fraction of data to
            reserve for validation.
        subset: Subset of the data to return. One of `"training"`,
            `"validation"` or `"both"`. Only used if `validation_split` is set.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.
        format: The format of the return object. Defaults to `"tf"`. Available
            options are:
            - `"tf"`: returns a `tf.data.Dataset` object. Requires
                TensorFlow to be installed.
            - `"grain"`: returns a `grain.IterDataset` object. Requires
                Grain to be installed.
        verbose: Whether to display number information on classes and
            number of files found. Defaults to `True`.

    Returns:

    A `tf.data.Dataset` (`format="tf"`) or `grain.IterDataset`
    (`format="grain"`) object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
      `(batch_size,)`, containing the contents of a batch of audio files.
    - Otherwise, it yields a tuple `(audio, labels)`, where `audio`
      has shape `(batch_size, sequence_length, num_channels)` and `labels`
      follows the format described
      below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
    """
    if format not in ("tf", "grain"):
        raise ValueError(
            '`format` should be either "tf" or "grain". '
            f"Received: format={format}"
        )

    if format == "grain":
        if ragged:
            raise ValueError(
                "Ragged datasets are not supported when `format='grain'`."
            )
        if wavfile is None or signal is None:
            raise ImportError(
                "To use the grain audio dataset, you should install scipy. "
                "You can install it via `pip install scipy`."
            )
        if file_utils.is_remote_path(directory):
            raise ValueError(
                "Remote directories are not supported when `format='grain'`."
            )

    if labels not in ("inferred", None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
                "The `labels` argument should be a list/tuple of integer "
                "labels, of the same size as the number of audio files in "
                "the target directory. If you wish to infer the labels from "
                "the subdirectory names in the target directory,"
                ' pass `labels="inferred"`. '
                "If you wish to get a dataset that only contains audio samples "
                f"(no labels), pass `labels=None`. Received: labels={labels}"
            )
        if class_names:
            raise ValueError(
                "You can only pass `class_names` if "
                f'`labels="inferred"`. Received: labels={labels}, and '
                f"class_names={class_names}"
            )
    if label_mode not in {"int", "categorical", "binary", None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", '
            '"binary", '
            f"or None. Received: label_mode={label_mode}"
        )

    if ragged and output_sequence_length is not None:
        raise ValueError(
            "Cannot set both `ragged` and `output_sequence_length`"
        )

    if sampling_rate is not None:
        if not isinstance(sampling_rate, int):
            raise ValueError(
                "`sampling_rate` should have an integer value. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if sampling_rate <= 0:
            raise ValueError(
                "`sampling_rate` should be higher than 0. "
                f"Received: sampling_rate={sampling_rate}"
            )

        if format == "tf" and not tfio.available:
            raise ImportError(
                "To use the argument `sampling_rate`, you should install "
                "tensorflow_io. You can install it via `pip install "
                "tensorflow-io`."
            )

    if labels is None or label_mode is None:
        labels = None
        label_mode = None

    dataset_utils.check_validation_split_arg(
        validation_split, subset, shuffle, seed
    )

    if seed is None:
        seed = np.random.randint(1e6)
    if batch_size is not None:
        shuffle_buffer_size = batch_size * 8
    else:
        shuffle_buffer_size = 1024

    file_paths, labels, class_names = dataset_utils.index_directory(
        directory,
        labels,
        formats=ALLOWED_FORMATS,
        class_names=class_names,
        shuffle=shuffle,
        seed=seed,
        follow_links=follow_links,
        verbose=verbose,
    )

    if label_mode == "binary" and len(class_names) != 2:
        raise ValueError(
            'When passing `label_mode="binary"`, there must be exactly 2 '
            f"class_names. Received: class_names={class_names}"
        )

    if subset == "both":
        train_dataset, val_dataset = get_training_and_validation_dataset(
            file_paths=file_paths,
            labels=labels,
            validation_split=validation_split,
            directory=directory,
            label_mode=label_mode,
            class_names=class_names,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            format=format,
        )
        if format == "tf":
            train_dataset = _prepare_dataset_tf(
                dataset=train_dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
                ragged=ragged,
            )
            val_dataset = _prepare_dataset_tf(
                dataset=val_dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
                ragged=ragged,
            )
        else:
            train_dataset = _prepare_dataset_grain(
                dataset=train_dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
            )
            val_dataset = _prepare_dataset_grain(
                dataset=val_dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
            )
        return train_dataset, val_dataset

    else:
        dataset = get_dataset(
            file_paths=file_paths,
            labels=labels,
            directory=directory,
            validation_split=validation_split,
            subset=subset,
            label_mode=label_mode,
            class_names=class_names,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            format=format,
        )
        if format == "tf":
            dataset = _prepare_dataset_tf(
                dataset=dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
                ragged=ragged,
            )
        else:
            dataset = _prepare_dataset_grain(
                dataset=dataset,
                batch_size=batch_size,
                class_names=class_names,
                output_sequence_length=output_sequence_length,
            )
        return dataset


def _prepare_dataset_tf(
    dataset,
    batch_size,
    class_names,
    output_sequence_length,
    ragged,
):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if output_sequence_length is None and not ragged:
            dataset = dataset.padded_batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset


def _prepare_dataset_grain(
    dataset,
    batch_size,
    class_names,
    output_sequence_length,
):
    dataset = dataset.to_iter_dataset()
    if batch_size is not None:
        if output_sequence_length is None:
            dataset = dataset.batch(
                batch_size, batch_fn=_make_audio_padded_batch_grain
            )
        else:
            dataset = dataset.batch(batch_size, batch_fn=make_batch)

    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset


def get_training_and_validation_dataset(
    file_paths,
    labels,
    validation_split,
    directory,
    label_mode,
    class_names,
    sampling_rate,
    output_sequence_length,
    ragged,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
    format="tf",
):
    (
        file_paths_train,
        labels_train,
    ) = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "training"
    )
    if not file_paths_train:
        raise ValueError(
            f"No training audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    file_paths_val, labels_val = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, "validation"
    )
    if not file_paths_val:
        raise ValueError(
            f"No validation audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    train_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_train,
        labels=labels_train,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        format=format,
    )

    val_dataset = paths_and_labels_to_dataset(
        file_paths=file_paths_val,
        labels=labels_val,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
        shuffle=False,
        format=format,
    )

    return train_dataset, val_dataset


def get_dataset(
    file_paths,
    labels,
    directory,
    validation_split,
    subset,
    label_mode,
    class_names,
    sampling_rate,
    output_sequence_length,
    ragged,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
    format="tf",
):
    file_paths, labels = dataset_utils.get_training_or_validation_split(
        file_paths, labels, validation_split, subset
    )
    if not file_paths:
        raise ValueError(
            f"No audio files found in directory {directory}. "
            f"Allowed format(s): {ALLOWED_FORMATS}"
        )

    return paths_and_labels_to_dataset(
        file_paths=file_paths,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names) if class_names else 0,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        format=format,
    )


def paths_and_labels_to_dataset(
    file_paths,
    labels,
    label_mode,
    num_classes,
    sampling_rate,
    output_sequence_length,
    ragged,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
    format="tf",
):
    """Constructs a fixed-size dataset of audio and labels."""
    if format == "tf":
        return _paths_and_labels_to_dataset_tf(
            file_paths=file_paths,
            labels=labels,
            label_mode=label_mode,
            num_classes=num_classes,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            ragged=ragged,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
    elif format == "grain":
        return _paths_and_labels_to_dataset_grain(
            file_paths=file_paths,
            labels=labels,
            label_mode=label_mode,
            num_classes=num_classes,
            sampling_rate=sampling_rate,
            output_sequence_length=output_sequence_length,
            shuffle=shuffle,
            seed=seed,
        )
    else:
        raise ValueError(
            '`format` should be either "tf" or "grain". '
            f"Received: format={format}"
        )


def _paths_and_labels_to_dataset_tf(
    file_paths,
    labels,
    label_mode,
    num_classes,
    sampling_rate,
    output_sequence_length,
    ragged,
    shuffle=False,
    shuffle_buffer_size=None,
    seed=None,
):
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_tf(
            labels, label_mode, num_classes
        )
        ds = tf.data.Dataset.zip((path_ds, label_ds))
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size or 1024, seed=seed)

    if label_mode:
        ds = ds.map(
            lambda x, y: (
                _read_and_decode_audio_tf(
                    x, sampling_rate, output_sequence_length
                ),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if ragged:
            ds = ds.map(
                lambda x, y: (tf.RaggedTensor.from_tensor(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    else:
        ds = ds.map(
            lambda x: _read_and_decode_audio_tf(
                x, sampling_rate, output_sequence_length
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if ragged:
            ds = ds.map(
                lambda x: tf.RaggedTensor.from_tensor(x),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

    return ds


def _read_and_decode_audio_tf(
    path, sampling_rate=None, output_sequence_length=None
):
    """Reads and decodes audio file."""
    audio = tf.io.read_file(path)

    if output_sequence_length is not None and sampling_rate is None:
        desired_samples = output_sequence_length
    else:
        desired_samples = -1

    audio, default_audio_rate = tf.audio.decode_wav(
        contents=audio, desired_samples=desired_samples
    )
    if sampling_rate is not None:
        # default_audio_rate should have dtype=int64
        default_audio_rate = tf.cast(default_audio_rate, tf.int64)
        # `rate_in` is read from the (untrusted) WAV header. Reject a
        # non-positive rate and a resample target that looks like a
        # decompression bomb before `resample` allocates it (see
        # `_reject_resample_bomb`). Both checks are wired in as explicit
        # control dependencies so that in graph mode they run before the
        # division they protect and before the resample itself; the asserts
        # return `None` in eager mode, where they have already run.
        rate_is_positive = tf.debugging.assert_positive(
            default_audio_rate, message="Invalid WAV file sample rate."
        )
        with tf.control_dependencies(_as_control_inputs(rate_is_positive)):
            input_samples = tf.cast(tf.shape(audio)[0], tf.int64)
            num_samples = (
                input_samples * tf.cast(sampling_rate, tf.int64)
            ) // default_audio_rate
        is_bomb = tf.logical_and(
            num_samples > _AUDIO_RESAMPLE_BOMB_FLOOR,
            num_samples
            > _AUDIO_RESAMPLE_MAX_EXPANSION * tf.maximum(input_samples, 1),
        )
        is_not_bomb = tf.debugging.Assert(
            tf.logical_not(is_bomb),
            [
                "Refusing to resample audio: requested expansion looks like a "
                "decompression bomb driven by an implausible file sample rate. "
                "Target samples:",
                num_samples,
                "stored samples:",
                input_samples,
            ],
        )
        with tf.control_dependencies(_as_control_inputs(is_not_bomb)):
            audio = tfio.audio.resample(
                input=audio, rate_in=default_audio_rate, rate_out=sampling_rate
            )
        if output_sequence_length is not None:
            audio = _trim_and_pad_audio(audio, output_sequence_length)
    return audio


def _trim_and_pad_audio(audio, output_sequence_length):
    audio = audio[:output_sequence_length]
    padding = output_sequence_length - tf.shape(audio)[0]
    audio = tf.pad(audio, [[0, padding], [0, 0]])
    audio.set_shape((output_sequence_length, None))
    return audio


def _paths_and_labels_to_dataset_grain(
    file_paths,
    labels,
    label_mode,
    num_classes,
    sampling_rate,
    output_sequence_length,
    shuffle=False,
    seed=None,
):
    """Constructs a dataset of audio and labels using grain."""
    path_ds = grain.MapDataset.source(file_paths)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset_grain(
            labels, label_mode, num_classes
        )
        ds = grain.experimental.ZipMapDataset([path_ds, label_ds])
    else:
        ds = path_ds

    if shuffle:
        ds = ds.shuffle(seed=seed)

    args = (sampling_rate, output_sequence_length)
    if label_mode:
        ds = ds.map(
            lambda data: (
                _read_and_decode_audio_grain(data[0], *args),
                data[1],
            )
        )
    else:
        ds = ds.map(lambda x: _read_and_decode_audio_grain(x, *args))

    return ds


def _read_and_decode_audio_grain(
    path, sampling_rate=None, output_sequence_length=None
):
    """Reads and decodes audio file using scipy."""
    from keras.src import backend
    from keras.src import ops

    if wavfile is None or signal is None:
        raise ImportError(
            "To use the grain audio dataset, you should install scipy. "
            "You can install it via `pip install scipy`."
        )

    if hasattr(path, "decode"):
        path = path.decode("utf-8")
    else:
        path = str(path)

    default_audio_rate, audio = wavfile.read(path)
    default_audio_rate = int(default_audio_rate)
    if default_audio_rate <= 0:
        raise ValueError(
            f"Invalid sample rate {default_audio_rate} in WAV file '{path}'."
        )

    # Ensure audio is float32 and normalized between -1.0 and 1.0
    if audio.dtype == np.int16:
        audio = audio.astype("float32") / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype("float32") / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype("float32") - 128.0) / 128.0
    elif audio.dtype == np.float32:
        audio = audio.astype("float32")
    else:
        audio = audio.astype("float32")

    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=-1)

    # Resample if needed
    if sampling_rate is not None and default_audio_rate != sampling_rate:
        if len(audio) > 0:
            num_samples = int(
                round(len(audio) * sampling_rate / default_audio_rate)
            )
            num_samples = max(1, num_samples)
            _reject_resample_bomb(num_samples, len(audio))
            audio = signal.resample(audio, num_samples, axis=0)
            audio = audio.astype("float32")

    # Trim/pad to output_sequence_length
    if output_sequence_length is not None:
        diff = output_sequence_length - len(audio)
        if diff > 0:
            padding = np.zeros((diff, audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, padding], axis=0)
        elif diff < 0:
            audio = audio[:output_sequence_length]

    with backend.device_scope("cpu"):
        audio = ops.convert_to_tensor(audio, dtype="float32")

    return audio


def _make_audio_padded_batch_grain(values):
    from keras.src import backend
    from keras.src import ops
    from keras.src import tree

    if not values:
        raise ValueError("Cannot batch 0 values. Please file a bug.")

    def pad_and_stack(*xs):
        shapes = [x.shape for x in xs]
        if len(shapes[0]) == 2:
            max_len = max(shape[0] for shape in shapes)
            padded_xs = []
            for x in xs:
                pad_len = max_len - x.shape[0]
                if pad_len > 0:
                    padded_xs.append(ops.pad(x, [[0, pad_len], [0, 0]]))
                else:
                    padded_xs.append(x)
            return ops.stack(padded_xs)
        else:
            return ops.stack(xs)

    with backend.device_scope("cpu"):
        return tree.map_structure(pad_and_stack, *values)
