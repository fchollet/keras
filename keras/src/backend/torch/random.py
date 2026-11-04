import torch
import torch._dynamo as dynamo
import torch.nn.functional as tnn

from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.config import floatx
from keras.src.backend.torch.ops.core import convert_to_tensor
from keras.src.backend.torch.ops.core import get_device
from keras.src.backend.torch.ops.core import to_torch_dtype
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


# torch.Generator not supported with dynamo
# see: https://github.com/pytorch/pytorch/issues/88576
@dynamo.disable()
def torch_seed_generator(seed):
    device = get_device()
    if device == "meta":
        # Generator is not supported by the meta device.
        return None
    generator = torch.Generator(device=get_device())
    first_seed, second_seed = draw_seed(seed)
    # Re-interpret as uint32 and combine; the counter in second_seed ensures
    # each SeedGenerator call produces a distinct seed.
    generator.manual_seed(int(first_seed + second_seed) & 0xFFFFFFFF)
    return generator


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.normal(
            mean, stddev, size=shape, dtype=dtype, device=get_device()
        )
    generator = torch_seed_generator(seed)
    return torch.normal(
        mean,
        stddev,
        size=shape,
        generator=generator,
        dtype=dtype,
        device=get_device(),
    )


def categorical(logits, num_samples, dtype="int32", seed=None):
    logits = convert_to_tensor(logits)
    dtype = to_torch_dtype(dtype)
    probs = torch.softmax(logits, dim=-1)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.multinomial(
            probs,
            num_samples,
            replacement=True,
        ).type(dtype)
    generator = torch_seed_generator(seed)
    return torch.multinomial(
        probs,
        num_samples,
        replacement=True,
        generator=generator,
    ).type(dtype)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    requested_shape = shape
    if len(requested_shape) == 0:
        shape = (1,)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        rand_tensor = torch.rand(size=shape, dtype=dtype, device=get_device())
    else:
        generator = torch_seed_generator(seed)
        rand_tensor = torch.rand(
            size=shape, generator=generator, dtype=dtype, device=get_device()
        )

    output = (maxval - minval) * rand_tensor + minval

    if len(requested_shape) == 0:
        return output[0]
    return output


def randint(shape, minval, maxval, dtype="int32", seed=None):
    dtype = to_torch_dtype(dtype)
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        return torch.randint(
            low=minval,
            high=maxval,
            size=shape,
            dtype=dtype,
            device=get_device(),
        )
    generator = torch_seed_generator(seed)
    return torch.randint(
        low=minval,
        high=maxval,
        size=shape,
        generator=generator,
        dtype=dtype,
        device=get_device(),
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = to_torch_dtype(dtype)
    # Take a larger standard normal dist, discard values outside 2 * stddev
    # Offset by mean and stddev
    x = normal(tuple(shape) + (4,), mean=0, stddev=1, dtype=dtype, seed=seed)
    valid = (x > -2) & (x < 2)
    indexes = valid.max(-1, keepdim=True)[1]
    trunc_x = torch.empty(shape, dtype=dtype, device=get_device())
    trunc_x.data.copy_(x.gather(-1, indexes).squeeze(-1))
    trunc_x.data.mul_(stddev).add_(mean)
    return trunc_x


def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    inputs = convert_to_tensor(inputs)
    if rate == 1.0:
        return torch.zeros_like(inputs, device=get_device())
    if rate == 0.0:
        return inputs
    if (
        seed is not None
        and not (isinstance(seed, SeedGenerator) and seed._initial_seed is None)
        or noise_shape is not None
    ):
        keep_prob = 1.0 - rate
        noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
        keep_prob_matrix = torch.full(
            noise_shape, keep_prob, device=get_device()
        )
        generator = torch_seed_generator(seed)

        # Do not use generator during symbolic execution.
        if get_device() == "meta":
            mask = torch.bernoulli(keep_prob_matrix)
        else:
            mask = torch.bernoulli(keep_prob_matrix, generator=generator)

        mask = mask.bool()
        mask = torch.broadcast_to(mask, inputs.shape)
        return torch.where(
            mask,
            inputs / keep_prob,
            torch.zeros_like(inputs, dtype=inputs.dtype),
        )
    # Fast path, unseeded (since torch doesn't support seeding dropout!!!!)
    # Using the above implementation is possible, but much slower.
    return torch.nn.functional.dropout(
        inputs, p=rate, training=True, inplace=False
    )


def shuffle(x, axis=0, seed=None):
    # Ref: https://github.com/pytorch/pytorch/issues/71409
    x = convert_to_tensor(x)

    axis = canonicalize_axis(axis, x.ndim)

    # Get permutation indices
    # Do not use generator during symbolic execution.
    if get_device() == "meta":
        row_perm = torch.rand(x.shape[: axis + 1], device=get_device()).argsort(
            axis
        )
    else:
        generator = torch_seed_generator(seed)
        row_perm = torch.rand(
            x.shape[: axis + 1], generator=generator, device=get_device()
        ).argsort(axis)
    for _ in range(x.ndim - axis - 1):
        row_perm.unsqueeze_(-1)

    # Reformat this for the gather operation
    row_perm = row_perm.repeat(
        *[1 for _ in range(axis + 1)], *(x.shape[axis + 1 :])
    )
    return x.gather(axis, row_perm)


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    alpha = torch.broadcast_to(convert_to_tensor(alpha), shape)
    beta = torch.ones(shape, device=get_device())
    prev_rng_state = torch.random.get_rng_state()
    # Do not draw seed during symbolic execution
    if not get_device() == "meta":
        first_seed, second_seed = draw_seed(seed)
        torch.manual_seed(int(first_seed + second_seed) & 0xFFFFFFFF)
    gamma_distribution = torch.distributions.gamma.Gamma(alpha, beta)
    sample = gamma_distribution.sample().type(dtype)
    torch.random.set_rng_state(prev_rng_state)
    return sample


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    counts = torch.broadcast_to(convert_to_tensor(counts), shape)
    probabilities = torch.broadcast_to(convert_to_tensor(probabilities), shape)
    prev_rng_state = torch.random.get_rng_state()
    # Do not draw seed during symbolic execution
    if not get_device() == "meta":
        first_seed, second_seed = draw_seed(seed)
        torch.manual_seed(int(first_seed + second_seed) & 0xFFFFFFFF)
    binomial_distribution = torch.distributions.binomial.Binomial(
        total_count=counts, probs=probabilities
    )
    sample = binomial_distribution.sample().type(dtype)
    torch.random.set_rng_state(prev_rng_state)
    return sample


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    dtype = to_torch_dtype(dtype)
    alpha = torch.broadcast_to(convert_to_tensor(alpha), shape)
    beta = torch.broadcast_to(convert_to_tensor(beta), shape)
    prev_rng_state = torch.random.get_rng_state()
    # Do not draw seed during symbolic execution
    if not get_device() == "meta":
        first_seed, second_seed = draw_seed(seed)
        torch.manual_seed(int(first_seed + second_seed) & 0xFFFFFFFF)
    beta_distribution = torch.distributions.beta.Beta(
        concentration1=alpha, concentration0=beta
    )
    sample = beta_distribution.sample().type(dtype)
    torch.random.set_rng_state(prev_rng_state)
    return sample
