"""Population targets and shared, seed-determined splits."""

import numpy as np
import torch

from . import config as c

CENTER = np.array(c.GAUSSIAN_CENTER)
PRECISION = np.array(c.GAUSSIAN_PRECISION)
FREQUENCIES = 0.6 + 0.1 * np.arange(c.FEATURES)
UNPAIRED = sorted(set(range(c.FEATURES)) - set(sum(c.PLANTED, ())))


def gaussian(x):
    delta = x - CENTER
    return c.GAUSSIAN_AMPLITUDE * np.exp(-0.5 * np.einsum("...i,ij,...j->...", delta, PRECISION, delta))


def gaussian_moment(order):
    scaled = order * PRECISION
    system = np.eye(2) + scaled
    exponent = CENTER @ scaled @ np.linalg.solve(system, CENTER)
    return c.GAUSSIAN_AMPLITUDE**order * np.exp(-0.5 * exponent) / np.sqrt(np.linalg.det(system))


def additive_effect(x, indices):
    rates = FREQUENCIES[indices]
    variance = (-0.5 * np.expm1(-2 * rates**2)).sum()
    return np.sin(x[:, indices] * rates).sum(axis=1) / np.sqrt(variance)


def target(name, x):
    if name == "additive":
        return additive_effect(x, slice(None))
    if name != "sparse":
        raise ValueError(f"Unknown dataset {name!r}")
    product = x[:, c.PLANTED[0]].prod(axis=1)
    frequency = 1.4
    sine_sd = np.sqrt(0.5 * (1 - (1 + 4 * frequency**2)**-0.5))
    sine = np.sin(frequency * x[:, c.PLANTED[1]].prod(axis=1)) / sine_sd
    gaussian_mean = gaussian_moment(1)
    gaussian_sd = np.sqrt(gaussian_moment(2) - gaussian_mean**2)
    rotated = (gaussian(x[:, c.PLANTED[2]]) - gaussian_mean) / gaussian_sd
    # Disjoint feature sets make these centered unit-variance terms independent.
    return (product + sine + rotated + 0.5 * additive_effect(x, UNPAIRED)) / np.sqrt(3.25)


def make_data(name, seed, sizes=c.SIZES, noise_sd=c.NOISE_SD):
    feature_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    x = np.random.default_rng(feature_seed).standard_normal((sum(sizes), c.FEATURES))
    noise = np.random.default_rng(noise_seed).normal(0, noise_sd, len(x))
    clean = target(name, x)
    train_x = x[:sizes[0]]
    mean, sd = train_x.mean(axis=0), train_x.std(axis=0)
    columns = [(x - mean) / sd, clean + noise, clean]
    tensors = [torch.tensor(column, dtype=torch.float32).split(sizes) for column in columns]
    # Population N(0, 1) expressed in the train-standardized coordinates.
    return tuple(zip(*tensors)), torch.tensor(-mean / sd), torch.tensor(1 / sd)
