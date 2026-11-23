"""Synthetic tasks and train-only input scaling."""

import numpy as np
import torch

from . import config as c


def _components(x):
    i, j, k = c.ADDITIVE_FEATURES
    (a, b), (d, e), pair = c.PLANTED_PAIRS
    angle = c.GAUSSIAN_ANGLE
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    principal = (x[:, pair] - c.GAUSSIAN_CENTER) @ rotation
    gaussian = c.GAUSSIAN_AMPLITUDE * np.exp(
        -0.5 * (principal**2 @ np.array(c.GAUSSIAN_EIGENVALUES))
    )
    return np.column_stack((
        np.sin(x[:, i]), c.QUADRATIC_WEIGHT * (x[:, j]**2 - 1), np.tanh(x[:, k]),
        x[:, a] * x[:, b], np.sin(x[:, d] * x[:, e]), gaussian,
    ))


def _sum_components(task, standardized):
    if task == "additive":
        return standardized[:, :3].sum(axis=1)
    if task in ("sparse", "sparse_overlap"):
        return standardized.sum(axis=1)
    raise ValueError(f"Unknown task {task!r}")


def _overlap_components(x, additive):
    (a, b), (d, e), (f, g) = c.OVERLAP_PAIRS
    return np.column_stack((
        additive,
        np.tanh(x[:, a]) * np.tanh(x[:, b]),
        np.exp(-0.5 * ((x[:, d] - x[:, e]) / c.RIDGE_WIDTH)**2),
        np.sin(c.OSCILLATION_RATE * x[:, f]) * np.sin(c.OSCILLATION_RATE * x[:, g]),
    ))


def estimate_reference(tasks=c.TASKS):
    x = np.random.default_rng(c.REFERENCE_SEED).standard_normal((c.REFERENCE_SIZE, c.FEATURES))
    components = _components(x)
    mean, sd = components.mean(axis=0), components.std(axis=0)
    constants = {"component_mean": mean.tolist(), "component_sd": sd.tolist(), "targets": {}}
    for task in tasks:
        if task == "sparse_overlap":
            overlap = _overlap_components(x, components[:, :3])
            overlap_mean, overlap_sd = overlap.mean(axis=0), overlap.std(axis=0)
            constants["overlap"] = {"component_mean": overlap_mean.tolist(), "component_sd": overlap_sd.tolist()}
            standardized = (overlap - overlap_mean) / overlap_sd
        else:
            standardized = (components - mean) / sd
        total = _sum_components(task, standardized)
        constants["targets"][task] = {"mean": float(total.mean()), "sd": float(total.std())}
    return constants


def make_data(task, seed, reference, sizes=c.SPLIT_SIZES, noise_sd=c.NOISE_SD):
    """Return (x, observed y) per split and evaluator-only diagnostics."""
    feature_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    x = np.random.default_rng(feature_seed).standard_normal((sum(sizes), c.FEATURES))
    noise = np.random.default_rng(noise_seed).normal(0, noise_sd, len(x))
    components = _components(x)
    component_stats = reference
    if task == "sparse_overlap":
        components = _overlap_components(x, components[:, :3])
        component_stats = reference["overlap"]
    standardized = (components - component_stats["component_mean"]) / component_stats["component_sd"]
    constants = reference["targets"][task]
    clean = (_sum_components(task, standardized) - constants["mean"]) / constants["sd"]

    train_x = x[:sizes[0]]
    mean, sd = train_x.mean(axis=0), train_x.std(axis=0)
    scaled_x = torch.from_numpy(((x - mean) / sd).astype(np.float32)).split(sizes)
    observed_y = torch.from_numpy((clean + noise).astype(np.float32)).split(sizes)
    bounds = np.cumsum((0, *sizes))
    diagnostics = [
        {"split": split, "signal_variance": float(clean[start:end].var()),
         "noise_variance": float(noise[start:end].var())}
        for split, start, end in zip(("train", "validation", "test"), bounds[:-1], bounds[1:])
    ]
    return tuple(zip(scaled_x, observed_y)), diagnostics


def planted_pairs(task):
    return {"sparse": c.PLANTED_PAIRS, "sparse_overlap": c.OVERLAP_PAIRS, "additive": ()}[task]
