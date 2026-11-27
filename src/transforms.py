from __future__ import annotations

from typing import Any

import numpy as np


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def augment_ecg(signal: np.ndarray, config: dict[str, Any], seed: int | None = None) -> np.ndarray:
    """Apply physiologically plausible training augmentation to a 12 x T signal."""
    if not config.get("enabled", True):
        return signal
    rng = _rng(seed)
    x = signal.astype(np.float32, copy=True)
    fs = 500

    if rng.random() < config.get("gaussian_noise_prob", 0.0):
        sigma = float(config.get("gaussian_noise_sigma_mv", 0.01))
        x += rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

    if rng.random() < config.get("amplitude_scale_prob", 0.0):
        low = float(config.get("amplitude_scale_min", 0.9))
        high = float(config.get("amplitude_scale_max", 1.1))
        x *= rng.uniform(low, high)

    if rng.random() < config.get("time_shift_prob", 0.0):
        max_shift = int(round(float(config.get("time_shift_max_s", 0.2)) * fs))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        x = np.roll(x, shift=shift, axis=-1)

    if rng.random() < config.get("baseline_wander_prob", 0.0):
        amplitude = float(config.get("baseline_wander_amplitude_mv", 0.05))
        frequency = rng.uniform(0.08, 0.33)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        t = np.arange(x.shape[-1], dtype=np.float32) / fs
        wander = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
        x += wander[None, :]

    if rng.random() < config.get("lead_dropout_prob", 0.0):
        max_drop = int(config.get("lead_dropout_max", 1))
        count = int(rng.integers(1, max_drop + 1))
        leads = rng.choice(x.shape[0], size=count, replace=False)
        x[leads] = 0.0

    return np.clip(x, -5.0, 5.0)


def perturb_ecg(
    signal: np.ndarray,
    spec: dict[str, Any],
    seed: int | None = None,
) -> np.ndarray:
    """Apply a deterministic robustness perturbation."""
    rng = _rng(seed)
    x = signal.astype(np.float32, copy=True)
    kind = spec.get("name", "clean")
    fs = int(spec.get("sampling_rate_hz", 500))

    if kind == "clean":
        return x
    if kind == "gaussian_noise":
        x += rng.normal(0.0, float(spec["sigma_mv"]), size=x.shape).astype(np.float32)
    elif kind == "amplitude_scale":
        x *= float(spec["factor"])
    elif kind == "time_shift":
        shift = int(round(float(spec["seconds"]) * fs))
        x = np.roll(x, shift=shift, axis=-1)
    elif kind == "baseline_wander":
        amplitude = float(spec.get("amplitude_mv", 0.1))
        frequency = float(spec.get("frequency_hz", 0.2))
        t = np.arange(x.shape[-1], dtype=np.float32) / fs
        phase = rng.uniform(0.0, 2.0 * np.pi)
        x += amplitude * np.sin(2.0 * np.pi * frequency * t + phase)[None, :]
    elif kind == "lead_dropout":
        count = int(spec.get("count", 1))
        leads = rng.choice(x.shape[0], size=min(count, x.shape[0]), replace=False)
        x[leads] = 0.0
    elif kind == "limb_lead_dropout":
        x[:6] = 0.0
    elif kind == "chest_lead_dropout":
        x[6:] = 0.0
    elif kind == "powerline_50hz":
        amplitude = float(spec.get("amplitude_mv", 0.05))
        t = np.arange(x.shape[-1], dtype=np.float32) / fs
        phase = rng.uniform(0.0, 2.0 * np.pi)
        x += amplitude * np.sin(2.0 * np.pi * 50.0 * t + phase)[None, :]
    elif kind == "quantize":
        step = float(spec.get("step_mv", 0.01))
        x = np.round(x / step) * step
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return np.clip(x, -5.0, 5.0)

