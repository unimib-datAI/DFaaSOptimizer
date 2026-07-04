from __future__ import annotations

import numpy as np

from plasma.core.types import LOCAL, REJ, PlasmaOptions


def target_weights(
    D_f: np.ndarray, local_open: bool, nbr_spare: np.ndarray,
    eps_explore: float
  ) -> np.ndarray:
  w = D_f.copy()
  if not local_open:
    w[LOCAL] = 0.0
  gates = np.where(nbr_spare > 0.0, 1.0, eps_explore)
  w[2:] = w[2:] * gates
  return w


def choose_target(
    rng: np.random.Generator, weights: np.ndarray, unsplittable: bool
  ) -> int:
  if unsplittable:
    return int(np.argmax(weights))
  total = weights.sum()
  if total <= 0.0:
    return REJ
  return int(rng.choice(len(weights), p=weights / total))


def update_conductance(
    D: np.ndarray, phi: np.ndarray, rewards: np.ndarray, opts: PlasmaOptions
  ) -> np.ndarray:
  reinforced = (1.0 - opts.mu) * D + opts.mu * (phi ** opts.kappa) * rewards
  return np.clip(reinforced, opts.D_min, opts.D_max)
