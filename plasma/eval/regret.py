from __future__ import annotations

from typing import List

import numpy as np


def cumulative_regret(
    method_obj: np.ndarray, oracle_obj: np.ndarray
  ) -> np.ndarray:
  return np.cumsum(np.asarray(oracle_obj) - np.asarray(method_obj))


def adaptation_lag(
    times: np.ndarray, method_obj: np.ndarray, oracle_obj: np.ndarray,
    change_points: List[float], threshold: float = 0.1
  ) -> List[float]:
  times = np.asarray(times)
  method_obj = np.asarray(method_obj)
  oracle_obj = np.asarray(oracle_obj)
  lags: List[float] = []
  for cp in change_points:
    after = times >= cp
    recovered = after & (method_obj >= (1.0 - threshold) * oracle_obj)
    idx = np.flatnonzero(recovered)
    lags.append(float(times[idx[0]] - cp) if len(idx) else float("nan"))
  return lags
