"""merge_sol_dict must tolerate methods covering different timestep ranges.

Merging a baseline whose solution has more timesteps than another method (e.g. a
stale centralized folder from a run with a longer horizon) used to crash with
KeyError on the missing 't = N'. Postprocessing should skip the absent method
for that timestep instead.
"""

import pandas as pd

from run import merge_sol_dict


def _cnt(key, values, idx):
  return pd.DataFrame({key: values}, index=idx)


def test_merge_tolerates_missing_timestep():
  idx = ["f1", "f2"]
  a = {
    "tot": _cnt("tot", [3, 4], idx),
    "t = 0": _cnt("t = 0", [1, 2], idx),
    "t = 1": _cnt("t = 1", [2, 2], idx),   # baseline has an extra timestep
  }
  b = {
    "tot": _cnt("tot", [1, 1], idx),
    "t = 0": _cnt("t = 0", [1, 1], idx),
    # no "t = 1": previously KeyError inside merge_sol_dict
  }
  out = merge_sol_dict([a, b], ["A", "B"])
  assert set(out["time"]) == {"tot", 0, 1}
  # shared timestep keeps both methods (B joins without suffix, no name clash)
  assert "t = 0_A" in out.columns and "t = 0" in out.columns
  # the baseline-only timestep survives with just method A, no crash
  assert "t = 1_A" in out.columns
