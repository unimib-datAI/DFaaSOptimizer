from typing import Tuple
import numpy as np
import json
import ast
import os


class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NpEncoder, self).default(obj)


def delete_tuples(_dict: dict):
  """Delete tuple keys recursively from all of the dictionaries"""
  serializable_dict = {}
  for key, value in _dict.items():
    new_key = key if not isinstance(key, tuple) is not None else str(key)
    if isinstance(value, dict):
      serializable_dict[new_key] = delete_tuples(value)
    else:
      serializable_dict[new_key] = value
  return serializable_dict


def float_to_int(fval: float) -> int:
  ival = None
  if np.floor(fval) > 0 and ((fval / np.floor(fval) - 1) > 1e-6):
    ival = int(np.ceil(fval))
  else:
    if int(np.floor(fval)) == 0 and (fval > 1e-6):
      ival = int(np.ceil(fval))
    else:
      ival = int(fval)
  return ival


def generate_random_float(
    rng: np.random.Generator, low: float, high: float
  ) -> float:
  val = low
  if high > low:
    val = round(rng.uniform(low, high), 3)
  return val


def generate_random_int(
    rng: np.random.Generator, limits: dict
  ) -> int:
  val = 0
  if "min" in limits and "max" in limits:
    val = int(rng.integers(limits["min"], limits["max"], endpoint = True))
  elif "values_from" in limits:
    val = rng.choice(limits["values_from"])
  else:
    raise ValueError("Missing values to define limits")
  return val


def int_keys_decoder(pairs: dict) -> dict:
  return {int(k): v for k, v in pairs}


def load_configuration(config_file: str) -> dict:
  """Load configuration file"""
  config = {}
  with open(config_file, "r") as istream:
    config = json.load(istream)
  return config


def load_requests_traces(folder: str) -> Tuple[dict, int, int, int]:
  # load requests
  requests = {}
  with open(
      os.path.join(folder, "input_requests_traces.json"), "r"
    ) as ist:
    requests = {
      int(f): {
        int(a): np.array(r) for a,r in v.items()
      } for f,v in json.load(ist).items()
    }
  # load time info
  mt = 0
  Mt = len(requests[0][0])
  ts = 1
  if os.path.exists(os.path.join(folder, "config.json")):
    config = load_configuration(os.path.join(folder, "config.json"))
    mt = config.get("min_run_time", 0)
    Mt = config.get("max_run_time", config["max_steps"])
    ts = config.get("run_time_step", 1)
  return requests, mt, Mt, ts


def restore_types(serialized_dict: dict):
  """Restore the original types"""
  _dict = {}
  for key, value in serialized_dict.items():
    new_key = ast.literal_eval(key)
    if isinstance(value, dict):
      _dict[new_key] = restore_types(value)
    else:
      _dict[new_key] = value
  return _dict
