import numpy as np
import json
import ast


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


def generate_random_float(
    rng: np.random.Generator, low: float, high: float
  ) -> float:
  val = low
  if high > low:
    val = round(rng.uniform(low, high), 3)
  return val


def load_configuration(config_file: str) -> dict:
  """Load configuration file"""
  config = {}
  with open(config_file, "r") as istream:
    config = json.load(istream)
  return config


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
