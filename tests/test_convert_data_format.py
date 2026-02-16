import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from convert_data_format import (
  build_arrivals_list,
  build_base_spec_dict,
  compute_avg_demand,
  load_and_convert,
  update_arrivals_info,
  update_functions_info,
  update_nodes_info,
)


def _sample_requests():
  return {
    0: {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])},
    1: {0: np.array([5.0, 6.0]), 1: np.array([7.0, 8.0])},
  }


def _sample_instance_data():
  return {
    None: {
      "Nn": {None: 2},
      "Nf": {None: 2},
      "memory_requirement": {1: 128, 2: 256},
      "memory_capacity": {1: 1024, 2: 2048},
      "demand": {
        (1, 1): 1.0,
        (2, 1): 3.0,
        (1, 2): 2.0,
        (2, 2): 6.0,
      },
    }
  }


def test_build_base_spec_dict_defaults():
  spec = build_base_spec_dict()
  assert spec["classes"][0]["name"] == "critical"
  assert spec["nodes"] == []
  assert spec["functions"] == []
  assert spec["arrivals"] == []


def test_build_arrivals_list_layout():
  arrivals = build_arrivals_list(_sample_requests(), mt = 0, Mt = 2, ts = 1)
  assert len(arrivals) == 2
  assert len(arrivals[0]) == 4
  assert arrivals[0][0] == {"node": "n1", "function": "f1", "rate": 1.0}
  assert arrivals[1][-1] == {"node": "n2", "function": "f2", "rate": 8.0}


def test_update_arrivals_info_is_pure():
  base = build_base_spec_dict()
  updated = update_arrivals_info(base, _sample_requests(), t = 1)
  assert len(base["arrivals"]) == 0
  assert len(updated["arrivals"]) == 4
  assert updated["arrivals"][0]["rate"] == 2.0


def test_compute_avg_demand_returns_mean_and_sample_std():
  demand = _sample_instance_data()[None]["demand"]
  mean, std = compute_avg_demand(demand, Nn = 2, Nf = 2)

  assert mean[0] == pytest.approx(2.0)
  assert mean[1] == pytest.approx(4.0)
  assert std[0] == pytest.approx(np.sqrt(2.0))
  assert std[1] == pytest.approx(np.sqrt(8.0))


def test_update_functions_info_and_nodes_info():
  base = build_base_spec_dict()
  instance_data = _sample_instance_data()

  with_nodes = update_nodes_info(base, instance_data)
  assert [n["name"] for n in with_nodes["nodes"]] == ["n1", "n2", "cloud"]

  with_functions = update_functions_info(base, instance_data)
  assert [f["name"] for f in with_functions["functions"]] == ["f1", "f2"]
  assert with_functions["functions"][0]["memory"] == 128
  assert with_functions["classes"][0]["max_resp_time"] > 0


def test_load_and_convert_writes_base_spec(tmp_path: Path):
  instance_folder = tmp_path / "inst"
  dest_folder = tmp_path / "out"
  instance_folder.mkdir()

  base_payload = {
    "None": {
      "Nn": {"None": 2},
      "Nf": {"None": 2},
      "memory_requirement": {"1": 128, "2": 256},
      "memory_capacity": {"1": 1024, "2": 2048},
      "demand": {
        "(1, 1)": 1.0,
        "(2, 1)": 3.0,
        "(1, 2)": 2.0,
        "(2, 2)": 6.0,
      },
    }
  }
  (instance_folder / "base_instance_data.json").write_text(json.dumps(base_payload))
  (instance_folder / "load_limits.json").write_text(json.dumps({"0": {"0": {"min": 1, "max": 2}}}))

  spec = load_and_convert(str(instance_folder), str(dest_folder))
  assert len(spec["nodes"]) == 3
  assert (dest_folder / "base_spec.yaml").exists()

  loaded = yaml.safe_load((dest_folder / "base_spec.yaml").read_text())
  assert loaded["functions"][0]["name"] == "f1"
