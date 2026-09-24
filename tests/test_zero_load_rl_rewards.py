"""Exercise real RL reward arithmetic without installing or starting RL stacks."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def environments(monkeypatch):
  # Only unavailable framework imports are replaced. monkeypatch restores any
  # prior modules after each test, including the real source modules loaded here.
  exports = {
    "RL4CC.environment": {
      "BaseEnvironment": object, "BaseMultiAgentEnvironment": object,
    },
    "RL4CC.callbacks": {"BaseCallbacks": object},
    "ray.rllib.utils.spaces.simplex": {"Simplex": object},
    "ray.rllib.env.env_context": {"EnvContext": dict},
    "gymnasium.spaces": {"Box": object, "Dict": object, "Discrete": object},
  }
  for name, attributes in exports.items():
    parts = name.split(".")
    for length in range(1, len(parts) + 1):
      module_name = ".".join(parts[:length])
      module = ModuleType(module_name)
      module.__path__ = []
      if length == len(parts):
        module.__dict__.update(attributes)
      monkeypatch.setitem(sys.modules, module_name, module)

  root = Path(__file__).resolve().parents[1] / "rlagents"
  package = ModuleType("rlagents")
  package.__path__ = [str(root)]
  monkeypatch.setitem(sys.modules, "rlagents", package)
  for filename in ("sa_environment", "ma_environment"):
    name = f"rlagents.{filename}"
    spec = importlib.util.spec_from_file_location(name, root / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
  return module.FaaSMARLEnvironment, module.FaaSMARLEnvironment2


def _receiver(loads, per_node):
  """Source serves half locally, forwards a quarter, and rejects a quarter."""
  functions = list(range(1, len(loads) + 1))
  data = {
    "Nn": {None: 2}, "Nf": {None: len(loads)},
    "alpha": {(n, f): 2.0 for n in (1, 2) for f in functions},
    "beta": {(n, j, f): 3.0 for n in (1, 2) for j in (1, 2) for f in functions},
    "gamma": {(n, f): 4.0 for n in (1, 2) for f in functions},
    "max_utilization": {f: 1.0 for f in functions},
    "memory_capacity": {1: len(loads), 2: len(loads)},
    "memory_requirement": {f: 1.0 for f in functions},
  }
  info = {"__common__": {}}
  for n in (1, 2):
    rates = loads if n == 1 else [0.0] * len(loads)
    metrics = {
      "input_rate": rates,
      "loc": [v / 2 for v in rates],
      "total_fwd": [v / 4 for v in rates],
      "rej": [v / 4 for v in rates],
      "n_replicas": [1] * len(loads),
      "cpu_utilization": [v / (2 if n == 1 else 4) for v in loads],
    }
    if per_node:
      metrics["fwd"] = [metrics["total_fwd"]]
      metrics[f"fwd_to_{3 - n}"] = metrics["total_fwd"]
      info[str(n)] = metrics
    else:
      for f in functions:
        scalar = {key: values[f - 1] for key, values in metrics.items()}
        scalar["fwd"] = [scalar["total_fwd"]]
        scalar[f"fwd_to_{3 - n}_{f}"] = scalar["total_fwd"]
        info[f"{n}_{f}"] = scalar
  neighbors = {"1": ["2"], "2": ["1"]} if per_node else {1: [2], 2: [1]}
  return SimpleNamespace(
    instance_data=data, nodes=[1, 2], functions=functions,
    agents=[key for key in info if key != "__common__"],
    agent_neighbors=neighbors, info=info,
  )


@pytest.mark.parametrize("per_node", [False, True], ids=["node-function", "node"])
@pytest.mark.parametrize("loads,expected", [
  ([0.0], 0.0), ([0.0, 0.5], 2.75), ([0.5], 2.75),
], ids=["zero", "mixed-functions", "positive-fraction"])
def test_rewards_are_zero_safe_and_keep_positive_normalization(environments, per_node, loads, expected):
  receiver = _receiver(loads, per_node)
  environment = environments[int(per_node)]
  with np.errstate(divide="raise", invalid="raise"):
    rewards = environment.compute_reward(receiver)

  assert receiver.info["__common__"]["feasible"]
  # Existing centralized objective subtracts cloud cost, while these MA reward
  # methods add it. Preserve both conventions; this change only handles zeros.
  assert receiver.info["__common__"]["cobj"] == pytest.approx(0.75 if expected else 0)
  assert sum(rewards.values()) == pytest.approx(expected)
  assert all(np.isfinite(value) for value in rewards.values())
  for agent in receiver.agents:
    if agent.startswith("2") or (agent == "1_1" and loads[0] == 0):
      assert rewards[agent] == 0
      assert receiver.info[agent]["loc_utility"] == 0
      assert receiver.info[agent]["fwd_utility"] == 0
      assert receiver.info[agent]["cloud_penalty"] == 0
  if expected:
    source = "1" if per_node else f"1_{len(loads)}"
    assert receiver.info[source]["loc_utility"] == pytest.approx(1.0)
    assert receiver.info[source]["fwd_utility"] == pytest.approx(0.75)
    assert receiver.info[source]["cloud_penalty"] == pytest.approx(-1.0)
