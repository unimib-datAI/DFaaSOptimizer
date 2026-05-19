import numpy as np

from hierarchical_auction.runner import (
  build_auction_options,
  compute_offloaded_demand,
)


def test_build_auction_options_accepts_scalar_or_list_eta():
  config = {
    "solver_options": {
      "auction": {
        "epsilon": 0.01,
        "eta": [0.5, 0.3],
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      }
    }
  }
  opts = build_auction_options(config)
  assert opts["eta"] == [0.5, 0.3]


def test_build_auction_options_scalar_eta():
  config = {
    "solver_options": {
      "auction": {
        "epsilon": 0.01,
        "eta": 0.5,
        "zeta": 0.1,
        "latency_weight": 0.0,
        "fairness_weight": 0.0,
      }
    }
  }
  opts = build_auction_options(config)
  assert opts["eta"] == 0.5


def test_compute_offloaded_demand_sums_seller_axis():
  y = np.zeros((2, 3, 2))
  y[0, 1, 0] = 3.0
  y[0, 2, 0] = 4.0
  y[1, 0, 1] = 5.0

  rmp_omega = compute_offloaded_demand(y)

  assert np.array_equal(rmp_omega, np.array([
    [7.0, 0.0],
    [0.0, 5.0],
  ]))


def test_extract_graph_latency_returns_numpy_array_with_edge_weights():
  import networkx as nx
  from hierarchical_auction.runner import _extract_latency

  g = nx.path_graph(3)
  nx.set_edge_attributes(
    g, {(0, 1): 5.0, (1, 2): 3.0}, "network_latency"
  )
  lat = _extract_latency(g)

  assert isinstance(lat, np.ndarray)
  assert lat.shape == (3, 3)
  assert lat[0, 1] == 5.0
  assert lat[1, 0] == 5.0   # undirected
  assert lat[1, 2] == 3.0
  assert lat[0, 0] == 0.0


def test_run_py_accepts_hierarchical_method(monkeypatch):
  monkeypatch.setattr(
    "sys.argv",
    ["run.py", "-c", "config_files/config.json", "--methods", "hierarchical"],
  )
  from run import parse_arguments
  args = parse_arguments()
  assert args.methods == ["hierarchical"]
