import numpy as np

import decentralized_frankwolfe as dfw
import hierarchical_frankwolfe as hfw


def test_variants_select_hierarchy_and_refinement():
  assert hfw.run_hierarchical.keywords == {"hierarchy": True, "refine_welfare": False}
  assert hfw.run_hierarchical_pg.keywords == {"hierarchy": True, "refine_welfare": True}
  assert hfw.run_flat_pg.keywords == {"hierarchy": False, "refine_welfare": True}
  assert hfw.RESULT_NAMES[(True, True)] == "Hierarchical-MAFW-PG"


def test_relaxed_flows_expose_receiver_prices():
  Nn, Nf = 3, 1
  data = {None: {
    "Nn": {None: Nn}, "Nf": {None: Nf},
    "beta": {(i + 1, j + 1, 1): 1.0 for i in range(Nn) for j in range(Nn)},
    "gamma": {(i + 1, 1): 0.05 for i in range(Nn)},
    "demand": {(i + 1, 1): 1.0 for i in range(Nn)},
    "memory_capacity": {i + 1: 4 for i in range(Nn)},
    "memory_requirement": {1: 2}, "max_utilization": {1: 1.0},
  }}
  params = dfw.fw_parameters(data)
  omega = np.array([[8.0], [0.0], [0.0]])
  options = dfw.frankwolfe_options({"frankwolfe": {"kappa": 10.0}})
  gains, mask = dfw.fw_gains_and_mask(
    params, omega, np.ones((Nn, Nn)) - np.eye(Nn), omega > 0,
    np.zeros((Nn, Nn)), np.zeros((Nn, Nf)), options,
  )
  q, stats = dfw.relaxed_flows(
    params, omega, np.zeros((Nn, Nf)), np.zeros((Nn, Nf)), np.full(Nn, 4.0),
    gains, mask, options,
  )
  price = stats["price"]
  assert price.shape == (Nn, Nf) and (price >= 0).all()
  # both receivers are saturated (2 req/s of free memory each, 8 offered)
  assert price[1, 0] > 0 and price[2, 0] > 0
