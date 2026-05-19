from hierarchical_auction.runner import build_auction_options


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


def test_run_py_accepts_hierarchical_method(monkeypatch):
  monkeypatch.setattr(
    "sys.argv",
    ["run.py", "-c", "config_files/config.json", "--methods", "hierarchical"],
  )
  from run import parse_arguments
  args = parse_arguments()
  assert args.methods == ["hierarchical"]
