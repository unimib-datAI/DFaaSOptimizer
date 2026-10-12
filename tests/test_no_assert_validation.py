"""Feasibility and invariant checks of the solvers must survive `python -O` (no `assert`)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = [
  "decentralized_bestresponse.py", "decentralized_diffusion.py", "decentralized_dual.py",
  "decentralized_frankwolfe.py", "decentralized_gcaa.py", "decentralized_potentialgame.py",
  "decentralized_powerd.py", "heuristic_coordinator.py", "hierarchical_auction/engine.py",
  "hierarchical_auction/iterative_engine.py", "hierarchical_frankwolfe.py",
  "plasma/baselines/milp_baseline.py", "plasma/runner.py", "run_faasmacro.py", "run_faasmadea.py",
]


def test_solver_modules_validate_without_assert():
  found = [f"{name}:{node.lineno}" for name in CHECKED
           for node in ast.walk(ast.parse((ROOT / name).read_text())) if isinstance(node, ast.Assert)]
  assert found == []
