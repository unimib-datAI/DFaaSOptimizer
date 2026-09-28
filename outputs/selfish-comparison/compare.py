"""Run from the repo root: PYTHONPATH=. .venv/bin/python outputs/selfish-comparison/compare.py"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import gurobipy
import numpy as np
import pyomo.environ as pyo

from generators.generate_data import update_data
from models.model import LoadManagementModel
from models.selfish import SelfishLoadManagementModel
from utils.centralized import get_current_load, validate_centralized_solution
from utils.common import load_base_instance, load_requests_traces


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, nargs="+", default=[0, 50, 75])
  parser.add_argument("--time-limit", type=float, default=30)
  parser.add_argument("--resume", action="store_true", help="Reuse completed pairs with the same settings")
  args = parser.parse_args()
  output = Path(__file__).parent
  options = {"OutputFlag": 0, "MIPGap": 1e-5, "TimeLimit": args.time_limit,
             "Threads": 1, "Seed": 0, "FeasibilityTol": 1e-8}
  rows, nodes = [], []
  if args.resume:
    rows = list(csv.DictReader((output / "summary.csv").open()))
    nodes = list(csv.DictReader((output / "nodes.csv").open()))
  completed = {(row["instance"], int(row["step"])) for row in rows}
  for folder in sorted(Path("test_instances").glob("*/base_instance_data.json")):
    folder = folder.parent
    base, _ = load_base_instance(str(folder))
    traces = load_requests_traces(str(folder))[0]
    nn, nf = base[None]["Nn"][None], base[None]["Nf"][None]
    for step in args.steps:
      if (folder.name, step) in completed:
        continue
      load = get_current_load(traces, list(range(nn)), step)
      data = update_data(base, {"incoming_load": load})
      solved = []
      for cls in (LoadManagementModel, SelfishLoadManagementModel):
        model = cls()
        instance = model.generate_instance(data)
        start = perf_counter()
        local_runtime = 0
        if cls is SelfishLoadManagementModel:
          local_runtime = model.compute_local_gain_floors(instance, options, "gurobi")
        # Direct solve retains the bounds and time-limited incumbents for both
        # formulations; BaseAbstractModel currently discards those incumbents.
        solve_start = perf_counter()
        result = pyo.SolverFactory("gurobi").solve(instance, options=options)
        solution = {
          key: [pyo.value(var) for var in getattr(instance, key).values()]
          for key in ("x", "y", "z", "r")
        }
        solution.update({
          "obj": pyo.value(instance.OBJ),
          "runtime": perf_counter() - solve_start + local_runtime,
          "local_reference_runtime": local_runtime,
          "termination_condition": str(result.solver.termination_condition),
          "solution_exists": True,
        })
        upper_bound = result.problem.upper_bound
        wall = perf_counter() - start
        assert solution["solution_exists"], solution
        arrays = {key: np.asarray(solution[key]).reshape(shape) for key, shape in (
          ("x", (nn, nf)), ("y", (nn, nn, nf)), ("z", (nn, nf)), ("r", (nn, nf))
        )}
        validate_centralized_solution(**arrays, data=data, tolerance=1e-5)
        gains = {n: pyo.value(sum(
          instance.alpha[n, f] * instance.x[n, f] / (instance.incoming_load[n, f] or 1)
          for f in instance.F
        )) for n in instance.N}
        row = {"instance": folder.name, "step": step, "nodes": nn, "functions": nf,
               "model": model.name, "objective": solution["obj"],
               "global_upper_bound": upper_bound,
               "termination": solution["termination_condition"], "wall_seconds": wall,
               "solver_seconds": solution["runtime"],
               "local_reference_seconds": solution.get("local_reference_runtime", 0),
               "local_requests": arrays["x"].sum(), "offloaded_requests": arrays["y"].sum(),
               "cloud_requests": arrays["z"].sum()}
        solved.append((row, gains, instance))
      floors = {n: pyo.value(solved[1][2].minimum_local_gain[n]) for n in solved[1][2].N}
      for row, gains, _ in solved:
        row["violating_nodes"] = sum(gains[n] < floors[n] - 1e-6 for n in floors)
        row["min_gain_margin"] = min(gains[n] - floors[n] for n in floors)
        rows.append(row)
        nodes.extend({"instance": folder.name, "step": step, "model": row["model"],
                      "node": n, "gain_floor": floors[n], "gain_x": gains[n],
                      "margin": gains[n] - floors[n]} for n in floors)
      assert solved[1][0]["violating_nodes"] == 0
      if solved[0][0]["termination"] == "optimal":
        assert solved[1][0]["objective"] <= solved[0][0]["objective"] + (
          abs(solved[0][0]["objective"]) * options["MIPGap"] + 1e-5
        )
      for filename, records in (("summary.csv", rows), ("nodes.csv", nodes)):
        with (output / filename).open("w") as stream:
          writer = csv.DictWriter(stream, fieldnames=list(records[0]))
          writer.writeheader()
          writer.writerows(records)
      print(folder.name, step, [(r["model"], round(r["objective"], 6),
                                r["termination"], r["violating_nodes"])
                               for r, _, _ in solved], flush=True)
  (output / "settings.json").write_text(json.dumps({
    "gurobi_version": gurobipy.gurobi.version(), "solver": "gurobi",
    "options": options, "steps": args.steps, "pi": 0,
  }, indent=2) + "\n")


if __name__ == "__main__":
  main()
