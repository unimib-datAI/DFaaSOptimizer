# Selfish centralized load management

`models.selfish.SelfishLoadManagementModel` extends `LoadManagementModel`
with one local-processing gain guarantee per node. It keeps the original
global welfare objective, flow balance, memory, utilization, neighborhood
and no-ping-pong constraints.

For each node n, first solve `LSP_detailed` with `whoami=n`, the central
instance's evaluated parameters, and free local x, y, z and replica variables.
The local objective now maximizes gain:

```text
G_n = sum_f [alpha[n,f] * x[f]
             + sum_m beta[n,m,f] * y[m,f]
             - gamma[n,f] * z[f]
             - pi[f] * sum_m y[m,f]] / D[n,f]
D[n,f] = incoming_load[n,f] if positive, otherwise 1
```

This is exactly the negative of the previous minimized cost. The local
optimal solutions are unchanged; the reported objective changes sign.
Other local model classes retain their existing conventions.

Only the x contribution of the returned local optimum defines the floor:

```text
minimum_local_gain[n] = sum_f alpha[n,f] * x_local_star[f] / D[n,f]
```

The centralized model then maximizes the original global welfare subject to:

```text
sum_f alpha[n,f] * x[n,f] / D[n,f] >= minimum_local_gain[n]   for every n
```

This is a guarantee on the weighted sum over functions, not an individual
lower bound on every x[n,f]. Local offloading and cloud rejection influence
which x is selected, but their objective terms do not enter the guarantee.
The local offloading feasibility rules remain those of `LSP_detailed`.

## Usage

Use the existing Pyomo data dictionary for a single load snapshot:

```python
from models.selfish import SelfishLoadManagementModel

model = SelfishLoadManagementModel()
instance = model.generate_instance(data)
solution = model.solve(
  instance,
  {"MIPGap": 0, "TimeLimit": 60, "OutputFlag": 0},
  solver_name="gurobi",
)
```

`solve()` computes the local floors before solving the central problem. It
requires optimal termination of every local solve, subject to the solver's
configured optimality tolerances. A failed or interrupted local optimization
does not silently become a zero guarantee. The floor parameters intentionally
have no default: bypassing this method with a raw solver requires explicitly
providing the floors first.

`pi[f]` is optional and defaults to zero, as in `LSP_detailed`; it affects the
local reference optimization only. Central coefficients and defaults are copied
to the local problems, including gamma, whose class defaults otherwise differ.
For tied local optima, the reference uses the solver's returned x; no secondary
optimization is imposed.

`solution["obj"]` is global gain. `runtime` is the sum of local and central
solver times; `local_reference_runtime` and `global_runtime` expose the two
parts. Construction and other Python overhead are excluded from these times.
The concrete instance exposes `minimum_local_gain[n]` and
`local_processing_gain[n]` for checking each guarantee.

For direct Pyomo solver control, call
`model.compute_local_gain_floors(instance, options, "gurobi")` first and then
solve the concrete instance with `pyo.SolverFactory`. This exposes solver
bounds and time-limited incumbents. The standard `solve()` interface inherits
the existing `BaseAbstractModel` handling, which can discard such incumbents
after Pyomo auto-loads them; the comparison script uses direct solver control.

## Comparison on saved instances

See the [comparison report](../outputs/selfish-comparison/REPORT.md) for results.

The reproducible experiment and results are in `outputs/selfish-comparison/`:

```bash
PYTHONPATH=. .venv/bin/python outputs/selfish-comparison/compare.py
```

It compares all four saved instances under `test_instances/` at t=0, 50, 75,
using identical inputs and Gurobi options for both central models. It writes
global results to `summary.csv`, per-node floors and gains to `nodes.csv`, and
solver settings to `settings.json`. Every solution is checked with the existing
physical-feasibility validator; every selfish solution must satisfy all floors.
The global solves use a 30-second limit and relative MIP gap 1e-5. Time-limited
incumbents are explicitly labeled and validated; they are not reported as
proven optimal solutions. `--resume` reuses completed pairs with the same settings.
