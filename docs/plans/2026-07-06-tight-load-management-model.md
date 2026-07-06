# TightLoadManagementModel: centralized model reformulation and solver findings

## Context

Investigation into speeding up the centralized `LoadManagementModel`
(`models/model.py`). Benders decomposition was considered and rejected;
benchmarking found the real bottleneck elsewhere.

## Key finding: MIPGap, not the model

Benchmarks (Gurobi 12.0.3, Apple M3 Pro, random instances with the
`eval_full.json` parameters, Nf=2, sinusoidal load traces):

| Nn  | MIPGap 1e-5 (configs today) | MIPGap 1e-2 | speedup |
|-----|-----------------------------|-------------|---------|
| 20  | 0.16 s/step                 | 0.035 s/step| ~5x     |
| 50  | 45.8 s/step, 1 of 3 steps hit TimeLimit=120 and returned **no solution** | 0.17 s/step | ~270x |
| 100 | 83.5 s/step                 | 0.65 s/step | ~130x   |

Objective degradation at gap 1e-2 was 0.07–0.2% on the same instances.
The tight gap is also less robust: the step that timed out at 1e-5 solved
in 0.2 s at 1e-2.

**Recommendation: set `"MIPGap": 1e-2` (or `1e-3`) in
`config_files/*.json` and `test_instances/config.json`.** Pyomo instance
build time is negligible (0.08 s at Nn=50, 0.3 s at Nn=100), so a
persistent solver interface is not needed. At these sizes (Nn <= 100
solving sub-second with a sane gap), Benders or Lagrangian decomposition
would add per-iteration overhead for nothing; revisit only well beyond
Nn ~ 500.

## TightLoadManagementModel

New class in `models/model.py`, subclassing `LoadManagementModel` (the
original is untouched). Two changes:

1. **Tighter big-M in `no_ping_pong2`**: the bound
   `sum_m incoming_load[m,f]` is restricted to actual neighbors
   (`incoming_load[m,f] * neighborhood[m,n]`). Tighter LP relaxation,
   same feasible set, since `offload_only_to_neighbors` already forces
   `y[m,n,f] = 0` for non-neighbors.
2. **`utilization_equilibrium2` dropped**, replaced by a `-r_penalty *
   sum(r)` term in the objective (mutable Param, default 1e-4). The
   original constraint only pins `r` to its minimal value; the epsilon
   penalty achieves the same without |N|*|F| constraints coupling `r` to
   the flow variables. `r_penalty` must stay well below the smallest
   marginal gain of serving one request (~beta/incoming_load); at the
   default it never trades replicas against served traffic.

Selection: `"model_variant": "tight"` in the run config
(`run_centralized_model.py`); default behavior is unchanged.

## Measured effect

Same Nn=50 instance, 3 timesteps:

| model    | gap 1e-5    | gap 1e-2     |
|----------|-------------|--------------|
| original | 46.5 s/step | 0.160 s/step |
| tight    | 47.2 s/step | 0.132 s/step |

The reformulation is equivalence-verified and ~18% faster at a
reasonable gap, but it does **not** fix the 1e-5 pathology — that time
goes into proving the bound, not into branching that tighter big-Ms
help. The MIPGap change is the fix; the tight model is a bonus on top.

## Verification

`tests/test_tight_model.py`: on 2 random seeds, checks that the tight
model's objective (net of the epsilon penalty) matches the original
within 1e-4 relative tolerance and that total replicas stay minimal.

Note for future stress testing: the Nn=50 / seed=42 / step t=1 instance
makes both models hit TimeLimit=120 at gap 1e-5 — a good hard case to
keep around.
