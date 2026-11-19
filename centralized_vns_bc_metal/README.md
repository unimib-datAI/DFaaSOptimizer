# HiGHS with Metal primal search

The experiment compares one HiGHS branch-and-cut solver, the same solver with the existing Metal VNS-R population, and the solver with a modified Metal neighborhood. All arms use the same fixed local allocation, initial integer flows, root pair cuts and RAM formulation. HiGHS receives a full feasible warm start including continuous RAM path variables. Subsequent VNS records use the same encoding. Model construction checks the common LP upper bound with the dyadic certificate before running B&C. Each arm starts a fresh solver.

Run from the optimizer directory on an Apple Silicon Mac with the existing MLX environment:

```sh
.venv/bin/python -m centralized_vns_bc_metal INSTANCE ARCHIVED_TREE_RESULT NEW_OUTPUT \
  highs vns coordinated --seconds 300
```

The budget covers warm-start submission, GPU population initialization, integer search and exchanges. Data load and common model construction are outside this budget. Stop when the gap on the coordinator objective reaches 0.001 or the search time expires. HiGHS uses one thread and seed 4850. The two Metal arms use 64 trajectories and 512 candidate slots per trajectory. Metal generates candidates and tests feasibility, with float64 gain calculation and move selection on the host. When a better record becomes available, one stale trajectory restarts there and the rest keep exploring. Submitted solutions are recorded only when a HiGHS user-solution callback receives them.

In the coordinated arm, half of level-2 candidate slots propose withdrawing flow on one or two incoming arcs and admitting a different function at the same receiver. Each such proposal must admit more traffic than the original free RAM would permit. The existing feasibility checker scores the complete move. The remaining slots use VNS-R neighborhoods. The report includes accepted moves of the new type, how many used two donors, the source of each incumbent, and the global upper-bound trace. Each arm writes its solution and a JSON report to its own output directory.
