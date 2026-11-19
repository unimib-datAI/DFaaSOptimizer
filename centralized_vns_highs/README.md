# VNS-HiGHS

VNS-HiGHS maximizes welfare for a **fixed local plan x**. A native HiGHS solve
runs on a CPU thread while one VNS-R pool searches on CPU, Mac Metal or Spark
CUDA. Both start with the same feasible allocation. All edges, objectives and
feasibility checks use the existing VNS problem representation.

When HiGHS improves the global record, the pool takes the latest checked record
at its next checkpoint and reseeds the trajectory with the highest stall counter.
The other trajectories keep searching. A VNS improvement goes back to the same
HiGHS tree on its next user-solution callback. The initial warm start and later
submissions are counted separately. A submission counts a vector passed to
HiGHS. The solver decides whether to use it. The exchange is asynchronous, so
repeating a seed can give a different sequence of incumbents.

## Run

From the optimizer repository, use the project environment with highspy 1.15.1
or newer and the chosen VNS backend. For example, on Spark:

```bash
python -m centralized_vns_highs /path/to/instance --output /path/to/new-run \
  --backend cuda --init-device cuda --pool-size 256 --candidates 512 \
  --time-limit 300 --relative-gap 0.0001
```

Use `--backend metal` on the Mac or `--backend cpu` for a CPU-only hybrid.
The runner computes the local DP once, or accepts `--solution /path/to/solution.npz`
to take both x and starting flows from a prior allocation. `--step` selects the
trace step. Spark uses the resident VNS-R CUDA engine. `--batch-size` sets the
number of CUDA iterations between exchanges (default 8). CPU and Metal check
after each iteration. `--iterations 0` runs HiGHS alone with the same MIP, warm
start and budget. A positive iteration cap applies only to VNS.

## Bounds and stopping

The integer model chooses edge flows, receiver replica counts and binary sender
roles. Its constraints enforce residual load, sender/receiver exclusivity,
processing capacity, a replica for positive processing, and shared node RAM.
With x fixed, welfare equals zero-flow welfare plus the sum of flow rewards.
HiGHS minimizes the negative flow reward. Subtracting its dual bound from
zero-flow welfare gives a global upper bound on fixed-x welfare.

The initial upper bound lets each residual request use its best outgoing edge
with unlimited receiving capacity. HiGHS tightens it during branch-and-cut.
Each new incumbent passes the original feasibility checker. Both searches stop
when the wall budget expires or `(upper - welfare) / max(1, abs(welfare))`
reaches the requested gap. Limits include solver and pool setup, after instance
loading and local initialization. An in-flight VNS step or CUDA batch can overrun
the deadline. Bounds are numerical and subject to solver tolerances.

A fresh output directory holds the checked `solution.npz` and `result.json`
with welfare, bound, gap (as a fraction), timed bound and incumbent history,
solver status, nodes, exchange counts, and options. The allocation can be passed
to standalone VNS-RX for a separate variable-x result. The reported bound
applies only to the saved x.

Run the focused tests with `python -m pytest -q centralized_vns_highs/tests`.
The CUDA reseeding test can also run with `NUMBA_ENABLE_CUDASIM=1` when Numba
is installed. Hardware throughput requires a real device.
