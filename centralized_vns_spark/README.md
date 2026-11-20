# VNS-R resident CUDA execution for DGX Spark

This separate runner preserves `centralized_vns` and `centralized_vns_r`. The
fixed-x search uses the same VNS-R neighborhoods, integer random streams,
float64 gains, deterministic candidate ties and archive rules. It is a CUDA
implementation, not a Metal benchmark and not evidence of achieved GPU occupancy.

## What executes where

| Work | Execution |
|---|---|
| Initial local DP | CUDA by default; unsupported cases fail explicitly |
| Candidate generation and feasibility | CUDA |
| Float64 gains and best-move reduction | CUDA, fused with generation |
| Ordered compound-move application | CUDA; independent trajectories in parallel |
| Archive copies | Cooperative CUDA threads |
| All-record/global-best rolling plateau | CUDA, every iteration |
| Winner reduction | CUDA |
| Standalone x proposals, floor/physics checks, repairs, selection and application | CUDA |
| Reading, launch submission, deadline polling, final independent verification and saving | CPU |

The fixed stage fuses candidate work and acceptance into one block per trajectory.
Blocks use 64, 128 or 256 threads; candidates beyond the width are processed in
strides. The default polls every 16 iterations and transfers only a three-integer
status and the best objective per checkpoint. GPU stop masking preserves the
first plateau iteration inside a queued batch. Recorded timestamps are actual
host checkpoints, not interpolated iteration times. Wall deadlines are checked
between batches: one in-flight batch can overrun the limit. `--batch-size 1`
provides the finest deadline polling at the cost of synchronization overhead.

Inputs, states, candidate buffers and plateau history remain on device. Allocations
are bounded before creating device populations; initial states are broadcast on
GPU. Memory estimates cover program arrays and upload staging, not driver/JIT
overhead. Keep a reserve on Spark's shared system memory.

## Run

From the solver repository, use a Linux ARM64 environment with the existing CPU
requirements and NVIDIA's maintained `numba-cuda` package plus a CUDA toolkit and
driver supporting GB10 (compute capability 12.1). Follow the
[Numba CUDA installation guide](https://nvidia.github.io/numba-cuda/user/installation.html)
and the [Spark software documentation](https://docs.nvidia.com/dgx/dgx-spark/).
The Mac simulator environment is not a deployable Spark environment.

```bash
python -m centralized_vns_spark /path/to/instance \
  --output /path/to/new-result --autotune --calibration-time 2 \
  --time-limit 300 --iterations 100000 --max-device-mib 4096 \
  --patience 200 --plateau-scope all --refine-x \
  --x-iterations 1000000 --x-time-limit 300
```

`--autotune` tests pool sizes 64–1024, candidates 256–2048 and all three block
widths with two seeds. Compilation is warmed separately. Trials have equal search
time budgets with plateau disabled; real elapsed search times and any deadline
overrun are recorded. Trials exceeding the budget by more than
`max(0.05 seconds, 10% of the budget)` are retained in the report but excluded
from selection. Selection maximizes independently checked mean welfare,
then throughput. The default grid costs approximately four minutes plus setup
and compilation. `calibration.json` records trials, exclusions and hardware.
Calibration refuses CUDASIM and does not silently fall back to CPU.

Without calibration, explicit `--pool-size`, `--candidates` and `--threads` are
used (defaults 256, 512 and 128). They are starting values, not measured optima.
Use `--plateau-scope global` to stop on the pool leader; `all` requires every
individual archive to plateau. Both use `--plateau-tolerance 1e-6` by default.

Both stages retain separate `fixed_x/` and `variable_x/` artifacts. To refine an
existing VNS or selfish-distributed NPZ/native output folder independently:

```bash
python -m centralized_vns_spark /path/to/instance \
  --solution /path/to/selfish-distributed-output --solution-row 0 \
  --output /path/to/new-refinement --refine-only \
  --x-iterations 1000000 --x-time-limit 300 --max-device-mib 4096
```

The x stage protects each input node's `sum(alpha*x/max(load,1))` floor and saves
the input provenance. It is deterministic **parallel best improvement**, whereas
the earlier CPU RX uses randomized first improvement. It therefore need not
follow the same x trajectory. Its iteration budget counts indexed candidate slots
(including duplicates/infeasible slots); a candidate includes receiver repairs.
The seed is accepted for interchange compatibility but does not change this
enumeration. Scratch arrays put neighboring candidates in adjacent memory cells.
The batch target is eight 64-thread blocks per GPU multiprocessor (at least
4096 candidate slots), capped by available work and the configured memory budget.
Fixed-x algorithm decisions retain their reference ordering.

## Validation and limits

Run real-device correctness tests before a Spark campaign:

```bash
python -m pytest centralized_vns_spark/tests -q
```

For local logical validation only, explicitly enable `NUMBA_ENABLE_CUDASIM=1`.
Simulator results do not establish real CUDA compilation, speed, occupancy or
bitwise agreement on hardware. Float64 is retained and decision-sensitive gain
products use explicitly rounded CUDA operations to avoid multiply-add contraction.
The x-stage reductions mirror NumPy's pairwise summation and runtime buffer size,
so large and small contributions do not falsely violate unchanged-node floors.
Real-device boundary/near-tie tests remain required. Local results and their
limits are recorded in [validation/README.md](validation/README.md).

No separate phase profiler is included. Maximize feasible welfare per experiment
budget; the advertised FP4 AI rate does not predict this integer/float64 search.
See [NVIDIA's hardware description](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
