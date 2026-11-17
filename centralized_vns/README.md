# Centralized fixed-x population VNS

Phase 1 implementation: a CPU reference and optional resident CUDA search.
All new algorithm code, tests, example and documentation are in this directory.
Existing LSP/DP modules are imported; no existing method or campaign is changed.

## Run on Titan or a Mac (CPU)

From the DFaaSOptimizer repository root, using Python 3.10:

```bash
python3.10 -m venv .venv-vns
.venv-vns/bin/python -m pip install -r centralized_vns/requirements-cpu.txt
.venv-vns/bin/python -m pytest centralized_vns/tests -q
.venv-vns/bin/python -m centralized_vns centralized_vns/examples/tiny \
  --output /tmp/vns-tiny --pool-size 4 --candidates 32 --iterations 50
```

The existing solver `.venv/bin/python` can also run these commands. The minimal
requirements do not install CUDA, RL, Gurobi or the dispatcher. CPU imports do
not load Numba. An Apple GPU is not a CUDA device; use CPU on the Mac.
The example is a handcrafted regression fixture, not a benchmark result.
Choose a new output path for each run: existing directories are never overwritten.

For a real instance, replace the example path with a materialized directory
containing base_instance_data.json, input_requests_traces.json, load_limits.json,
graph.json and their metadata.json checksums. Inputs are read and hashed, never
modified. `--step 0` selects the first trace interval.

## Algorithm

1. Compute fixed local x with the existing exact `LSP_detailed` DP. Initialization
   may run on CPU or CUDA (`--init-device cuda`), independently of search backend.
   The CUDA initializer falls back to CPU DP for unsupported individual problems,
   reporting counts; if CPU DP also declines, fail explicitly. No hidden external
   solver fallback and no change to x is allowed.
2. Start each trajectory with no offloading and all residual load rejected. Its
   initial insertion neighborhood constructs a feasible allocation. Seeded
   candidate streams provide different constructions across the pool.
3. Search three sampled neighborhoods: insertion (unit or maximal feasible block),
   same-source/function rerouting (unit or whole flow), and atomic two-flow
   destination exchange (unit or whole blocks, possibly different functions).
   Exchange can release memory for a different function in the same transition.
4. Apply the best improving sampled move per trajectory. Return to insertion on
   improvement; otherwise advance the neighborhood. This is sampled VND, not
   exhaustive local optimality certification.
5. After stagnation, choose a seeded feasible destruction rather than ranking
   destruction by gain. Remove up to 1–4 whole flows, increasing the requested
   count every nine iterations and cycling. Repeated sampled edges are skipped,
   so actual destruction can be smaller. Then rebuild with local search.
6. Maintain an independent best archive per trajectory and return the best of
   those archives. Perturbations can lower current welfare, never the archive.
   This first version has independent multistart trajectories, not elite migration
   between trajectories; population replacement is a future experiment.

State uses only permitted directed `(source, destination, function)` edges.
Self edges are excluded. x is shared and immutable. y, rejected load, and received
load are trajectory-specific. Replicas are derived from `(x + received)/capacity`.
Source conservation, nonnegative integral flows, per-node memory and the prohibition
on simultaneously sending/receiving a function are checked for the complete move.
There is one application thread per trajectory, so moves cannot race for resources.

Weights use the same source-load-normalized social welfare as
`utils.faasmacro.compute_centralized_objective`; no best-of-pool objective change.
The all-rejected residual state's welfare is `initial_welfare` in outputs, not the
optimistic sum of LSP objectives. x's contribution remains constant during search.

### Numerical policy

Replicas use `max(1, ceil(processing/capacity - 1e-9))` for positive processing,
zero otherwise. The `1e-9` ratio tolerance is consistent with the existing DP's
replica boundary convention and avoids rejecting its fixed x due to roundoff.
Generation, move validation and full validation use the same convention. This
is a numerical tolerance, not a relaxation intended to create extra capacity.
Improvement acceptance uses absolute welfare tolerance `1e-12`.

Individual integer load/x/memory/RAM are bounded by 2^40; aggregate load is bounded
by 2^52. Unsupported replica requirements are rejected. Memory accumulation uses
floating products to avoid int64 wraparound; any feasible integer sum is at most
RAM (2^40) and exactly representable. The full checker independently reconstructs
flows, replicas, conservation, roles and welfare from the final sparse allocation.

## CUDA, when a server is available

Use the repository's optional GPU versions (`numba==0.68.0`,
`numba-cuda==0.30.4`) and a compatible NVIDIA driver/runtime. No automatic backend
switch occurs when requesting CUDA. In the solver environment:

```bash
python -m centralized_vns /path/to/materialized-instance \
  --output /path/to/new-output --backend cuda --init-device cuda \
  --pool-size 64 --candidates 256 --iterations 1000 --time-limit 300
```

Population, caches, candidate buffers and archives remain on device. Candidate
creation and feasibility/gain evaluation are parallel over population × candidates.
A separate kernel selects and applies one move per trajectory. Selection is serial
within a trajectory in phase 1; benchmark before introducing a parallel reduction.
Each iteration copies only one incumbent scalar per trajectory to the host; the
final winning allocation is copied once. No full candidate-solution copies exist.

`--max-device-mib 512` caps the estimated persistent array allocation before
allocating the host/device population. This excludes driver/JIT overhead. Device
allocation failure propagates; real GPU performance and peak usage remain untested.

Simulator tests on Linux without a GPU (temporary environment recommended):

```bash
python -m pip install numba==0.68.0 numba-cuda==0.30.4 nvidia-cuda-runtime-cu12==12.9.79
NUMBA_ENABLE_CUDASIM=1 python -m pytest centralized_vns/tests/test_gpu.py -q
```

The simulator requires the CUDA runtime library with these package versions but
no driver. This command is for Linux, not a required Mac setup. Simulator equality
does not prove device compilation, numerical agreement on hardware, or speedup.
CPU-only environments skip GPU tests when Numba is absent.

## Budgets and reproducibility

`--time-limit` is a **search wall-time** budget including engine construction/JIT;
it is checked between iterations, so one iteration may overrun it. Input loading
and LSP initialization are recorded separately and in total wall time. It is not
the distributed modelled-time clock used in older campaign comparisons.

For reproducible CPU/CUDA trajectory tests use an iteration limit and a generous
wall budget. A binding wall limit depends on machine speed. Candidate generation
uses integer deterministic streams keyed by seed, iteration, trajectory and slot;
tie breaking is candidate order. Real-device floating-point differences still
need validation. Pool members are part of one multistart run, not independent
statistical repetitions of the final algorithm.

`solution.npz`: fixed x, sorted sparse edges, y, z, replicas (zero-based edge indices).
`result.json`: full options, input hashes, actual solver commit/dirty flag, platform,
initialization device/counts/time, welfare trace, candidate count and search/total
wall time. It marks simulator execution explicitly. `feasible: true` is emitted
only after the independent final checker passes.

## Next phase

1. On Mac: run the minimal CPU setup/tests and example; extend CPU neighborhoods
   or explore quality without changing experiment inputs.
2. On an idle NVIDIA host: compare full CPU and CUDA allocations, including boundary,
   zero-load and compound exchange cases; test the LSP CUDA initializer separately.
3. Check device compilation and peak allocations, then benchmark pools 32/64/128 and
   candidate batch sizes on representative 30/160/500-node instances. Separate
   compilation warm-up from steady-state timing while retaining total wall time.
4. Compare quality against existing methods at explicit, comparable budgets and
   multiple algorithm seeds. Tiny exact enumeration tests are correctness checks,
   not proof of optimality on benchmark instances.
5. Only then consider elite migration, stronger removal/reinsertion neighborhoods,
   dispatcher integration or a full campaign. No .35/.175 jobs run in phase 1.
