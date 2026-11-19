# VNS-Frontier-PR

Fixed-x forwarding search over an exhaustive frontier of disjoint integer regions.
The original VNS–HiGHS package remains a separate entry point.

From the optimizer directory, run on Mac with the existing MLX environment:

```bash
python -m centralized_vns_frontier /path/to/instance --step 0 --backend metal \
  --branches 256 --pool-size 256 --candidates 512 --time-limit 300 \
  --global-share 0.80 --relative-gap 0.001 --final-relink-top 32 --output /path/to/fresh-output
```

For Spark use `--backend cuda --init-device cuda` in an environment containing
Numba CUDA and highspy (tested solver version 1.15.1). Use `--backend cpu` for
the reference implementation. Supply `--solution solution.npz` to reuse both
the fixed local plan and initial flows. The default computes the local DP and
starts with zero forwarding.

## Search and proof

HiGHS solves the continuous relaxation of the original model. Fractional flow
columns with disjoint endpoints define splits at floor/ceil. The upper child
must admit its minimum flow. These checks produce a feasible seed for each
combination. The frontier size is a power of two bounded by the requested
branches, pool size and available splits. The region containing each integer
flow vector is unique. Regions are defined in original coordinates by the
driver, independently of HiGHS's private node queue.

Frontier requires a neighbor to change at least one integer edge flow after
summing all increments on each edge. Empty proposals, zero transfers and
moves that cancel on every edge are discarded before feasibility and reward
scoring. Removing the last request on an edge and shifting traffic between
edges are valid changes. The reported `candidates_evaluated` counts generated
slots, including rejected slots.

One population covers all open regions. Every move, including shaking, must
preserve its trajectory's bounds after adding repeated updates to the same
edge. Every eight iterations the population shares records, reassigns slots
released by closing regions and gives spare slots to the better half of the
open regions. Each region can retain three distinct elite records. At exchanges
the leader is paired with its most different retained alternative by count of
unequal flows. A stripe through the population rotates alongside the best slots.

For each pair, differing edges that share a physical node form connected
components. Replacing an entire component preserves sender roles, node memory
and capacities because each affected node uses the incident flows of one
parent. Following the components in decreasing order of welfare gain defines a
feasible path. Its highest point can be constructed directly by applying all
positive-gain components. If all differences are connected the path has only
its endpoints. A child enters the exchange after independent feasibility and
region checks. Metal checks candidate moves on GPU and computes these components
on CPU. CUDA generates, scores, filters and applies moves on device and computes
the components and children for multiple parental pairs in a GPU batch.

## Optional final path relinking across regions

After the solver and VNS finish, `--final-relink-top K` collects the final
trajectory records and freezes one leader per represented region. It takes
the K highest-scoring leaders, pairs each with every other selected leader
once, and retains the best feasible child or the previous incumbent. With
32 represented parents it checks 496 pairs. The default 0 skips this stage.
Every child passes the original feasibility checker and is assigned to its
own region. The existing regional upper bounds continue to cover the full
space. An improved incumbent can therefore tighten the gap and close regions
after the HiGHS worker finishes.

`--time-limit` budgets the search; the final pass is additional.
`result.json` records the search and postprocessing times, the search record,
the best feasible value after collecting final trajectories, the final value,
and any closures triggered by the pass. The parents are fixed throughout
the pass. `final-parents.npz` stores their flows, region IDs, x, edge order,
and the search and pre-relink incumbents. `solution.npz` stores the final
checked allocation.

## Bounds and stopping

With `--tighten-receive`, the root LP, global MIP and regional MIPs all use
a receiver-specific coefficient in the sender-role constraint. For a given
node, reserve the RAM needed for the fixed local workload of every other
function. The remaining RAM bounds the integer replica count for this
function. Its maximum integer inflow must fit both those replicas (including
its local workload) and the total supply of its neighbors. Feasibility at a
replica boundary uses the same rounding rule as the independent checker.
This replaces only the coefficient and right-hand side of the receive-role
row. Omitting the flag selects the original formulation. The report includes
`coordinator_root_lp` for the initial continuous relaxation and
`coordinator_upper` for the final covering bound.

Enable `--pair-cuts` to limit the combined inbound traffic of two functions
sharing receiver RAM. The solver chooses the region splits before separating
cuts. For pairs receiving traffic in the LP, the separator reserves RAM for
mandatory replicas, enumerates extra integer requests for one function and
finds how many requests of the other fit in the remaining RAM. Only violated
inequalities are added. Tables above 65,537 entries are skipped. The root
cuts apply to every MIP. Each visited region also gets a separation pass
using its own lower and upper edge bounds, within its existing time slice.
The solver reports `root_pair_cuts` and the local `pair_cuts` count for each
regional solve. The option can be combined with `--tighten-receive`.

With `--group-cuts`, a single separation pass considers the functions with
positive LP inflow at each receiver. For three or more eligible functions it
solves an integer knapsack over their shared residual RAM and adds the joint
traffic limit if violated. With both cut options, a fresh relaxation after the
pair cuts supplies the group separator. A branch uses its mandatory flows and
its own edge upper bounds. The root splits are chosen before either option.
Oversized request tables and RAM tables over 65,537 entries are skipped.

With `--lagrangian`, the proof worker also splits each arc into a sending
and a receiving copy. For arc reward `w` and price `q`, the two copies earn
`w-q` and `q`. At each node the local maximization enforces sender/receiver
exclusivity per function, residual supply, fixed local service, integer
replicas, shared RAM and the region's flow bounds. Greedy allocation within
a function and integer knapsack across functions give the local maximum.
The sum is an upper bound because a feasible network flow agrees at both
endpoints. The solver keeps the smallest certified upper bound for each
region, including any tighter HiGHS bound.

The oracle batches the root nodes and both endpoints on each side of each
split. The existing node-disjoint partition makes their corrections additive
for all regions. LP prices seed projected subgradient updates. Certificates
use upward-rounded rewards on a binary grid and integer DP arithmetic with
an overflow check. The report under `global_highs.lagrangian` includes the
number of completed price evaluations, per-region bounds and their maximum.
A partial evaluation contributes no certificate. Up to 12 price evaluations
can use the first 10% of the remaining proof time. The option uses NumPy on
CPU for CPU/Metal search and CuPy integer arrays on CUDA for Spark.
The CUDA environment must provide the CuPy wheel matching its CUDA toolkit (on the GB10 install `cupy-cuda13x`) in addition
to the normal Spark VNS dependencies. The device memory option caps the
estimated oracle workspace. Runs without these switches retain the previous
proof settings.

One background worker initially spends a configurable fraction of its remaining
budget (`--global-share`, default 0.20) on a global MIP to obtain a shared
bound and incumbent. It then solves restricted MIPs in decreasing order of
regional upper bound, preferring less-visited regions on ties. Revisits receive
longer slices; they start new solver trees. With one region the global solve
gets the full budget. The CPU solver and heuristic exchange checked solutions.
Closed regions retain their numerical bounds in the final certificate.

The coordinator objective strips the constant local gain from welfare. For
target epsilon the stopping test is (U-L)/max(1,abs(L)) <= epsilon on the
coordinator objective, with U the maximum upper bound across all regions and
L the best global feasible value. Each closure records its branch, bound,
incumbent, time and triggering update. A closure is exact when U_branch <= L.
Numerical solver bounds are subject to the underlying HiGHS tolerances.

The wall-clock budget starts after loading and local initialization. An in-flight
pool exchange or solver shutdown may overrun it. The fresh output directory
contains a rechecked allocation and JSON with the welfare offset, coordinator
gap, covering bound, branch upper bounds, closure events and relinking counts.
Run the focused regression tests with
`python -m pytest -q centralized_vns_frontier/tests`.
