# Experimental Plan for the Hierarchical DFaaS Optimization Model

## 1. Study objective

This study will evaluate the proposed hierarchical auction model against the
centralized optimizer and the distributed optimization methods implemented in
DFaaSOptimizer. The evaluation is organized by research question rather than by
a single omnibus benchmark. Each experiment family isolates the factors needed
to answer one question while keeping all other factors controlled. This design
supports a full journal article with confirmatory comparisons, scalability
analysis, robustness tests, and mechanism-oriented ablations.

The central claim to test is that hierarchical coordination offers a favorable
compromise between solution quality and computational scalability. The study
must therefore demonstrate not only that the hierarchical model performs well,
but also when hierarchy helps, which components produce the gain, and what costs
are incurred in runtime, communication, latency, or fairness.

## 2. Compared methods

The following methods form the comparison set available in the repository.

| Label | Repository method | Role in the study |
|---|---|---|
| Centralized | `centralized` | Exact or bounded reference for small and medium instances |
| FaaS-MACrO | `faas-macro` | Primary decomposition-based comparator |
| FaaS-MACrO-v0 | `faas-macro-v0` | Non-accelerated MACrO reference |
| FaaS-MADeA | `faas-madea` | Distributed auction comparator |
| Hierarchical | `hierarchical` | Proposed method |
| FaaS-MADiG | `faas-diffuse` | Diffusion-based comparator |
| FaaS-MAPoD | `faas-powd` | Power-diffusion comparator |
| FaaS-MABR-S | `faas-br-s` | Sequential best-response comparator |
| FaaS-MABR-R | `faas-br-r` | Randomized best-response comparator |
| FaaS-MABR-O | `faas-br-o` | Re-optimization best-response comparator |

All methods will be run on the same generated instance for a given experimental
cell and seed. The centralized method will be omitted from large cells where it
cannot provide a useful solution or bound within the predefined time limit; the
omission will be reported explicitly rather than interpreted as a hierarchical
win.

## 3. Research questions and hypotheses

### RQ1 — Solution quality

Does the hierarchical model achieve better social welfare and lower rejection
than distributed alternatives on instances for which all methods can run?

**H1.** The hierarchical model will reduce the welfare gap to the centralized
reference relative to the distributed baselines, with the largest improvement
on sparse graphs where local coordination alone has limited reach.

### RQ2 — Quality–runtime trade-off

How close does the hierarchical model come to the centralized solution, and
what runtime and convergence cost does it pay?

**H2.** The hierarchical model will retain most of the centralized welfare while
requiring substantially less wall-clock time as network size increases.

### RQ3 — Scalability

How do runtime, iteration count, completion probability, and solution quality
scale with the number of nodes and functions?

**H3.** Hierarchical coordination will scale more favorably than centralized and
global decomposition methods, while degrading more slowly in solution quality
than purely local methods.

### RQ4 — Network structure

Under which topology, density, and connectivity conditions does hierarchical
coordination provide a measurable advantage when topology families are compared
at a common mean degree?

**H4.** The advantage will be strongest in sparse, high-diameter topologies and
will shrink as graph density makes direct coordination sufficient. This effect
will remain detectable after controlling for mean degree and edge-latency
distribution.

### RQ5 — Robustness to operating conditions

Is the hierarchical advantage preserved under load pressure, scarce memory,
and heterogeneous nodes and functions?

**H5.** Hierarchy will provide the largest reduction in rejected requests under
resource scarcity and heterogeneous capacity, where coordination across local
neighborhoods is most valuable.

### RQ6 — Mechanism and parameter ablations

Which hierarchy depths and pricing parameters are responsible for performance,
and is the full hierarchy necessary?

**H6.** Depth one will behave as a flat local auction, moderate depths will
improve welfare, and excessive depth will yield diminishing returns and higher
runtime. Congestion-aware price updates will outperform zero or fixed updates.

### RQ7 — Latency and fairness trade-offs

How does the hierarchical model trade social welfare against network latency
and allocation fairness when these terms are activated?

**H7.** Small positive latency and fairness weights will improve their respective
outcomes with limited welfare loss, whereas large weights will expose a clear
Pareto trade-off.

### RQ8 — Spatial scaling and distance-dependent latency

How does coupling network latency to Euclidean edge length affect performance as
the number of nodes grows at constant spatial density?

**H8.** Distance-dependent latency will favor allocations that use short local
paths. Hierarchical coordination will retain its quality advantage while
reducing latency-weighted traffic relative to methods that do not exploit the
spatial structure.

## 4. Common experimental protocol

### 4.1 Experimental unit and pairing

The experimental unit is one algorithm run on one generated DFaaS instance.
The instance is identified by the complete factor tuple and an instance seed.
All applicable methods receive the same topology, workload trace, resource
capacities, costs, and seed. Confirmatory cells use 30 independent instance
seeds. Pilot and tuning seeds must be disjoint from confirmatory seeds.

The seed lists will be frozen before confirmatory execution. If an algorithm
uses internal randomness beyond instance generation, its random seed will also
be recorded. Repeated runs on the same instance will only be introduced if a
pilot shows nondeterministic output.

### 4.2 Default scenario

Unless an experiment family states otherwise, the default scenario uses 50
nodes, 4 functions, a connected Euclidean planar graph with target mean degree
3, sinusoidal load, heterogeneous function demand, heterogeneous objective
weights, hierarchy depth 3, and the auction parameters `eta = [0.5, 0.3, 0.15]`,
`epsilon = 0.01`, and `zeta = 0.1`. Memory and utilization ranges will be
calibrated in the pilot so that the base scenario produces neither zero
rejection nor universal overload. The calibrated values will then be frozen for
all confirmatory runs.

### 4.3 Execution controls

Every run will use the same solver version, solver options, Python environment,
and code revision. Gurobi will use deterministic settings where applicable and
a fixed thread count per job. Jobs will run on homogeneous VM types. The remote
manifest will record host assignment and duration so host effects can be
checked. Concurrent jobs must not oversubscribe physical cores or solver
licenses.

The default time limit is 600 seconds for confirmatory runs. A shorter limit may
be used only after the pilot demonstrates that it does not censor relevant
methods. Timeouts, infeasible terminations, missing outputs, and validation
failures are outcomes and must remain in the analysis dataset.

### 4.4 Tuning policy

Hyperparameter tuning will use only dedicated pilot seeds. Each comparator may
use the parameterization recommended by its implementation, but the search
budget must be comparable across methods. The selected configuration and search
space will be reported. Confirmatory results must not be used to retune any
method.

### 4.5 Spatial graph and latency protocol

Euclidean planar instances will sample node coordinates at constant spatial
density. For an instance with `Nn` nodes and density `rho`, coordinates will be
sampled in a square with area `Nn / rho`, so the side length grows as
`sqrt(Nn / rho)`. This convention prevents larger instances from becoming
artificially crowded and keeps the characteristic distance between nearby nodes
approximately stable as network size increases.

The planar candidate graph will be the Delaunay triangulation of the sampled
coordinates. A connected Euclidean backbone will be retained, and additional
Delaunay edges will be sampled until the edge budget
`m = round(Nn * mean_degree / 2)` is reached. This construction guarantees a
straight-line planar embedding, connectedness, and the requested mean degree up
to the unavoidable integer rounding. The supported target range is bounded
below by the connected-graph requirement and above by the planar limit; the
confirmatory study uses target mean degrees 3 and 5.

Every Euclidean edge will store its geometric length separately from its network
latency. Under the distance-dependent mode, latency will be generated as a
fixed access delay plus a calibrated distance coefficient times edge length and
a bounded random jitter term. Bandwidth remains a separate edge attribute. For
multi-hop communication, end-to-end latency will be the weighted shortest-path
sum rather than the adjacency-matrix value of a non-edge.

Topology-only comparisons will use a common latency regime across topology
families, because assigning distance-dependent latency only to the planar family
would confound topology with edge cost. The spatial experiment instead compares
distance-dependent latencies with a paired permutation control on the same
Euclidean graph. The control preserves the multiset of edge latencies while
breaking their association with edge length, thereby isolating the effect of
spatially structured latency.

## 5. Outcome measures

### 5.1 Primary outcome

The primary outcome is normalized social welfare. On cells where the
centralized solver proves optimality, the main quality measure is the relative
optimality gap

\[
g = \frac{W_{\mathrm{centralized}} - W_{\mathrm{method}}}
         {\max(|W_{\mathrm{centralized}}|, \varepsilon)}.
\]

If the centralized run ends with a nonzero solver gap, comparisons will use its
incumbent and bound separately. Such runs will not be described as comparisons
against the exact optimum.

### 5.2 Secondary outcomes

Secondary outcomes are total rejected requests, rejection rate, locally
processed requests, offloaded requests, wall-clock runtime, algorithm-reported
runtime, iteration count, termination condition, completion rate, replica count,
memory utilization, and CPU utilization. Network cost will be computed by
weighting forwarded traffic by end-to-end shortest-path latency. Fairness will
be measured with Jain's index over per-node served-load ratios. The analysis
will also record the maximum and 95th-percentile node-level rejection rate to
expose tail behavior.

### 5.3 Required instrumentation

Objective, rejection, runtime, termination, offloading, replica, and utilization
outputs already exist in the current result format. Before RQ7, the analysis
pipeline must add deterministic derivations for latency-weighted traffic and
Jain fairness. Before making communication-efficiency claims, the hierarchical
runner must record counts of bids, accepted allocations, hierarchy-level
messages, and transferred tokens. Communication overhead is excluded from the
primary claims until that instrumentation is validated. Before RQ8, generated
instances must also persist node coordinates, edge lengths, direct-edge
latencies, and the all-pairs shortest-path latency matrix used by the methods.

## 6. Experiment families

### E0 — Pilot, feasibility, and calibration

This non-confirmatory experiment validates the full remote pipeline and selects
resource ranges that avoid trivial workloads. It uses 10 pilot seeds, node counts
10, 20, and 50, two functions, and the centralized, hierarchical, FaaS-MACrO,
and FaaS-MADeA methods. The pilot checks output completeness, central feasibility,
runtime distributions, timeout frequency, deterministic replay, and whether the
base scenario produces meaningful offloading and rejection. It also calibrates
the spatial density, fixed access delay, distance-to-latency coefficient, and
jitter range before these values are frozen for confirmatory experiments.

No E0 result will appear as confirmatory evidence. Its outputs may be reported
as implementation validation in supplementary material.

### E1 — Controlled quality comparison for RQ1 and RQ2

| Factor | Values |
|---|---|
| Nodes | 10, 20, 30 |
| Functions | 2, 4 |
| Topology | Connected Euclidean planar, target mean degree 3 |
| Load | Medium sinusoidal |
| Methods | All ten methods |
| Seeds | 30 confirmatory seeds |

E1 contains 1,800 runs. It is the primary comparison because all methods,
including the centralized reference, are expected to remain usable. The main
analysis compares paired welfare gaps and rejection rates. Runtime is analyzed
jointly with quality to determine whether improvements are obtained at an
acceptable computational cost.

### E2 — Scalability for RQ3

| Factor | Values |
|---|---|
| Nodes | 10, 20, 50, 100, 200 |
| Functions | 2, 4, 8 |
| Topology | Random regular, degree 3 |
| Methods | All non-centralized methods; centralized only for 10 and 20 nodes |
| Seeds | 30 confirmatory seeds |

E2 contains 4,050 non-centralized runs and 180 centralized runs. Primary outcomes
are wall-clock runtime, completion probability, and welfare normalized by the
best feasible result available for the same instance. Scaling curves will be
reported against both node count and the approximate problem size `Nn × Nf`.
Timeouts will not be discarded.

### E3 — Density-controlled topology and connectivity for RQ4

| Factor | Values |
|---|---|
| Nodes and functions | 50 nodes, 4 functions |
| Euclidean planar topology | Delaunay-derived connected graph, target mean degree 3 or 5 |
| Regular topology | Random regular degree 3 or 5 |
| Random topology | Connected Erdős–Rényi `G(n,m)`, target mean degree 3 or 5 |
| Methods | Hierarchical, FaaS-MACrO, FaaS-MADeA, FaaS-MADiG, FaaS-MAPoD, best MABR variant from E1 |
| Seeds | 30 confirmatory seeds |

E3 contains 1,080 runs. All topology families use the same edge budget, or its
closest feasible equivalent, at each target mean degree. Disconnected random
graphs will be regenerated using the same deterministic seed sequence, because
connectivity must not be confounded with topology family. Direct-edge latency
distributions will be matched across families and will not depend on Euclidean
length in E3. The analysis will relate method performance to realized mean
degree, diameter, clustering coefficient, and average shortest-path length, not
only to the generator parameters.

### E4 — Resource and workload robustness for RQ5

E4 varies one operating condition at a time around the frozen default scenario.
The seven unique conditions are baseline, low load, high load, scarce memory,
ample memory, homogeneous nodes, and strongly heterogeneous nodes. It uses 50
nodes, 4 functions, the default Euclidean planar topology, 30 seeds, and the same
six representative methods used in E3, for 1,260 runs.

Load levels will be calibrated by offered load relative to aggregate nominal
capacity rather than by arbitrary request counts. Memory conditions will be
defined by the fraction of function replicas that can be placed. Heterogeneity
will be controlled through predeclared capacity and demand distributions. The
primary robustness result is the method-by-condition interaction for welfare and
rejection rate.

### E5 — Dynamic workload behavior for RQ5

| Factor | Values |
|---|---|
| Trace | Sinusoidal, clipped sinusoidal, fixed-sum min/max |
| Timesteps | 100 |
| Nodes and functions | 50 nodes, 4 functions |
| Methods | Same six representative methods as E3 |
| Seeds | 30 confirmatory seeds |

E5 contains 540 runs, each evaluated across 100 timesteps. Analysis will report
time-averaged outcomes, worst-window rejection, adaptation lag after load changes,
and temporal variance. Timesteps are repeated observations within an instance
and will not be treated as independent samples.

### E6 — Hierarchical ablation and sensitivity for RQ6

E6 evaluates the proposed method only. The base configuration is compared with
hierarchy depths 1, 2, 4, and 5; zero congestion update (`eta = 0`); fixed updates
`eta = 0.1` and `eta = 0.5`; and alternative `epsilon` values `0.001` and `0.1`.
Together with the base configuration, this gives 10 variants. Variants are run
at 20 and 50 nodes on Euclidean planar graphs with target mean degree 3 and on
random-regular degree-3 graphs using 30 seeds, for 1,200 runs.

Depth one is the operational flat-auction ablation. The analysis will estimate
the marginal benefit of each additional hierarchy level and identify the
smallest depth whose welfare is practically equivalent to the best depth. A
parameter setting will not replace the base confirmatory configuration unless
the decision rule was specified before E1–E5 are inspected.

### E7 — Latency and fairness trade-offs for RQ7

E7 uses the supported `(latency_weight, fairness_weight)` pairs `(0,0)`,
`(0.25,0)`, `(1,0)`, `(0,0.25)`, `(0,1)`, `(0.25,0.25)`, and `(1,1)`. The
hierarchical model and the three strongest compatible distributed comparators
identified in E1 are tested on Euclidean planar and random-regular graphs with
target mean degree 3, 50 nodes, 4 functions, and 30 seeds. E7 uses the common
topology-only latency regime so that the weight sweep is not confounded with a
change in the latency generator. This produces 1,680 runs.

Results will be presented as Pareto fronts over welfare, latency-weighted
traffic, and fairness. A weighted objective value alone is insufficient because
it hides the magnitude of each constituent outcome.

### E8 — Spatial scaling and latency coupling for RQ8

E8 uses Euclidean planar graphs with target mean degree 3 at 20, 50, and 100
nodes. Spatial density, function count, workload intensity per node, and all
non-network parameters remain constant. Each generated topology is evaluated
under two paired latency assignments: distance-dependent latency and a
permutation of the same edge-latency values across the same edges. The
hierarchical method and the three strongest latency-compatible distributed
comparators identified in E1 use the predeclared latency weight 0.25 and are run
on 30 confirmatory seeds, producing 720 runs.

The primary contrast is the method-by-latency-assignment interaction for
normalized welfare, rejection, and latency-weighted traffic. Scaling trends are
estimated against node count while realized node density, edge-length
distribution, mean degree, graph diameter, and shortest-path latency are
reported as manipulation checks. Because the topology and latency multiset are
paired within each seed, differences can be attributed to the spatial coupling
between physical edge length and network latency rather than to graph density or
overall latency magnitude.

## 7. Statistical analysis plan

All main comparisons are paired by instance seed. For each primary contrast,
the analysis will report the paired median difference, paired bootstrap 95%
confidence interval, and a paired effect size. Method rankings will be secondary
to effect estimates.

For factorial experiment families, a mixed-effects model will include method,
the manipulated factor, and their interaction as fixed effects, with instance
seed as a random intercept. Runtime will be log-transformed when residual
diagnostics support that model. Rejection proportions will use a model suitable
for bounded outcomes or a bootstrap analysis if model assumptions are not met.

Pairwise post-hoc comparisons will use Holm correction within each research
question. The study will use a two-sided family-wise significance level of 0.05,
but scientific conclusions will prioritize confidence intervals and practical
effect sizes. A practical-equivalence margin of 1% normalized welfare will be
used when claiming that two methods have comparable solution quality.

Timeout and failure rates will be analyzed explicitly. Runtime summaries will
include completion-conditioned runtime and restricted mean runtime under the
common time limit. Assigning the time limit as if it were an observed runtime
will be used only for clearly labeled sensitivity analysis.

## 8. Planned tables and figures

The main article should contain a study-design diagram mapping each RQ to its
experiment family, a paired quality–runtime plot for E1, scalability curves for
E2, topology interaction plots for E3, robustness effect plots for E4–E5, an
ablation plot for E6, a Pareto plot for E7, and a spatial scaling interaction
plot for E8. Tables should report experimental factors, completion rates, effect
estimates with confidence intervals, and the final hyperparameters. Full
per-cell summaries and termination conditions belong in supplementary material.

No plot will aggregate across incompatible instance sizes without normalization
or stratification. Error bars will represent confidence intervals across
independent instances, not variability across timesteps treated as independent.

## 9. Reproducibility and data management

Each batch definition must be immutable and serialized before execution. All
problem instances will then be materialized once under the structured
`remote_experiments/instances/<suite>/` store before any algorithm is run. A
materialized instance includes the base optimization data, load limits, complete
temporal request traces, and the exact graph with node coordinates, edge
lengths, and latencies where applicable. Algorithms compared on the same cell
must reference the same instance identifier rather than regenerate inputs.

Each instance records a generation seed and SHA-256 checksums for all payload
files. Checksum validation is performed before dispatch and again when the
runner loads the transferred data. Stochastic algorithm seeds remain experiment
parameters and are not part of the instance payload. Every result will be linked
to the Git commit, configuration file, batch file, instance identifier, suite
manifest, method, algorithm seed, VM host, solver version, Python environment,
start time, termination condition, and output directory. The existing
`remote_experiments` workflow will be used for submission and resume.

Raw outputs remain read-only after collection. Postprocessing creates a tidy
dataset with one row per method–instance–timestep observation and a separate
instance table containing topology and workload descriptors. Analysis scripts
will consume only these versioned tables. Euclidean instances will additionally
record the spatial density, bounding area, node coordinates, edge lengths,
latency-generation mode, direct-edge latency summary, and shortest-path latency
summary. Exclusions and reruns require a reason code; failed runs must never be
silently replaced.

The confirmatory seed list, hypotheses, primary outcomes, practical-equivalence
margin, timeout policy, and statistical contrasts will be frozen in a protocol
file before confirmatory execution. Pilot, tuning, and confirmatory artifacts
will be stored separately.

## 10. Execution order and decision gates

Execution proceeds in the order E0, E1, E2, E3, E4–E5, E6, E7, and E8. E0 must
show complete artifacts and centrally feasible outputs before E1 starts. E1
selects the representative MABR method and the three compatible comparators used
in later families. E2 determines whether the 200-node cells are feasible within
the compute budget. E3–E5 establish external validity, E6 and E7 explain the
mechanism and trade-offs, and E8 tests whether the conclusions survive a
physically grounded spatial latency model.

If more than 10% of runs in a confirmatory cell fail for infrastructure reasons,
the entire cell will be rerun after the infrastructure issue is fixed. Algorithmic
timeouts and infeasibility are retained as results. Any protocol change after E1
begins must be versioned and described as exploratory.

The planned confirmatory workload is approximately 12,510 runs. After E0, the
observed runtime distribution will be used to estimate VM-hours and schedule the
families, but not to remove scientifically necessary cells. If resources are
insufficient, E7 becomes supplementary first, followed by reducing E8 to 20 and
100 nodes while preserving its paired design. The primary E1–E3 comparisons and
E6 ablation remain mandatory for the central claim.

## 11. Criteria for supporting the paper's claim

The evidence will support the proposed model if the hierarchical method shows a
practically meaningful paired welfare or rejection improvement over the strongest
distributed comparator in E1, maintains a favorable quality–runtime trade-off in
E2, and does not lose that advantage across most topology and robustness
conditions in E3–E5. E6 must show that genuine multi-level coordination, rather
than an incidental parameter choice, explains the improvement. Claims about
communication efficiency, latency, or fairness require the corresponding
instrumentation and E7 evidence.

Failure to outperform every baseline in every cell will not invalidate the
study. Evidence for spatial robustness additionally requires a favorable or
neutral method-by-latency-assignment interaction in E8; E7 alone cannot support
claims about physically grounded latency. The paper must report the conditions
where hierarchy is neutral or inferior, because identifying its operating
envelope is part of the scientific contribution.
