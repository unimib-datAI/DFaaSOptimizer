# VNS-R and VNS-RX

User-approved direction: preserve existing VNS; introduce a differently named
variant with replica-aware neighborhoods; then refine x while retaining both
solutions. User explicitly chose the selfish-centralized floor, not full node
utility: sum_f alpha[n,f]*x[n,f]/max(load[n,f],1) must stay at least its initial
DP value for every node (float64 tolerance 1e-12).

New package `centralized_vns_r`, algorithm names VNS-R and VNS-RX. Existing
`centralized_vns` remains callable with identical behavior. Reuse its Problem,
checker, Options, acceptance and backend allocation; do not edit old algorithms.

VNS-R keeps fixed x and the three sampled neighborhoods, augmenting them with
replica-boundary insertion/rerouting and atomic relocation plus insertion. Four
edge changes remain the bound. Keep ordinary unit/full candidates and seeded
shaking. CPU, Metal and CUDA must generate identical moves at fixed states/seeds;
Metal gets integer boundary amounts computed on CPU in float64. CUDA preserves
float64 arithmetic and existing parallel acceptance. No tensor float32 shortcut.

The first VNS-RX stage is explicit CPU refinement of the fixed-stage incumbent,
not a hidden fallback of the chosen fixed-stage GPU backend. Consider local
admission, localization of offloaded work and paired local-function changes,
with receiver insertion repair where useful. Apply whole transactions only after
independent physical checks, all local floors and global welfare improvement.
Keep immutable initial DP floors throughout; never recompute them from a trial.
No promise of an exhaustive neighborhood, optimality or gain on every instance.

`core.solve(p, options=None, initial_y=None)` returns fixed-stage x/y and baseline
result fields, named VNS-R. `xstage.refine_x(p,y,*,iterations=1000,time_limit=15.,
seed=4850)` returns x/y, checked result fields, floor/local-gain vectors, attempted
and accepted proposals, trace and elapsed time, named VNS-RX with backend cpu.
The input p contains initial DP x and must remain unchanged; y is not mutated.

CLI `python -m centralized_vns_r INSTANCE --output NEW_DIR` runs the fixed stage
and optionally x refinement (explicit --refine-x, separate iterations/time flags).
Save `fixed_x/{solution.npz,result.json}` before any variable stage starts and
`variable_x/{solution.npz,result.json}` separately. Output directory must be new.
Persist hashes, options, timing scopes, algorithm/stage identity and local floors.
Both variants remain independently runnable/comparable with original VNS.

Additional explicit user requirement: x refinement is standalone and applicable
to selfish-distributed outputs. Command `python -m centralized_vns_r.refine
INSTANCE --solution FILE_OR_FOLDER --output NEW_DIR` imports checked dense/sparse
NPZ or native LSPc CSV/gzip results. It never calls initialization or fixed-stage
VNS. The imported x defines the permanent floor, matching the original local plan
for selfish-distributed; report that provenance rather than rerunning tied DP.
Input rounding tolerance is 1e-6 solely for integer solver-export noise; after
rounding, independently verify all physical constraints. Preserve input artifacts.

Alternative considered: replacing the old generator in-place would lose a clean
baseline; rejected. A fully trajectory-resident GPU x stage is a larger state/
kernel change; defer it until the guarded CPU stage establishes useful moves.
Targeted large destroy/repair is deferred: this version tests the first two
recommended neighborhood improvements separately from x revision.

Validation: hand-crafted threshold and coupled-move wins, exact CPU/Metal/CUDA
move/state parity, boundary tolerance, infeasible roles, zero edges/load, floor
violations despite global improvement, meaningful guarded x improvement,
input immutability, zero budgets and CLI preservation. Run real Mac smoke and
500-node comparison; never infer Spark compilation/performance from CUDASIM.
