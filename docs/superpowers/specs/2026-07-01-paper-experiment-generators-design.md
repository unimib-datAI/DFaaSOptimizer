# Paper Experiment Generators Design

## Goal

Add registered `remote_experiments` suites that generate the complete experiment
matrices defined in `docs/hierarchical_model_experiment_plan.md`. The generated
`Experiment` objects must serialize through the existing `Batch` format and run
without manual config editing.

## Suite boundaries

One module, `remote_experiments/definitions/paper.py`, will register eight suites:

| Suite | Scientific scope | Default size |
|---|---|---:|
| `paper-e0-pilot` | Pipeline calibration | 120 |
| `paper-e1-quality-runtime` | RQ1 and RQ2; shared runs avoid duplication | 1,800 |
| `paper-e2-scalability` | RQ3 | 4,230 |
| `paper-e3-topology` | RQ4 | 1,260 |
| `paper-e4-robustness` | RQ5, static conditions | 1,260 |
| `paper-e5-dynamics` | RQ5, temporal conditions | 540 |
| `paper-e6-ablation` | RQ6 | 1,200 |
| `paper-e7-tradeoffs` | RQ7 | 1,680 |

The confirmatory suites total 11,970 runs. E0 uses 10 pilot seeds. E1–E7 use
the fixed confirmatory seeds 1001 through 1030. Builders accept optional `seeds`
and, where useful, optional `algorithms` so tests can exercise small matrices;
the registry and CLI call them without arguments and therefore generate the full
sets.

## Method sets

E1 uses all ten repository methods. E2 uses all nine non-centralized methods at
all sizes and adds centralized runs for 10 and 20 nodes. E3–E5 use the fixed,
predeclared representative set `hierarchical`, `faas-macro`, `faas-madea`,
`faas-diffuse`, `faas-powd`, and `faas-br-o`. E7 uses `hierarchical`,
`faas-macro`, `faas-madea`, and `faas-diffuse`. Fixing these sets before results
exist avoids post-hoc comparator selection.

## Configuration construction

All suites start from `config_files/eval_full.json`. Small helper functions copy
the base config, assign one exact factor cell, and build one `Experiment` per
method and seed. Every config has singleton node and function limits, an exact
topology definition, a fixed workload definition, and a unique
`base_solution_folder` equal to `solutions/<experiment-id>`.

Function-dependent arrays are expanded deterministically when `Nf` is 4 or 8:
demand cycles over `(1.0, 1.2)` and memory requirement cycles over `(2, 3)`.
This prevents the current two-function base vectors from producing invalid
larger-function configurations.

Experiment IDs contain the suite, factor labels, method, and seed. IDs use only
letters, digits, underscores, periods, and hyphens accepted by
`ray_dispatcher.Job`, and must be globally unique within each suite.

## Experiment-specific factors

E0 varies nodes `(10, 20, 50)` with two functions and four methods. E1 uses
nodes `(10, 20, 30)`, functions `(2, 4)`, and planar `k=3`. E2 uses nodes
`(10, 20, 50, 100, 200)`, functions `(2, 4, 8)`, and random-regular degree 3.

E3 uses 50 nodes, four functions, and seven topology cells: planar `k=3`,
random-regular degree `(3, 5, 7)`, and Erdős–Rényi probability `(0.1, 0.2, 0.3)`.
E4 uses the seven predefined conditions baseline, low/high load, scarce/ample
memory, homogeneous nodes, and strongly heterogeneous nodes. E5 uses the three
supported traces `sinusoidal`, `clipped`, and `fixed_sum_minmax`, with 100 time
steps.

E6 creates ten hierarchical variants: the base, hierarchy depth `(1, 2, 4, 5)`,
`eta` values `(0, 0.1, 0.5)`, and `epsilon` values `(0.001, 0.1)`. These are
crossed with node count `(20, 50)` and planar/random-regular degree-3 topology.
E7 uses seven `(latency_weight, fairness_weight)` pairs crossed with planar and
random-regular degree-3 topology.

## Connected random topology requirement

The existing probability-based topology generator can return disconnected
graphs. E3 requires connected graphs so connectivity is not confounded with
topology. `generate_neighborhood` will therefore resample probability-based
graphs with the same seeded RNG until connected, with a fixed maximum of 1,000
attempts. Exhaustion raises a descriptive `ValueError`. Existing deterministic
behavior remains reproducible for a fixed seed and configuration.

## Registration and CLI behavior

`remote_experiments/definitions/__init__.py` imports `paper` for registration.
No CLI changes are needed. A full batch is created with, for example:

```bash
uv run -m remote_experiments define paper-e1-quality-runtime \
  -o batches/paper-e1-quality-runtime.json
```

Each suite produces a separate batch and manifest, allowing execution, resume,
and postprocessing by research question.

## Validation

Focused tests will verify registration of all eight names, exact default counts,
unique IDs, output-folder consistency, dimension-vector expansion, E2's
centralized-size restriction, E3 topology coverage, E4 condition coverage, E5
trace coverage, E6 variant coverage, and E7 weight-pair coverage. A reduced-seed
batch will be serialized and loaded to prove compatibility with `Batch`.

The topology tests will prove that probability-based generation returns a
connected graph and raises after bounded failure. Full `remote_experiments` and
generator tests must pass before completion.

## Non-goals

This work does not run the experiments, implement analysis metrics, add dynamic
CLI arguments, tune parameters, or change result formats. Calibration values are
centralized as module constants so a later pilot can change them without altering
the suite structure.
