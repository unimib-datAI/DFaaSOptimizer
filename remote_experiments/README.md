# remote_experiments

Define batches, materialize immutable inputs once, then run DFaaSOptimizer
experiments on Gurobi-equipped VMs via `ray-dispatcher` with a live TUI.

## Prerequisites

- VMs in the inventory need SSH reachability and Gurobi installed with a
  license (see `--gurobi-license`).
- VMs need outbound network access while provisioning Python and dependencies.
- `--project-path` defaults to `.` (this repo's root) — that's what gets
  rsynced to each VM, excluding `.venv/`, `.git/`, `solutions/`, `results/`,
  `batches/`, and the materialized instance store.

## Define a batch

```
uv run -m remote_experiments define smoke -o batches/smoke.json
```

Builds every `Experiment` for the named suite (a registered function under
`remote_experiments/definitions/`) and writes them to a static JSON file.

## Materialize the instances

```bash
uv run -m remote_experiments materialize batches/smoke.json
```

This creates `remote_experiments/instances/<suite>/`. Experiments whose input
generation specification is identical share one instance directory. Each
instance contains base optimization data, load limits, the complete temporal
request traces, the exact graph with edge attributes, and SHA-256 metadata.
Existing valid instances are reused; missing or modified files are rejected.

The generation seed recorded in instance metadata reproduces topology, weights,
capacities, and load traces. The experiment's runtime seed remains an algorithm
configuration. They are separate concepts even when a study initially assigns
them the same numeric value.

## Run (or resume) a batch

```
cp remote_experiments/inventory.yaml.example my-inventory.yaml  # edit hosts
uv run -m remote_experiments run batches/smoke.json --inventory my-inventory.yaml \
  --gurobi-license ~/gurobi.lic
```

Shows which experiments are pending (everything not yet `succeeded`,
tracked in `batches/smoke.manifest.json`), prompts for a selection
(indices, ranges, or `all` — default is every pending one), then submits
and shows a live progress view.

Ctrl-C cancels in-flight jobs and stops cleanly. Re-running the same `run`
command resumes — it defaults to selecting only what isn't `succeeded` yet.
Every selected job transfers only its own materialized instance. Use
`--instances <root>` when the store is not under `remote_experiments/instances`.

## Adding a suite

Add a file to `remote_experiments/definitions/` with a function decorated
`@register_suite("name")` returning a `list[Experiment]` (see `smoke.py`).

## Paper experiment batches

Generate the journal-study batches independently so each research question has
its own manifest and can be resumed separately:

```bash
for suite in \
  paper-e0-pilot \
  paper-e1-quality-runtime \
  paper-e2-scalability \
  paper-e3-topology \
  paper-e4-robustness \
  paper-e5-dynamics \
  paper-e6-ablation \
  paper-e7-tradeoffs \
  paper-e8-spatial-latency
do
  uv run -m remote_experiments define "$suite" -o "batches/$suite.json"
done
```

Materialize each batch before running it:

```bash
for batch in batches/paper-*.json; do
  uv run -m remote_experiments materialize "$batch"
done
```

Generated suite directories are ignored by Git. For archival, create a ZIP only
after materialization and retain its external SHA-256 checksum; ZIP files are
not used directly by the runners.
