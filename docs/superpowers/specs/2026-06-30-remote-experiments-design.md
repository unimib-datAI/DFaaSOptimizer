# Remote Experiments — Design

## Purpose

Run batches of DFaaSOptimizer algorithm comparisons on remote, SSH-accessible
VMs with Gurobi installed, dispatched via the `ray-dispatcher` library. The
work splits into two static phases: **define** a batch of experiments to a
file, then **run** that file through a TUI that tracks per-experiment and
per-VM progress and supports stopping/resuming across process restarts.

## Scope

New package `remote_experiments/` inside `DFaaSOptimizer` (not a separate
repo). It reuses the existing entrypoint scripts
(`run_centralized_model.py`, `run_faasmacro.py`, `run_faasmadea.py`,
`decentralized_diffusion.py`, `decentralized_powerd.py`,
`decentralized_bestresponse.py`, `hierarchical_auction/runner.py`) and the
algorithm set already enumerated in `run.py`'s `METHOD_RESULT_MODELS`:
`centralized, faas-macro, faas-macro-v0, faas-madea, hierarchical,
faas-diffuse, faas-powd, faas-br-s, faas-br-r, faas-br-o`.

Dispatch, SSH transport, provisioning, retries, and local download of
partial results are all handled by `ray-dispatcher`
(`miciav/ray-dispatcher`, local checkout at `~/Downloads/ray-dispatcher`)
and are out of scope here — this project only produces `Job` objects for it
and reads back `JobStatus`/`JobResult`.

## Package layout

```
remote_experiments/
├── definitions/
│   ├── __init__.py        # registry: @register_suite("name") -> builder fn
│   ├── planar_comparison.py
│   └── eval_full.py        # one file per pluggable suite
├── batch.py                 # Experiment, Batch dataclasses + JSON (de)serialization
├── jobs.py                  # Experiment -> ray_dispatcher.Job
├── manifest.py               # persisted per-experiment status, for stop/resume
├── cli.py                    # `define` and `run` subcommands
└── tui.py                    # experiment selection + live progress view (rich)
```

## Data model

```python
@dataclass(frozen=True)
class Experiment:
    id: str                  # stable, e.g. "planar_comparison-faas-macro-seed7"
    suite: str
    algorithm: str           # key into SCRIPT_BY_ALGORITHM
    seed: int
    graph_params: dict       # Nn, Nf, neighborhood, etc.
    load_params: dict
    config: dict              # full config JSON to ship to the VM

@dataclass(frozen=True)
class Batch:
    suite: str
    experiments: tuple[Experiment, ...]
```

`Batch` serializes to a single JSON file. There is no database — the file
*is* the batch.

## Suite registry (pluggability)

Each file under `remote_experiments/definitions/` registers a builder:

```python
@register_suite("planar_comparison")
def build(seeds: list[int]) -> list[Experiment]:
    ...
```

The builder returns the cartesian product of graph config × load config ×
seeds × algorithms for that suite, with stable experiment ids (so resume can
match experiments across runs). Adding a suite means adding a file to
`definitions/` — no external registration mechanism (no entry_points).

## Two-phase workflow

1. **Define** (static, run once per batch):
   ```
   uv run -m remote_experiments define planar_comparison -o batches/foo.json
   ```
   Looks up the suite in the registry, builds the `Experiment` list, writes
   `batches/foo.json`. This file is the static entity referenced by
   `run`.

2. **Run** (TUI):
   ```
   uv run -m remote_experiments run batches/foo.json
   ```
   - Loads `batches/foo.json` and the associated manifest
     `batches/foo.manifest.json` (created on first run).
   - Shows experiments with their current status (✓ succeeded / pending /
     failed / cancelled) and prompts for which to launch — default
     selection is everything not `SUCCEEDED` (this default *is* the resume
     mechanism, no separate `resume` command).
   - Selection input is plain text (indices / ranges / `all` via
     `rich.prompt`), not a graphical checkbox widget — avoids adding a new
     dependency (`textual`, `questionary`).
   - Submits selected experiments via `Dispatcher.submit()` under a fresh
     `batch_id` (uuid4), then switches to the live progress view.

## Job mapping

For each selected `Experiment`, `jobs.py` writes `experiment.config` to a
local temp file and builds:

```python
Job(
    id=experiment.id,
    command=("uv", "run", SCRIPT_BY_ALGORITHM[experiment.algorithm], "-c", "config.json"),
    inputs=(InputSpec(source=local_config_path, destination="config.json"),),
    outputs=(OutputSpec(source=f"solutions/{experiment.id}", destination=experiment.id),),
)
```

`SCRIPT_BY_ALGORITHM` mirrors `METHOD_RESULT_MODELS` from `run.py`. The
`Project` passed to `Dispatcher` points at the `DFaaSOptimizer` repo root
(so the VM has `models/`, `generators/`, the `decentralized_*.py` scripts,
etc. available) and carries the Gurobi license as a secret:

```python
Project(
    path=".",
    project_id="dfaas-optimizer",
    python="3.10.x",
    uv_version="...",
    secrets=(SecretFile(source="~/gurobi.lic", remote_name="gurobi.lic", env_var="GRB_LICENSE_FILE"),),
)
```

## Manifest and stop/resume

`manifest.py` persists, per `experiment_id`:
`{status, host, started_at, finished_at, duration_s, attempt}`, written on
every status transition observed during polling — so it survives a hard
kill, not just Ctrl-C.

- **Stop**: Ctrl-C in the TUI is caught; all non-terminal handles get
  `dispatcher.cancel(handle)`. The manifest reflects whatever state each
  job reached (`CANCELLED`/`RUNNING`/`PENDING` — any non-`SUCCEEDED` value
  is enough for the resume logic).
- **Resume**: re-running `run batches/foo.json` reads the manifest, offers
  every non-`SUCCEEDED` experiment as the default selection, and submits
  them under a new `batch_id`. `ray-dispatcher` itself has no batch-resume
  concept (`Dispatcher.submit` raises `BatchExistsError` if the batch
  directory already exists) — resume is implemented entirely in this
  project's manifest, keyed by the stable `experiment.id` rather than by
  `batch_id`.

## TUI content

`rich.Live`, refreshed roughly once per second by polling
`dispatcher.status(handle)` for every outstanding handle:

- **Batch summary panel**: total / succeeded / failed / running / pending
  counts, throughput (completed jobs per minute), elapsed time, and an
  **ETA** computed as
  `average_duration_of_completed_jobs * remaining_jobs / total_slots` —
  a simple heuristic, not a predictive model.
- **VM panel**: one row per `RemoteHost` from the `Inventory`, showing
  slots busy/total, the job currently running there, and jobs completed on
  that host so far. Busy/idle/lost state is derived by cross-referencing
  live job status with `inventory.hosts` (slot totals are static
  configuration; occupied slots are jobs currently `RUNNING` on that
  host).
- **Experiments table**: id, algorithm, seed, status, assigned host,
  duration/elapsed.

## Testing

No framework beyond `pytest` (already a dev dependency). One test per
non-trivial behavior:
- a sample suite builder produces the expected number of combinations with
  unique experiment ids
- `Batch` JSON round-trip
- `Experiment -> Job` mapping produces the expected command/config/io specs
- ETA heuristic computes correctly against fabricated completed/pending
  counts and durations

## Out of scope

- SSH transport, provisioning, retries, partial-result download: owned by
  `ray-dispatcher`.
- Interactive in-TUI stop/resume controls (textual-style keybindings) —
  stop/resume is a cross-process workflow (Ctrl-C, then re-run `run`).
- External/third-party suite plugins (entry_points) — suites live only
  inside `remote_experiments/definitions/`.
