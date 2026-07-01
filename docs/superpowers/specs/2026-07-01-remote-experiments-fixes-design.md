# Remote Experiments Reliability Fixes

## Goal

Make `remote_experiments` provisionable and truthful in normal and resumed runs,
without expanding its architecture.

## Design

Replace the local `ray-dispatcher` source override with its Git repository pinned
in `uv.lock`. This makes the same dependency resolvable locally and on remote
hosts, where the sibling checkout does not exist.

Build jobs with the Python interpreter already exposed by `ray-dispatcher`'s
provisioned virtual environment. Top-level algorithms run their script directly;
the hierarchical runner uses `python -m hierarchical_auction.runner` so its
`types.py` cannot shadow the standard-library `types` module. No second `uv sync`
is needed inside a job.

Keep `run_batch`'s boolean contract: `True` means polling reached terminal state,
and `False` means interruption. After a terminal run, the CLI inspects the
selected experiments in the manifest and reports either success or the number of
non-successful experiments.

Capture the number of already-succeeded experiments when the live view starts.
Throughput is `(current successes - initial successes) / current-session elapsed
time`; persisted durations and ETA behavior remain unchanged.

## Error Handling

Dependency resolution continues to fail during provisioning with the existing
`ProvisioningError`. Job command failures remain represented by the dispatcher's
terminal statuses. The CLI must not label a terminal batch successful when any
selected experiment is not `succeeded`.

## Tests

- Assert generated commands use `python`, and hierarchical uses module mode.
- Assert a terminal batch containing a failure is not announced as successful.
- Assert resumed-session throughput excludes successes present at startup.
- Run all `remote_experiments` tests and verify the hierarchical module entry
  point directly.

## Non-goals

No dispatcher API changes, vendoring, submodules, new retry behavior, manifest
format changes, or TUI redesign.
