# Materialized Paper Instances Design

## Goal and constraints

The paper experiments must generate each problem instance once and execute every
algorithm against exactly the same immutable input. An instance includes the
base optimization data, graph topology and edge attributes, load limits, and the
complete temporal request trace. Algorithm settings, including stochastic
algorithm seeds, remain in the experiment configuration and are not stored as
instance data. Generated instance payloads are local artifacts under
`remote_experiments/instances/`; they are not committed by default.

An instance still needs a generation seed because random topology, capacities,
weights, and traces must be reproducible. This value is recorded explicitly as
`generation_seed` in metadata and is distinct in meaning from the runtime
algorithm seed. The current paper matrices may use the same integer for both;
the storage and execution paths nevertheless keep the two roles separate.

## Architecture and layout

The existing `Experiment` schema remains unchanged. A canonical generation
specification is derived from the fields that affect the materialized input:
`limits`, time horizon, trace timing, and generation seed. A short SHA-256 digest
of that canonical JSON becomes the stable instance identifier. Consequently,
all algorithms in the same experimental cell resolve to the same instance,
without duplicating data or adding another registry abstraction.

Each suite is written as:

```text
remote_experiments/instances/<suite>/
  README.md
  manifest.json
  data/<instance-id>/
    base_instance_data.json
    load_limits.json
    input_requests_traces.json
    graph.json
    metadata.json
```

`graph.json` uses NetworkX node-link JSON so Euclidean coordinates, edge
lengths, and network latencies survive materialization. `metadata.json` records
the schema version, instance identifier, suite, generation specification, and a
SHA-256 checksum for every payload file. The suite manifest maps experiments to
instances and summarizes the materialized set. The generated README describes
the format and counts for that suite.

## Data flow and validation

The CLI gains a `materialize` command that loads an existing batch, derives and
deduplicates its instance specifications, generates each missing instance, and
writes the suite documentation and manifest. Existing valid instances are
reused. Missing files, malformed metadata, or checksum mismatches are hard
errors; there is no automatic repair, database, or content-addressed object
store.

At execution time, `remote_experiments run` resolves each experiment's instance
from the same canonical specification. The job receives the five instance
files plus its algorithm configuration. Only the job's copy of the
configuration is changed: `limits` points to the transferred `instance/`
directory and selects the materialized loading mode. The original batch remains
unchanged.

All algorithm runners already call the shared `run_centralized_model.init_problem`.
That function therefore receives one additive branch: materialized mode verifies
checksums, loads the three optimization/load JSON files and exact graph, copies
the inputs into the solution folder for provenance, and returns the same tuple
as the existing generator path. Random and legacy `load_existing` behavior stay
unchanged.

## Error handling and tests

Materialization rejects inconsistent data for an existing identifier and never
silently overwrites corrupt artifacts. Job creation fails before dispatch when
an instance is absent or invalid. Remote loading validates again, protecting
against transfer corruption. Error messages identify the instance, file, and
expected versus actual checksum.

Tests cover deterministic deduplication across algorithms, complete file
creation, temporal trace inclusion, checksum rejection, exact graph round-trip,
job input transfer, CLI parsing, and the materialized branch of `init_problem`.
The existing generator path and all remote-experiment tests remain regression
coverage. No solver is required for the focused tests.
