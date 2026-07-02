# Materialized experiment instances

The `materialize` command creates one ignored subdirectory per experiment suite
here. Each generated suite contains its own README, manifest, and checksummed
instance payloads. These directories are runtime data and are not committed.

If a frozen dataset must be archived, package the complete suite directory as a
ZIP and record the ZIP's SHA-256 checksum separately.
