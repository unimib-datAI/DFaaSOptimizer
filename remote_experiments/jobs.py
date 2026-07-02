"""Maps an Experiment to a ray_dispatcher.Job that runs it on a VM."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ray_dispatcher import InputSpec, Job, OutputSpec

from .batch import Experiment
from .instances import PAYLOAD_FILES, instance_id, validate_instance

SCRIPT_BY_ALGORITHM: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
  "centralized":   (("run_centralized_model.py",), ()),
  "faas-macro":    (("run_faasmacro.py",), ()),
  "faas-macro-v0": (("run_faasmacro.py",), ("--v0",)),
  "faas-madea":    (("run_faasmadea.py",), ()),
  "hierarchical":  (("-m", "hierarchical_auction.runner"), ()),
  "faas-diffuse":  (("decentralized_diffusion.py",), ()),
  "faas-powd":     (("decentralized_powerd.py",), ()),
  "faas-br-s":     (("decentralized_bestresponse.py",), ("--variant", "s")),
  "faas-br-r":     (("decentralized_bestresponse.py",), ("--variant", "r")),
  "faas-br-o":     (("decentralized_bestresponse.py",), ("--variant", "o")),
}


def experiment_to_job(
    experiment: Experiment, config_dir: Path, instances_root: Path | None = None,
  ) -> Job:
  if experiment.algorithm not in SCRIPT_BY_ALGORITHM:
    raise KeyError(f"unknown algorithm: {experiment.algorithm!r}")
  command, extra_args = SCRIPT_BY_ALGORITHM[experiment.algorithm]
  config_dir.mkdir(parents=True, exist_ok=True)
  config_path = config_dir / f"{experiment.id}.json"
  config = copy.deepcopy(experiment.config)
  inputs = [InputSpec(source=str(config_path), destination="config.json")]
  if instances_root is not None:
    instance_path = (
      Path(instances_root) / experiment.suite / "data" / instance_id(experiment)
    )
    validate_instance(instance_path)
    config["limits"] = {
      "instance_type": "materialized",
      "path": "instance",
      "load": {"trace_type": "load_existing", "path": "instance"},
    }
    inputs.extend(
      InputSpec(source=str(instance_path / filename), destination=f"instance/{filename}")
      for filename in (*PAYLOAD_FILES, "metadata.json")
    )
  config_path.write_text(json.dumps(config, indent=2))
  return Job(
    id=experiment.id,
    command=("python", *command, "-c", "config.json", "--disable_plotting", *extra_args),
    inputs=tuple(inputs),
    outputs=(OutputSpec(source=f"solutions/{experiment.id}", destination=experiment.id, required=True),),
  )
