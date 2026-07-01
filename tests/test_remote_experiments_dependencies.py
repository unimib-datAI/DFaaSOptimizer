from pathlib import Path

import tomli


def test_ray_dispatcher_source_is_remote_resolvable():
  config = tomli.loads(Path("pyproject.toml").read_text())
  source = config["tool"]["uv"]["sources"]["ray-dispatcher"]
  assert source == {
    "git": "https://github.com/miciav/ray-dispatcher.git",
    "rev": "30e91c81959eab30908675102e639a8953945049",
  }
