"""Registry of pluggable experiment suites: @register_suite("name") -> builder fn."""

from collections.abc import Callable

from ..batch import Experiment

_REGISTRY: dict[str, Callable[..., list[Experiment]]] = {}


def register_suite(name: str):
  def decorator(fn: Callable[..., list[Experiment]]) -> Callable[..., list[Experiment]]:
    if name in _REGISTRY:
      raise ValueError(f"suite already registered: {name}")
    _REGISTRY[name] = fn
    return fn
  return decorator


def get_suite(name: str) -> Callable[..., list[Experiment]]:
  if name not in _REGISTRY:
    raise KeyError(f"unknown suite: {name!r}; available: {sorted(_REGISTRY)}")
  return _REGISTRY[name]


def list_suites() -> list[str]:
  return sorted(_REGISTRY)
