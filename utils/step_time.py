"""Modelled distributed time of one algorithm step.

Each node is modelled as its own machine: a parallel phase (per-node work done
simultaneously) counts its longest task, sequential code counts its wall time,
observer code (checks, export, logging) does not count.

  modelled = wall - sum(untimed) - sum(phase wall) + sum(phase time)
  phase time = longest task + phase wall outside tasks and pool calls (set-up, merging)
"""
from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from time import perf_counter

import pandas as pd

COLUMNS = ("step", "algorithm_s", "wall_s", "parallel_phases", "workers")
_clock = contextvars.ContextVar("step_clock", default=None)
_phase = contextvars.ContextVar("step_phase", default=None)


class Phase:
  def __init__(self) -> None:
    self.tasks: dict = {}
    self.nodes = 0
    self.covered = 0.0  # wall time of in-process tasks and pool calls
    self.paused = 0.0   # untimed wall time inside the phase

  def add(self, key, seconds: float) -> None:
    self.tasks[key] = self.tasks.get(key, 0.0) + seconds

  def uniform(self, nodes: int) -> None:
    """Vectorized phase: the same work per node, longest task = wall / nodes."""
    self.nodes = max(self.nodes, int(nodes))

  def longest(self, wall: float) -> float:
    wall -= self.paused
    if self.nodes:
      return wall / self.nodes
    if not self.tasks:  # nothing recorded: count it all (a forgotten timed() errs high)
      return wall
    return min(wall, wall - self.covered + max(self.tasks.values()))

  def timed_task(self, key, started: float, paused: float) -> None:
    """In-process task from ``started``, minus the untimed time spent in it."""
    seconds = perf_counter() - started - (self.paused - paused)
    self.add(key, seconds)
    self.covered += seconds


class StepClock:
  def __init__(self) -> None:
    self.started = perf_counter()
    self.removed = 0.0
    self.phases = 0
    self.workers = 0

  def wall(self) -> float:
    return perf_counter() - self.started

  def modelled(self) -> float:
    return self.wall() - self.removed


def start_step() -> StepClock:
  clock = StepClock()
  _clock.set(clock)
  return clock


def end_step(clock: StepClock) -> dict:
  row = dict(algorithm_s=clock.modelled(), wall_s=clock.wall(), parallel_phases=clock.phases,
             workers=clock.workers)
  _clock.set(None)
  return row


@contextmanager
def parallel_phase():
  if _phase.get() is not None:
    raise RuntimeError("nested parallel phase: each per-node computation belongs to one phase")
  phase = Phase()
  token = _phase.set(phase)
  started = perf_counter()
  try:
    yield phase
  finally:
    _phase.reset(token)
    wall = perf_counter() - started
    clock = _clock.get()
    if clock is not None:
      clock.removed += wall - phase.longest(wall)
      clock.phases += 1


def parallel(fn):
  """Decorator: one call of ``fn`` is one parallel phase."""
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    with parallel_phase():
      return fn(*args, **kwargs)
  return wrapper


@contextmanager
def task(key):
  phase = _phase.get()
  started, paused = perf_counter(), phase.paused if phase else 0.0
  try:
    yield
  finally:
    if phase is not None:
      phase.timed_task(key, started, paused)


def record(key, seconds: float) -> None:
  """Task measured elsewhere (in a pool worker): call it inside ``offloaded``."""
  phase = _phase.get()
  if phase is not None:
    phase.add(key, seconds)


@contextmanager
def offloaded(workers: int):
  """A pool call of the current phase: its wall time is covered by the recorded tasks."""
  started = perf_counter()
  try:
    yield
  finally:
    phase, clock = _phase.get(), _clock.get()
    if phase is not None:
      phase.covered += perf_counter() - started
    if clock is not None:
      clock.workers = max(clock.workers, workers)


def timed(items, key=lambda item: item):
  """Iterate ``items`` timing each loop iteration as a task of ``key(item)``.

  Same as wrapping the loop body in ``task(key(item))``, without re-indenting it;
  an iteration left with ``break`` or an exception is recorded in its own phase.
  """
  for item in items:
    with task(key(item)):
      yield item


def current_modelled() -> float:
  clock = _clock.get()
  return 0.0 if clock is None else clock.modelled()


@contextmanager
def untimed():
  started = perf_counter()
  try:
    yield
  finally:
    phase, clock = _phase.get(), _clock.get()
    if phase is not None:
      phase.paused += perf_counter() - started
    elif clock is not None:
      clock.removed += perf_counter() - started


def algorithm_time_limit(config: dict) -> float:
  general = config.get("solver_options", {}).get("general", {})
  return float(general.get("algorithm_time_limit", float("inf")))


def workers(parallelism: int) -> int:
  if parallelism >= 0:
    return parallelism
  try:
    return len(os.sched_getaffinity(0))
  except AttributeError:  # macOS
    return os.cpu_count() or 1


def write_runtime(folder: str, rows: list[dict]) -> None:
  pd.DataFrame(rows, columns=list(COLUMNS)).to_csv(os.path.join(folder, "runtime.csv"), index=False)
