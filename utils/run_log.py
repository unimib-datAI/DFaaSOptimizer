"""One way to log: ``dfaas.<name>`` loggers, one handler per run on the ``dfaas`` logger.

  log = run_log.logger(__name__)              # once, at import
  with run_log.open_run_log(folder, verbose):  # folder: <run>/out.log; None: stdout
    log.info("t = %d", t)

verbose 0 -> WARNING, 1 -> INFO, >= 2 -> DEBUG; format "%(message)s" (out.log keeps the text the
log parsers read). Emission is observer code (step_time.untimed); keep the whole call site, with
its level guard and argument preparation, inside ``step_time.untimed()`` during a step.
"""
import logging
import os
import sys
from contextlib import contextmanager

from utils import step_time

ROOT = "dfaas"


def logger(name: str) -> logging.Logger:
  return logging.getLogger(f"{ROOT}.{name}")


def level(verbose: int) -> int:
  return logging.WARNING if verbose <= 0 else logging.INFO if verbose == 1 else logging.DEBUG


class _Untimed:
  def emit(self, record):
    with step_time.untimed():
      super().emit(record)


class _FileHandler(_Untimed, logging.FileHandler):
  pass


class _StreamHandler(_Untimed, logging.StreamHandler):
  pass


@contextmanager
def open_run_log(folder: str | None, verbose: int):
  """Attach this run's handler to ``dfaas``; an enclosing (batch) handler is restored on exit."""
  root = logging.getLogger(ROOT)
  previous = (root.handlers[:], root.level, root.propagate)
  if folder is None:
    handler = _StreamHandler(sys.stdout)       # closing it never closes stdout
  else:
    handler = _FileHandler(os.path.join(folder, "out.log"), mode="w")
  handler.setFormatter(logging.Formatter("%(message)s"))
  root.handlers, root.propagate = [handler], False
  root.setLevel(level(verbose))
  try:
    yield root
  finally:
    root.handlers, root.propagate = previous[0], previous[2]
    root.setLevel(previous[1])
    handler.close()
