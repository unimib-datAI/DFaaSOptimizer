import logging
import sys

import pytest

from utils import run_log, step_time


class FakeClock:
  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t


@pytest.fixture(autouse=True)
def no_clock():
  step_time._clock.set(None)
  yield
  step_time._clock.set(None)


@pytest.fixture
def now(monkeypatch):
  fake = FakeClock()
  monkeypatch.setattr(step_time, "perf_counter", fake)
  return fake


class SlowStdout:
  """stdout whose every write takes ``delay`` seconds of the fake clock."""

  def __init__(self, now, delay):
    self.now, self.delay, self.text = now, delay, ""

  def write(self, text):
    self.now.t += self.delay
    self.text += text

  def flush(self):
    pass


log = run_log.logger("tests.run_log")


@pytest.mark.parametrize("verbose, expected", [
  (0, ["warning"]), (1, ["info", "warning"]), (2, ["debug", "info", "warning"]),
  (3, ["debug", "info", "warning"]),
])
def test_verbose_maps_to_levels(tmp_path, verbose, expected):
  with run_log.open_run_log(str(tmp_path), verbose):
    log.debug("debug")
    log.info("info")
    log.warning("warning")
  assert (tmp_path / "out.log").read_text().split() == expected


def test_messages_keep_their_exact_text(tmp_path):
  with run_log.open_run_log(str(tmp_path), 2):
    log.info("t = %d", 3)
    log.debug("    sp: DONE (ok; obj = %s; runtime = %s)", 1.5, 0.25)
    log.info("two\nlines")
  assert (tmp_path / "out.log").read_text() == (
    "t = 3\n    sp: DONE (ok; obj = 1.5; runtime = 0.25)\ntwo\nlines\n")


def test_logger_names_live_under_dfaas():
  assert run_log.logger("run_faasmadea").name == "dfaas.run_faasmadea"


def test_none_folder_writes_to_stdout_and_leaves_it_open(capsys, tmp_path):
  with run_log.open_run_log(None, 1):
    log.info("on stdout")
  assert capsys.readouterr().out == "on stdout\n"
  assert not sys.stdout.closed and not (tmp_path / "out.log").exists()


def test_exception_detaches_and_closes_the_handler(tmp_path):
  root = logging.getLogger("dfaas")
  before = list(root.handlers)
  with pytest.raises(RuntimeError):
    with run_log.open_run_log(str(tmp_path), 1):
      handler = root.handlers[0]
      log.info("before the failure")
      raise RuntimeError
  assert root.handlers == before and handler.stream is None
  assert (tmp_path / "out.log").read_text() == "before the failure\n"


def test_sequential_runs_have_separate_logs(tmp_path):
  for name in ("a", "b"):
    (tmp_path / name).mkdir()
    with run_log.open_run_log(str(tmp_path / name), 1):
      log.info(name)
  assert (tmp_path / "a" / "out.log").read_text() == "a\n"
  assert (tmp_path / "b" / "out.log").read_text() == "b\n"


def test_nested_run_restores_the_batch_log(capsys, tmp_path):
  root = logging.getLogger("dfaas")
  with run_log.open_run_log(None, 1):          # batch status on stdout
    log.info("batch start")
    with run_log.open_run_log(str(tmp_path), 2):
      log.debug("run detail")
    assert root.level == logging.INFO and not root.propagate
    log.info("batch end")
  assert capsys.readouterr().out == "batch start\nbatch end\n"   # the run's messages stay in its log
  assert (tmp_path / "out.log").read_text() == "run detail\n"
  assert not sys.stdout.closed


def test_emission_is_untimed(now, monkeypatch):
  monkeypatch.setattr(sys, "stdout", SlowStdout(now, 2.0))
  with run_log.open_run_log(None, 1):
    clock = step_time.start_step()
    now.t += 1.0
    log.info("progress")                      # 2 s to write
    row = step_time.end_step(clock)
  assert row["algorithm_s"] == pytest.approx(1.0) and row["wall_s"] == pytest.approx(3.0)


def test_complete_call_site_is_untimed_once(now, monkeypatch):
  monkeypatch.setattr(sys, "stdout", SlowStdout(now, 2.0))

  def prepare():
    now.t += 3.0                              # expensive argument (e.g. state.y.tolist())
    return "state"

  with run_log.open_run_log(None, 2):
    clock = step_time.start_step()
    now.t += 1.0
    with step_time.untimed():                 # the whole call site, emission nested inside
      if log.isEnabledFor(logging.DEBUG):
        log.debug("%s", prepare())
    row = step_time.end_step(clock)
  assert row["algorithm_s"] == pytest.approx(1.0) and row["wall_s"] == pytest.approx(6.0)


def test_disabled_level_skips_guarded_preparation(now, monkeypatch):
  monkeypatch.setattr(sys, "stdout", SlowStdout(now, 2.0))
  calls = []

  def prepare():
    calls.append(1)
    return "state"

  with run_log.open_run_log(None, 1):
    with step_time.untimed():
      if log.isEnabledFor(logging.DEBUG):
        log.debug("%s", prepare())
  assert calls == [] and sys.stdout.text == ""


def test_call_site_inside_a_task_including_a_failure(now, monkeypatch):
  monkeypatch.setattr(sys, "stdout", SlowStdout(now, 2.0))

  def prepare(fail):
    now.t += 3.0
    if fail:
      raise ValueError("bad state")
    return "state"

  with run_log.open_run_log(None, 2):
    clock = step_time.start_step()
    with step_time.parallel_phase():
      with step_time.task(0):
        now.t += 1.0                          # node 0: 1 s of work
        with step_time.untimed():
          log.debug("%s", prepare(False))
    row = step_time.end_step(clock)
    assert row["algorithm_s"] == pytest.approx(1.0) and row["wall_s"] == pytest.approx(6.0)
    step_time.start_step()
    with pytest.raises(ValueError):
      with step_time.parallel_phase():
        with step_time.task(0):
          with step_time.untimed():
            log.debug("%s", prepare(True))
    clock = step_time.start_step()            # the next step starts clean
    now.t += 4.0
    with step_time.untimed():
      log.info("done")
    row = step_time.end_step(clock)
  assert row["algorithm_s"] == pytest.approx(4.0) and row["wall_s"] == pytest.approx(6.0)


def test_quiet_nested_run_neither_suppresses_nor_leaks(capsys):
  with run_log.open_run_log(None, 1):          # a report tool: INFO on stdout
    log.info("report 1")
    with run_log.open_run_log(None, 0):        # a runner with verbose 0, also on stdout
      log.info("runner detail")
      log.warning("runner warning")
    log.info("report 2")
  assert capsys.readouterr().out == "report 1\nrunner warning\nreport 2\n"
