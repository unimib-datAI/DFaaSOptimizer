from plasma.sim.clock import RoundClock


def test_events_delivered_before_round_callback():
  clock = RoundClock()
  trace = []
  clock.schedule(1, lambda: trace.append("hb@1"))
  clock.run(2, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["round0", "hb@1", "round1"]


def test_fifo_tie_break():
  clock = RoundClock()
  trace = []
  clock.schedule(0, lambda: trace.append("a"))
  clock.schedule(0, lambda: trace.append("b"))
  clock.run(1, on_round=lambda r: None)
  assert trace == ["a", "b"]


def test_round_counter_persists_across_runs():
  clock = RoundClock()
  seen = []
  clock.run(3, on_round=seen.append)
  clock.run(2, on_round=seen.append)
  assert seen == [0, 1, 2, 3, 4]


def test_past_due_events_flush():
  clock = RoundClock()
  trace = []
  clock.run(2, on_round=lambda r: None)
  clock.schedule(0, lambda: trace.append("late"))
  clock.run(1, on_round=lambda r: trace.append(f"round{r}"))
  assert trace == ["late", "round2"]
