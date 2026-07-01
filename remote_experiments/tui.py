"""rich-based live progress view: renders BatchStats/Manifest, no business logic."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ray_dispatcher import Inventory

from .batch import Batch
from .manifest import Manifest, SUCCEEDED
from .stats import BatchStats, summarize


def render(batch: Batch, manifest: Manifest, stats: BatchStats) -> Layout:
  summary = Table.grid(padding=(0, 2))
  summary.add_row(
    f"total: {stats.total}", f"succeeded: {stats.succeeded}", f"failed: {stats.failed}",
    f"running: {stats.running}", f"pending: {stats.pending}",
    f"throughput: {stats.throughput_per_min:.1f}/min",
    f"eta: {_format_seconds(stats.eta_s)}",
  )

  vm_table = Table(title="VMs")
  vm_table.add_column("host")
  vm_table.add_column("slots")
  vm_table.add_column("current jobs")
  vm_table.add_column("completed")
  for h in stats.hosts:
    vm_table.add_row(
      h.host, f"{h.slots_busy}/{h.slots_total}", ", ".join(h.current_jobs) or "-",
      str(h.jobs_completed),
    )

  exp_table = Table(title="Experiments")
  exp_table.add_column("id")
  exp_table.add_column("algorithm")
  exp_table.add_column("seed")
  exp_table.add_column("status")
  exp_table.add_column("host")
  for e in batch.experiments:
    exp_table.add_row(
      e.id, e.algorithm, str(e.seed), manifest.status(e.id), manifest.host(e.id) or "-",
    )

  layout = Layout()
  layout.split_column(
    Layout(Panel(summary, title="Batch"), size=3),
    Layout(vm_table, size=len(stats.hosts) + 4),
    Layout(exp_table),
  )
  return layout


def _format_seconds(seconds: float | None) -> str:
  if seconds is None:
    return "-"
  minutes, secs = divmod(int(seconds), 60)
  return f"{minutes}m{secs:02d}s"


@contextmanager
def live_view(
    batch: Batch, manifest: Manifest, inventory: Inventory, *, start_time: float,
  ) -> Iterator[Callable[[dict[str, str]], None]]:
  console = Console()
  experiment_ids = [e.id for e in batch.experiments]
  succeeded_at_start = sum(manifest.status(eid) == SUCCEEDED for eid in experiment_ids)
  with Live(console=console, refresh_per_second=4) as live:
    def on_tick(running_hosts: dict[str, str]) -> None:
      elapsed = time.monotonic() - start_time
      stats = summarize(
        experiment_ids, manifest, inventory, running_hosts, elapsed,
        succeeded_at_start=succeeded_at_start,
      )
      live.update(render(batch, manifest, stats))
    yield on_tick
