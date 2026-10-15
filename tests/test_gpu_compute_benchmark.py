import csv
import json
import pytest
from experiments.gpu_compute import benchmark


@pytest.mark.parametrize('extra', [['--methods', 'unknown'], ['--repeats', '0'], ['--repeats', '-1']])
def test_invalid_cli(extra):
  with pytest.raises(SystemExit):
    benchmark.parse_args(['--config', 'config.json', '--output', 'unused', *extra])


def test_existing_output_rejected(tmp_path):
  with pytest.raises(FileExistsError):
    benchmark.main(['--config', 'unused.json', '--output', str(tmp_path)])


def test_summary_schema(tmp_path):
  rows = [dict(method='MADEA', seed=7, dp_device='cpu', compute_device='cpu',
               parallelism=0, phase='warm', repeat=i, wall_s=t, algorithm_s=t/2,
               runner_wall_s=t+1, rounds='it: 1', objective=3., feasible=True,
               diagnostic=False, peak_rss_kib=100, folder='run') for i,t in enumerate((1.,2.,3.))]
  benchmark.write_results(tmp_path, rows, [])
  summary=json.loads((tmp_path/'summary.json').read_text())
  assert summary['groups'][0]['median_wall_s'] == 2.
  assert summary['groups'][0]['p95_wall_s'] == pytest.approx(2.9)
  with (tmp_path/'samples.csv').open() as f:
    assert set(rows[0]) == set(next(csv.DictReader(f)))
  assert (tmp_path/'coverage.csv').is_file()
