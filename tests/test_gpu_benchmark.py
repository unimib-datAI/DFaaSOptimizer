"""Benchmark CLI and result schema, with no long benchmarks in the suite."""
import csv
import json
import pytest
from experiments.gpu_dp import benchmark

@pytest.mark.parametrize('extra', [['--nodes', '0'], ['--nodes', '-1'], ['--repeats', '0'], ['--threads', '32']])
def test_invalid_benchmark_arguments(extra):
  with pytest.raises(SystemExit):
    benchmark.parse_args(['--instance', '.', '--output', 'unused', *extra])

def test_existing_output_is_never_overwritten(tmp_path):
  with pytest.raises(FileExistsError):
    benchmark.main(['--instance', str(tmp_path), '--output', str(tmp_path)])

def test_result_schema_and_percentiles(tmp_path):
  rows = [dict(scope='dp', problems=32, backend='cuda', effective_backend='cuda',
               gpu_fraction=1., phase='warm', threads=128, repeat=i, total_s=t,
               packing_s=.001, kernel_s=.002, upload_alloc_s=.003, download_s=.001,
               device_bytes=2000, synthetic=True, method='', welfare=None,
               runner_wall_s=None, algorithm_s=None) for i, t in enumerate([1., 2., 3.])]
  benchmark.write_results(tmp_path, rows, {'gpu': 'GB10', 'revision': 'abc'})
  result = json.loads((tmp_path / 'results.json').read_text())
  assert result['hardware']['gpu'] == 'GB10'
  assert result['summary'][0]['median_s'] == 2.
  assert result['summary'][0]['p95_s'] == pytest.approx(2.9)
  assert result['samples'] == rows
  with (tmp_path / 'samples.csv').open() as f:
    data = list(csv.DictReader(f))
  assert len(data) == 3 and data[0]['effective_backend'] == 'cuda'
  assert data[0]['gpu_fraction'] == '1.0'
  assert data[0]['hardware_gpu'] == 'GB10' and data[0]['revision'] == 'abc'


def test_binding_budget_reports_its_actual_gpu_share(monkeypatch, tmp_path):
  import sys
  from types import SimpleNamespace
  sys.path.insert(0, str(benchmark.ROOT / 'experiments/harmonization'))
  import capture_baseline
  def runner(config, *args, **kwargs):
    folder = __import__('pathlib').Path(config['base_solution_folder']) / 'run'
    folder.mkdir(parents=True)
    general = config['solver_options']['general']
    log = ('gpu_problems=3 cpu_problems=0' if 'algorithm_time_limit' in general else
           'gpu_problems=3 cpu_problems=3') if general.get('dp_device') == 'cuda' else ''
    (folder / 'out.log').write_text(log)
    (folder / 'base_instance_data.json').write_text('{"None":{"Nn":{"None":3}}}')
    (folder / 'runtime.csv').write_text('wall_s,algorithm_s\n1,1\n')
    (folder / 'obj.csv').write_text('objective\n1\n')
    return str(folder)
  monkeypatch.setattr(capture_baseline, 'baseline_runners', lambda: {'MADEA': runner, 'faas-macro': runner})
  monkeypatch.setattr(capture_baseline, 'capture_config', lambda instance, out, name:
                      {'base_solution_folder': str(out), 'solver_options': {'general': {}}})
  rows = []
  benchmark.benchmark_steps(SimpleNamespace(instance=tmp_path, output=tmp_path, repeats=1), rows)
  for row in rows:
    if row['scope'] == 'binding_budget':
      assert row['gpu_fraction'] == (1. if row['backend'] == 'cuda' else 0.)
