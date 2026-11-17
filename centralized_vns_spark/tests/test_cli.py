import json
from pathlib import Path
import numpy as np
import pytest
from centralized_vns_r.tests.test_io import INSTANCE
from centralized_vns_spark import __main__ as cli


@pytest.mark.parametrize('flags',[['--threads','32'],['--batch-size','0'],
  ['--patience','1'],['--x-iterations','-1'],['--refine-only'],['--x-time-limit','nan']])
def test_bad_cli_creates_no_output(tmp_path,flags):
  output=tmp_path/'result'
  with pytest.raises(SystemExit):
    cli.main([str(INSTANCE),'--output',str(output),*flags])
  assert not output.exists()


def test_both_stages_preserved_and_standalone_import(tmp_path,monkeypatch):
  from dataclasses import replace
  from centralized_vns_r.core import solve
  from centralized_vns_r.xstage import refine_x
  from centralized_vns.io import initialize
  def fixed(p,o,y,**kw):
    assert o.backend=='cuda'
    return solve(p,replace(o,backend='cpu'),y,patience=0)
  monkeypatch.setattr(cli,'solve',fixed)
  monkeypatch.setattr(cli,'initialize',initialize)
  monkeypatch.setattr(cli,'refine_x',lambda p,y,**kw:refine_x(p,y,
    **{k:v for k,v in kw.items() if k!='max_device_bytes'}))
  output=tmp_path/'both'
  cli.main([str(INSTANCE),'--output',str(output),'--init-device','cpu',
    '--pool-size','1','--candidates','2','--iterations','2','--refine-x','--x-iterations','0'])
  fixed_file=output/'fixed_x/solution.npz';original=fixed_file.read_bytes()
  assert (output/'variable_x/solution.npz').is_file()
  target=tmp_path/'standalone'
  cli.main([str(INSTANCE),'--output',str(target),'--solution',str(fixed_file),
    '--refine-only','--x-iterations','0'])
  assert fixed_file.read_bytes()==original
  with np.load(fixed_file) as a,np.load(target/'variable_x/solution.npz') as b:
    np.testing.assert_array_equal(a['x'],b['x'])
    np.testing.assert_array_equal(a['y'],b['y'])
  saved=json.loads((target/'variable_x/result.json').read_text())
  assert saved['floor_reference']=='input_solution_x'
  assert saved['solution_sha256']
  assert all('/' in name for name in saved['source_sha256'])


def test_output_exists_is_rejected_before_loading(tmp_path):
  with pytest.raises(SystemExit):
    cli.main(['missing','--output',str(tmp_path)])
