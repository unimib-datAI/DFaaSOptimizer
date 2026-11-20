"""Install the additional neighborhood on this experimental pool only."""

from pathlib import Path
import numpy as np
from centralized_vns_r import metal
from centralized_vns_highs.mip import ram_receive_limit


def install(engine):
  mx, p = engine.mx, engine.p
  supply = (p.load - p.x)[p.src, p.fun]
  incoming = np.zeros_like(p.x)
  np.add.at(incoming, (p.dst, p.fun), supply)
  base = p.check(np.zeros(len(supply), dtype=np.int64))['replicas']
  cap = ram_receive_limit(p, incoming.ravel(), base).reshape(p.x.shape)
  eligible = np.flatnonzero((supply > 0) & (cap[p.dst, p.fun] > 0))
  if eligible.nbytes + engine.device_bytes > engine.o.max_device_bytes:
    raise MemoryError('eligible edge list exceeds device budget')
  src = Path(metal.__file__).with_suffix('.metal').read_text()
  marker = 'if (level==0) {'
  if src.count(marker) != 1:
    raise RuntimeError('unexpected VNS-R kernel layout')
  src = src.replace(marker, Path(__file__).with_name('evict.metal').read_text() + marker)
  engine.data.append(mx.array(eligible))
  engine.kernel = mx.fast.metal_kernel(
    name='vns_bc_evict',
    input_names=[
      'load',
      'x',
      'ram',
      'src',
      'dst',
      'fun',
      'starts',
      'costs',
      'memory',
      'limits',
      'in_edges',
      'in_starts',
      'eligible',
      'y',
      'inc',
      'z',
      'space',
      'fill',
      'release',
      'levels',
      'params',
    ],
    output_names=['moves', 'valid'],
    source=src,
  )
  engine.evictions = engine.double_evictions = 0
  accept = engine.accept

  def observed(index, level, gains):
    if level == 2:
      best = int(np.argmax(gains))
      if gains[best] > 1e-12 and best % 2 == 0 and len(eligible):
        engine.evictions += 1
        engine.double_evictions += int(engine.moves[index, best, 2, 1] < 0)
    return accept(index, level, gains)

  engine.accept = observed
