"""Run an existing VNS-R engine continuously, reseeding one trajectory at a time."""
import numpy as np

from centralized_vns_r.core import CPU


class Pool:
  def __init__(self, p, options, y, checked, batch_size):
    self.cuda = options.backend == 'cuda'
    if self.cuda:
      from centralized_vns_spark.gpu import CUDA
      self.engine = CUDA(p, options, y, patience=0, batch_size=batch_size)
    else:
      engine_class = CPU
      if options.backend == 'metal':
        from centralized_vns_r.metal import Metal
        engine_class = Metal
      self.engine = engine_class(p, options)
      self._seed_host(slice(None), y, checked)

  def _seed_host(self, slot, y, checked):
    engine = self.engine
    engine.state.y[slot] = engine.best_y[slot] = y
    engine.state.incoming[slot] = checked['incoming']
    engine.state.rejected[slot] = checked['z']
    engine.value[slot] = engine.best[slot] = checked['welfare']
    engine.level[slot] = engine.stale[slot] = 0

  def inject(self, y, checked):
    engine = self.engine
    if self.cuda:
      slot = int(np.argmax(engine.stale.copy_to_host()))
      for target, value in ((engine.y, y), (engine.best_y, y),
                            (engine.inc, checked['incoming']), (engine.z, checked['z']),
                            (engine.best, checked['welfare']), (engine.value, checked['welfare']),
                            (engine.level, 0), (engine.stale, 0)):
        target[slot:slot+1].copy_to_device(np.asarray(value, dtype=target.dtype)[None])
    else:
      self._seed_host(int(np.argmax(engine.stale)), y, checked)

  def advance(self, iteration, count):
    if self.cuda:
      return self.engine.advance(iteration, count, 0., 'all')
    for step in range(iteration, iteration + count):
      best = self.engine.step(step)
    return float(np.max(best))
