"""VNS-R keeps the baseline checker and acceptance, with independent moves."""
from collections import deque
import math
import time
import numpy as np
from centralized_vns.core import CPU as ReferenceCPU,Options
from centralized_vns.moves import score_move
from .moves import generate_move,incoming_index


class CPU(ReferenceCPU):
  score = staticmethod(score_move)

  def __init__(self,p,o):
    super().__init__(p,o)
    self.in_edges,self.in_starts=incoming_index(p)

  def step(self,iteration):
    p,o,s=self.p,self.o,self.state
    for index in range(o.pool_size):
      level=3 if self.stale[index]>=o.shake_after*3 else self.level[index]
      gains=[]
      for c in range(o.candidates):
        generate_move(o.seed,iteration,index,c,level,self.moves,s.y,s.incoming,s.rejected,
          p.x,p.capacity,p.memory,p.ram,p.src,p.dst,p.fun,p.starts,self.in_edges,self.in_starts)
        gains.append(self.score(index,c,self.moves,s.y,s.incoming,s.rejected,*p.arrays()))
      self.accept(index,level,gains)
    self.evaluated+=o.pool_size*o.candidates
    return self.best.copy()


def validate_plateau(patience,plateau_tolerance,shake_after):
  """Validate API/CLI settings before allocating an engine or creating outputs."""
  if type(patience) is not int or patience<0 or 0<patience<3*shake_after+1:
    raise ValueError(f'patience must be 0 or an integer >= {3*shake_after+1}')
  if (isinstance(plateau_tolerance,bool) or not isinstance(plateau_tolerance,(int,float)) or
      not math.isfinite(plateau_tolerance) or plateau_tolerance<0):
    raise ValueError('plateau_tolerance must be nonnegative and finite')


def solve(p,options=None,initial_y=None,*,patience=200,plateau_tolerance=1e-6):
  """Stop when every trajectory's archive plateaus over a full patience window.

  Patience counts iterations, not candidates; zero disables early stopping.
  A window must allow at least one fully stalled neighborhood/shaking cycle.
  """
  o=options or Options();o.validate()
  validate_plateau(patience,plateau_tolerance,o.shake_after)
  start=time.perf_counter()
  initial_y=np.zeros(len(p.src),dtype=np.int64) if initial_y is None else np.asarray(initial_y).copy()
  initial=p.check(initial_y)
  if o.backend=='cuda':
    from .gpu import CUDA
    engine=CUDA(p,o)
    for target,values in ((engine.y,initial_y),(engine.best_y,initial_y),
        (engine.inc,initial['incoming']),(engine.z,initial['z'])):
      target.copy_to_device(np.repeat(values[None],o.pool_size,axis=0).astype(np.int64))
    for target in (engine.value,engine.best):
      target.copy_to_device(np.full(o.pool_size,initial['welfare']))
  else:
    if o.backend=='metal':
      from .metal import Metal
      engine=Metal(p,o)
    else:
      engine=CPU(p,o)
    engine.state.y[:]=initial_y;engine.best_y[:]=initial_y
    engine.state.incoming[:]=initial['incoming'];engine.state.rejected[:]=initial['z']
    engine.value[:]=initial['welfare'];engine.best[:]=initial['welfare']
  trace=[dict(iteration=0,elapsed_s=time.perf_counter()-start,best_welfare=initial['welfare'])]
  history=deque([np.full(o.pool_size,initial['welfare'])],maxlen=patience+1) if patience else None
  reason='iteration_limit'
  for iteration in range(o.iterations):
    if time.perf_counter()-start>=o.time_limit:
      reason='time_limit'
      break
    best=engine.step(iteration)
    trace.append(dict(iteration=iteration+1,elapsed_s=time.perf_counter()-start,
                      best_welfare=float(np.max(best))))
    if trace[-1]['elapsed_s']>=o.time_limit:
      reason='time_limit'
      break
    if iteration+1>=o.iterations:
      break
    if history is not None:
      history.append(best.copy())
      if len(history)==patience+1:
        reference=history[0]
        # Archives survive shaking; every trajectory must plateau, not just the leader.
        if np.all((best-reference)/np.maximum(1.,np.abs(reference))<=plateau_tolerance):
          reason='plateau'
          break
  y=engine.result();checked=p.check(y)
  return dict(algorithm='VNS-R',x=p.x.copy(),y=y,**checked,trace=trace,
    elapsed_s=time.perf_counter()-start,candidates_evaluated=engine.evaluated,
    termination_reason=reason,iterations_completed=len(trace)-1,
    plateau=dict(patience=patience,tolerance=plateau_tolerance),
    backend=o.backend,initial_welfare=initial['welfare'])
