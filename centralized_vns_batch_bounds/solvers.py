"""Compare sequential dual simplex and batched preconditioned PDHG on Metal."""

from time import monotonic

import highspy
import numpy as np
from centralized_vns_highs.mip import require_ok


def highs(lp, lo, hi, seconds):
  h = lp.solver
  changed = np.flatnonzero(((lo != lp.lo) | (hi != lp.hi)).any(axis=0)).astype(np.int32)
  start = monotonic()
  results = [None] * len(lo)
  optima = [None] * len(lo)
  try:
    for k, (lower, upper) in enumerate(zip(lo, hi)):
      left = seconds - (monotonic() - start)
      if left <= 0:
        break
      require_ok(h.changeColsBounds(len(changed), changed, lower[changed], upper[changed]))
      require_ok(h.setBasis(lp.basis))
      require_ok(h.setOptionValue('time_limit', left))
      if h.run() == highspy.HighsStatus.kError:
        raise RuntimeError('regional HiGHS solve failed')
      if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        if h.getModelStatus() != highspy.HighsModelStatus.kTimeLimit:
          raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
        continue
      ub = lp.certificate(-np.asarray(h.getSolution().row_dual), lower, upper)
      if monotonic() - start <= seconds:
        results[k] = ub
        optima[k] = float(lp.base) - h.getObjectiveValue()
  finally:
    require_ok(h.changeColsBounds(len(changed), changed, lp.lo[changed], lp.hi[changed]))
    require_ok(h.setOptionValue('time_limit', np.inf))
  return dict(bounds=results, lp_optima=optima, seconds=monotonic() - start)


def metal(lp, lo, hi, seconds, iterations=8192):
  start = monotonic()
  import mlx.core as mx

  if not mx.metal.is_available():
    raise RuntimeError('Metal device required')
  mx.set_default_device(mx.gpu)
  batch = len(lo)

  def tensor(a):
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))

  kernel = mx.fast.metal_kernel(
    name='regional_lp_spmm',
    input_names=['ptr', 'col', 'val', 'v'],
    output_names=['out'],
    source="""
    uint i = thread_position_in_grid.x;
    uint row = i / B, lane = i % B;
    float sum = 0;
    for (int k=ptr[row]; k<ptr[row+1]; ++k)
      sum = fma(val[k], v[col[k]*B+lane], sum);
    out[i] = sum;
  """,
  )

  def multiply(matrix):
    ptr, col, val = mx.array(matrix.indptr), mx.array(matrix.indices), tensor(matrix.data)
    count = matrix.shape[0]

    def op(v):
      return kernel(
        inputs=[ptr, col, val, mx.contiguous(v)],
        template=[('B', batch)],
        grid=(count * batch, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(count, batch)],
        output_dtypes=[mx.float32],
      )[0]

    return op

  A = multiply(lp.A)
  AT = multiply(lp.A.T.tocsr())
  c, b = tensor(lp.c[:, None]), tensor(lp.b[:, None])
  low, high = tensor(lo.T), tensor(hi.T)
  weight = float(np.median(lp.c[lp.c > 0])) if np.any(lp.c > 0) else 1.0
  tau = tensor(0.95 / (weight * np.maximum(1, np.asarray(abs(lp.A).sum(0)).T)))
  sigma = tensor(0.95 * weight / np.maximum(1, np.asarray(abs(lp.A).sum(1))))
  x = mx.clip(tensor(np.array(lp.root.col_value)[:, None]), low, high)
  y = mx.broadcast_to(
    tensor(np.maximum(0, -np.asarray(lp.root.row_dual))[:, None]), b.shape[:1] + (batch,)
  )
  extrapolated = x

  def estimate(prices):
    rc = c - AT(prices)
    return mx.sum(b * prices, axis=0) + mx.sum(mx.maximum(rc * low, rc * high), axis=0)

  best = y
  score = estimate(y)
  mx.eval(x, best, score)
  results = [None] * batch
  steps = 0
  checkpoint = 64
  cert_seconds = 0.0

  def certify():
    t = monotonic()
    host = np.asarray(best).T
    for k in range(batch):
      if monotonic() - start >= seconds:
        break
      bound = lp.certificate(host[k], lo[k], hi[k])
      if monotonic() - start <= seconds:
        results[k] = bound if results[k] is None else min(results[k], bound)
    return monotonic() - t

  cert_seconds = certify()
  while steps < iterations and monotonic() - start + 1.25 * cert_seconds < seconds:
    for _ in range(min(checkpoint, iterations - steps)):
      y = mx.maximum(0, y + sigma * (A(extrapolated) - b))
      new = mx.clip(x + tau * (c - AT(y)), low, high)
      extrapolated, x = 2 * new - x, new
      steps += 1
      if steps % 8 == 0:
        mx.eval(y, x, extrapolated)
        if monotonic() - start + 1.25 * cert_seconds >= seconds:
          break
    candidate = estimate(y)
    improved = mx.isfinite(candidate) & (candidate < score)
    best, score = mx.where(improved[None, :], y, best), mx.where(improved, candidate, score)
    mx.eval(best, score)
  if steps:
    certify()
  return dict(
    bounds=results,
    iterations=steps,
    seconds=monotonic() - start,
    device=mx.device_info()['device_name'],
  )
