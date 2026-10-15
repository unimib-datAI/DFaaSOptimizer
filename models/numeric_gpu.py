"""Lazy float64 CUDA kernels for shared numerical phases; no device state cache."""
import numpy as np

_residual_kernel = None


def _residual():
  global _residual_kernel, cuda
  if _residual_kernel is None:
    from numba import cuda
    from numba import float64, int64
    from numba.cuda import libdevice

    @cuda.jit(device=True)
    def source_sum(y, n, f):
      # Match NumPy's float64 pairwise sum (PW_BLOCKSIZE=128), including
      # its eight accumulators and balanced splits. No unordered atomics.
      starts = cuda.local.array(32, int64)
      sizes = cuda.local.array(32, int64)
      lefts = cuda.local.array(32, float64)
      states = cuda.local.array(32, int64)
      begin, size, depth = 0, y.shape[0], 0
      while True:
        if size > 128:
          left = size // 2
          left -= left % 8
          starts[depth], sizes[depth], states[depth] = begin + left, size - left, 0
          depth += 1
          size = left
          continue
        if size < 8:
          value = -0.
          for i in range(size):
            value = libdevice.dadd_rn(value, y[begin+i,n,f])
        else:
          a=y[begin,n,f]; b=y[begin+1,n,f]; c=y[begin+2,n,f]; d=y[begin+3,n,f]
          e=y[begin+4,n,f]; g=y[begin+5,n,f]; h=y[begin+6,n,f]; j=y[begin+7,n,f]
          end = size - size % 8
          for i in range(8, end, 8):
            a=libdevice.dadd_rn(a,y[begin+i,n,f]); b=libdevice.dadd_rn(b,y[begin+i+1,n,f])
            c=libdevice.dadd_rn(c,y[begin+i+2,n,f]); d=libdevice.dadd_rn(d,y[begin+i+3,n,f])
            e=libdevice.dadd_rn(e,y[begin+i+4,n,f]); g=libdevice.dadd_rn(g,y[begin+i+5,n,f])
            h=libdevice.dadd_rn(h,y[begin+i+6,n,f]); j=libdevice.dadd_rn(j,y[begin+i+7,n,f])
          value = libdevice.dadd_rn(
            libdevice.dadd_rn(libdevice.dadd_rn(a,b),libdevice.dadd_rn(c,d)),
            libdevice.dadd_rn(libdevice.dadd_rn(e,g),libdevice.dadd_rn(h,j)))
          for i in range(end,size):
            value=libdevice.dadd_rn(value,y[begin+i,n,f])
        go_right = False
        while depth:
          top = depth - 1
          if states[top] == 0:
            lefts[top], states[top] = value, 1
            begin, size = starts[top], sizes[top]
            go_right = True
            break
          value = libdevice.dadd_rn(lefts[top], value)
          depth -= 1
        if not go_right:
          return value

    @cuda.jit
    def kernel(x, y, r, demand, utilization, cap, residual, ell):
      k = cuda.grid(1)
      nf = x.shape[1]
      n, f = k // nf, k % nf
      if n < x.shape[0]:
        incoming = source_sum(y,n,f)
        load = libdevice.dadd_rn(x[n,f], incoming)
        capacity = libdevice.dmul_rn(r[n,f], libdevice.ddiv_rn(utilization[f], demand[n,f]))
        cap[n,f] = capacity
        ell[n,f] = load
        residual[n,f] = max(0., libdevice.dadd_rn(capacity, -load))
    _residual_kernel = kernel
  return _residual_kernel


def residual_capacity_cuda(x, y, r, demand, utilization):
  """Synchronous full phase, with source reduction in the CPU loop order."""
  from utils import step_time
  step_time.mark_shared_device()
  kernel = _residual()
  arrays = [np.array(a, dtype=np.float64, order='C', copy=True) for a in (x,y,r,demand,utilization)]
  device = [cuda.to_device(a) for a in arrays]
  outputs = [cuda.device_array(arrays[0].shape, dtype=np.float64) for _ in range(3)]
  size = arrays[0].size
  if size:
    kernel[(size+127)//128,128](*device,*outputs)
  result = tuple(a.copy_to_host() for a in outputs)
  cuda.synchronize()
  from utils.compute_backend import record_operation
  record_operation('residual_capacity','cuda',arrays[0].shape[0],shape=list(arrays[0].shape),
                   device_bytes=sum(a.nbytes for a in arrays)+3*arrays[0].nbytes)
  return result
