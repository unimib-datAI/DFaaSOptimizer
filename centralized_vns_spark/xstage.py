"""CUDA local-plan refinement with deterministic parallel best improvement.

The indexed neighborhood includes additions, replica-boundary drops and paired
reallocations. A transaction includes greedy receiver repairs. Unlike CPU RX,
``iterations`` counts candidate slots, including duplicate/infeasible slots.
"""
from dataclasses import replace
from functools import lru_cache
import math
import time

import numpy as np

from centralized_vns_r.xstage import local_gains


def _multiply(a, b):
  return float(a) * float(b)


def _add(a, b):
  return float(a) + float(b)


def _term_function(multiply, add):
  def term(index, kind, node, x, y, outgoing, load, alpha, gamma, src, fun, beta):
    if kind == 3:
      return multiply(beta[index], float(y[index])) / max(load[src[index], fun[index]], 1)
    g = index
    if kind != 0:
      node, g = index // x.shape[1], index % x.shape[1]
    local = multiply(alpha[node, g], float(x[node, g]))
    if kind == 0:
      return local / max(load[node, g], 1)
    rejected = load[node, g] - x[node, g]
    if kind == 2:
      rejected -= outgoing[node, g]
    return add(local, -multiply(gamma[node, g], float(rejected))) / max(load[node, g], 1)
  return term


def _sum_function(term, add, buffer_size=8192):
  def pairwise(count, kind, node, x, y, outgoing, load, alpha, gamma, src, fun,
               beta, lengths, partials):
    # Match NumPy's float64 pairwise reduction (128-element, eight-lane leaves).
    size = count
    count = min(size, buffer_size)
    start = depth = 0
    total = 0.0
    while True:
      while count > 128:
        lengths[depth] = count
        depth += 1
        count = (count // 2) // 8 * 8
      if count < 8:
        value = -0.0
        for index in range(start, start + count):
          value = add(value, term(index, kind, node, x, y, outgoing, load,
                                 alpha, gamma, src, fun, beta))
      else:
        for lane in range(8):
          partials[64 + lane] = term(start + lane, kind, node, x, y, outgoing,
                                    load, alpha, gamma, src, fun, beta)
        for offset in range(8, count - count % 8, 8):
          for lane in range(8):
            partials[64 + lane] = add(partials[64 + lane],
              term(start + offset + lane, kind, node, x, y, outgoing,
                   load, alpha, gamma, src, fun, beta))
        value = add(add(add(partials[64], partials[65]), add(partials[66], partials[67])),
                    add(add(partials[68], partials[69]), add(partials[70], partials[71])))
        for offset in range(count - count % 8, count):
          value = add(value, term(start + offset, kind, node, x, y, outgoing,
                                 load, alpha, gamma, src, fun, beta))
      start += count
      while depth > 0:
        parent = lengths[depth - 1]
        if parent > 0:
          partials[depth - 1] = value
          lengths[depth - 1] = -parent
          count = parent - (parent // 2) // 8 * 8
          break
        value = add(partials[depth - 1], value)
        depth -= 1
      else:
        total = add(total, value)
        if start >= size:
          return total
        count = min(size - start, buffer_size)
  return pairwise


def _numerical_functions(multiply, add, sum_terms):
  def room(x, incoming, capacity, memory, ram, node, fun, total_load):
    used = 0.0
    count = 0
    for other in range(x.shape[1]):
      processing = x[node, other] + incoming[node, other]
      replicas = max(1 if processing > 0 else 0,
                     math.ceil(add(processing / capacity[node, other], -1e-9)))
      used = add(used, multiply(float(replicas), float(memory[other])))
      if other == fun:
        count = replicas
    count += max(0, int(ram[node] - used)) // memory[fun]
    if count == 0:
      return 0
    limit = min(float(total_load), multiply(capacity[node, fun], add(float(count), 1e-9)))
    return max(0, int(math.floor(limit)) - x[node, fun] - incoming[node, fun])


  def check(x, y, incoming, outgoing, load, capacity, memory, ram,
             alpha, gamma, src, dst, fun, beta, floor, lengths, partials):
    for node in range(x.shape[0]):
      for g in range(x.shape[1]):
        incoming[node, g] = 0
        outgoing[node, g] = 0
    for edge in range(y.size):
      if y[edge] < 0:
        return -math.inf
      incoming[dst[edge], fun[edge]] += y[edge]
      outgoing[src[edge], fun[edge]] += y[edge]
    for node in range(x.shape[0]):
      used = 0.0
      for g in range(x.shape[1]):
        rejected = load[node, g] - x[node, g] - outgoing[node, g]
        if (x[node, g] < 0 or x[node, g] > load[node, g] or rejected < 0 or
            (incoming[node, g] > 0 and outgoing[node, g] > 0)):
          return -math.inf
        processing = x[node, g] + incoming[node, g]
        ratio = processing / capacity[node, g]
        if not math.isfinite(ratio) or ratio > 2**40:
          return -math.inf
        replicas = max(1 if processing > 0 else 0, math.ceil(add(ratio, -1e-9)))
        if replicas > 2**40:
          return -math.inf
        used = add(used, multiply(float(replicas), float(memory[g])))
      gain = sum_terms(x.shape[1], 0, node, x, y, outgoing, load, alpha, gamma,
                       src, fun, beta, lengths, partials)
      if used > ram[node] or not math.isfinite(gain) or gain < add(floor[node], -1e-12):
        return -math.inf
    base = sum_terms(x.size, 1, 0, x, y, outgoing, load, alpha, gamma,
                     src, fun, beta, lengths, partials)
    local = sum_terms(x.size, 2, 0, x, y, outgoing, load, alpha, gamma,
                      src, fun, beta, lengths, partials)
    forwarded = sum_terms(y.size, 3, 0, x, y, outgoing, load, alpha, gamma,
                          src, fun, beta, lengths, partials)
    value = add(local, forwarded)
    return value if math.isfinite(value) and math.isfinite(base) else -math.inf

  return room, check


def _proposal_function(room, check, multiply, add, sum_terms):
  # The same scalar transaction runs in Python tests or with CUDA device helpers.
  def proposal(index, x, y, tx, ty, incoming, outgoing, load, capacity, memory,
               ram, alpha, gamma, src, dst, fun, beta, reward, trim_order,
               repair_order, floor, total_load, lengths, partials):
    nf = x.shape[1]
    width = 5 + 9 * nf
    cell, kind = index // width, index % width
    node, g = cell // nf, cell % nf
    for i in range(x.shape[0]):
      for h in range(nf):
        tx[i, h] = x[i, h]
        incoming[i, h] = 0
        outgoing[i, h] = 0
    for edge in range(y.size):
      ty[edge] = y[edge]
      incoming[dst[edge], fun[edge]] += y[edge]
      outgoing[src[edge], fun[edge]] += y[edge]
    added_fun = -1
    if kind < 2:
      maximum = min(load[node, g] - x[node, g],
                    room(tx, incoming, capacity, memory, ram, node, g, total_load))
      amount = 1 if kind == 0 else maximum
      if amount <= 0 or amount > maximum:
        return -math.inf
      tx[node, g] += amount
      added_fun = g
    else:
      removal_kind = (kind - 2) // (1 + 3 * nf)
      option = (kind - 2) % (1 + 3 * nf)
      processing = x[node, g] + incoming[node, g]
      count = max(1 if processing > 0 else 0,
                  math.ceil(add(processing / capacity[node, g], -1e-9)))
      boundary = 0
      if count > 1:
        boundary = int(math.floor(min(float(processing),
                       multiply(capacity[node, g], add(float(count - 1), 1e-9)))))
      amount = 1
      if removal_kind == 1:
        amount = processing - boundary
      elif removal_kind == 2:
        amount = x[node, g]
      if amount <= 0 or amount > x[node, g]:
        return -math.inf
      tx[node, g] -= amount
      if option > 0:
        other, addition_kind = (option - 1) // 3, (option - 1) % 3
        if other == g:
          return -math.inf
        maximum = min(load[node, other] - x[node, other],
          room(tx, incoming, capacity, memory, ram, node, other, total_load))
        added = 1
        if addition_kind == 1:
          added = maximum
        elif addition_kind == 2:
          gain = sum_terms(nf, 0, node, x, y, outgoing, load, alpha, gamma,
                           src, fun, beta, lengths, partials)
          lost = multiply(float(amount), alpha[node, g]) / max(load[node, g], 1)
          deficit = max(0.0, add(add(add(floor[node], -gain), lost), -1e-12))
          unit = alpha[node, other] / max(load[node, other], 1)
          added = 0
          if unit > 0 and deficit <= multiply(unit, float(maximum)):
            added = int(math.ceil(deficit / unit))
        if added <= 0 or added > maximum:
          return -math.inf
        tx[node, other] += added
        added_fun = other
    if added_fun >= 0:
      excess = outgoing[node, added_fun] - (load[node, added_fun] - tx[node, added_fun])
      for position in range(trim_order.size):
        edge = trim_order[position]
        if excess <= 0:
          break
        if src[edge] == node and fun[edge] == added_fun:
          removed = min(excess, ty[edge])
          ty[edge] -= removed
          excess -= removed
    value = check(tx, ty, incoming, outgoing, load, capacity, memory, ram,
                  alpha, gamma, src, dst, fun, beta, floor, lengths, partials)
    if not math.isfinite(value):
      return -math.inf
    for position in range(repair_order.size):
      edge = repair_order[position]
      source, h = src[edge], fun[edge]
      if (dst[edge] != node or reward[edge] <= 0 or
          incoming[source, h] > 0 or outgoing[node, h] > 0):
        continue
      amount = min(load[source, h] - tx[source, h] - outgoing[source, h],
        room(tx, incoming, capacity, memory, ram, node, h, total_load))
      if amount <= 0:
        continue
      ty[edge] += amount
      incoming[node, h] += amount
      outgoing[source, h] += amount
      repaired = check(tx, ty, incoming, outgoing, load, capacity, memory, ram,
                       alpha, gamma, src, dst, fun, beta, floor, lengths, partials)
      if not math.isfinite(repaired):
        ty[edge] -= amount
        check(tx, ty, incoming, outgoing, load, capacity, memory, ram,
              alpha, gamma, src, dst, fun, beta, floor, lengths, partials)
    return check(tx, ty, incoming, outgoing, load, capacity, memory, ram,
                 alpha, gamma, src, dst, fun, beta, floor, lengths, partials)
  return proposal


@lru_cache(maxsize=1)
def kernels(buffer_size=8192):
  global cuda
  from numba import cuda as cuda_module
  cuda = cuda_module
  from numba import config
  from numba.cuda import libdevice
  device_options = dict(device=True, fastmath=False, opt=True)
  # Explicit RN intrinsics prevent NVVM contracting products into FMA.
  multiply = cuda.jit(**device_options)(_multiply) if config.ENABLE_CUDASIM else libdevice.dmul_rn
  add = cuda.jit(**device_options)(_add) if config.ENABLE_CUDASIM else libdevice.dadd_rn
  term = cuda.jit(**device_options)(_term_function(multiply, add))
  sum_terms = cuda.jit(**device_options)(_sum_function(term, add, buffer_size))
  scalar_room, scalar_check = _numerical_functions(multiply, add, sum_terms)
  room = cuda.jit(**device_options)(scalar_room)
  check = cuda.jit(**device_options)(scalar_check)
  proposal = cuda.jit(**device_options)(_proposal_function(room, check, multiply, add, sum_terms))

  @cuda.jit(fastmath=False, opt=True)
  def candidates(offset, count, x, y, tx, ty, incoming, outgoing, values,
                 load, capacity, memory, ram, alpha, gamma, src, dst, fun,
                 beta, reward, trim_order, repair_order, floor, total_load, lengths, partials):
    slot = cuda.grid(1)
    if slot < count:
      values[slot] = proposal(offset + slot, x, y, tx[:, :, slot], ty[:, slot],
        incoming[:, :, slot], outgoing[:, :, slot], load, capacity, memory, ram,
        alpha, gamma, src, dst, fun, beta, reward, trim_order, repair_order,
        floor, total_load, lengths[:, slot], partials[:, slot])

  @cuda.jit(fastmath=False, opt=True)
  def select(count, tx, ty, values, best_x, best_y, score):
    lane = cuda.threadIdx.x
    choices = cuda.shared.array(64, np.int64)
    gains = cuda.shared.array(64, np.float64)
    chosen = -1
    gain = -math.inf
    for slot in range(lane, count, cuda.blockDim.x):
      if values[slot] > gain:
        chosen, gain = slot, values[slot]
    choices[lane], gains[lane] = chosen, gain
    cuda.syncthreads()
    width = cuda.blockDim.x // 2
    while width > 0:
      if lane < width:
        other = choices[lane + width]
        value = gains[lane + width]
        if other >= 0 and (value > gains[lane] or
            (value == gains[lane] and other < choices[lane])):
          choices[lane], gains[lane] = other, value
      cuda.syncthreads()
      width //= 2
    # Strict ties retain the first candidate across both lanes and batches.
    chosen = choices[0]
    if chosen >= 0 and gains[0] > score[1] and gains[0] > add(score[0], 1e-12):
      for cell in range(lane, best_x.size, cuda.blockDim.x):
        i, g = cell // best_x.shape[1], cell % best_x.shape[1]
        best_x[i, g] = tx[i, g, chosen]
      for edge in range(lane, best_y.size, cuda.blockDim.x):
        best_y[edge] = ty[edge, chosen]
      cuda.syncthreads()
      if lane == 0:
        score[1] = gains[0]

  @cuda.jit(fastmath=False, opt=True)
  def apply(x, y, best_x, best_y, score, accepted):
    lane = cuda.threadIdx.x
    if score[1] > add(score[0], 1e-12):
      for cell in range(lane, x.size, cuda.blockDim.x):
        i, g = cell // x.shape[1], cell % x.shape[1]
        x[i, g] = best_x[i, g]
      for edge in range(lane, y.size, cuda.blockDim.x):
        y[edge] = best_y[edge]
      cuda.syncthreads()
      if lane == 0:
        score[0] = score[1]
        accepted[0] += 1
  return cuda, candidates, select, apply


def validate_x_options(iterations, time_limit, seed, max_device_bytes):
  """Validate standalone/CLI settings before loading inputs or creating outputs."""
  if type(iterations) is not int or iterations < 0:
    raise ValueError('iterations must be a nonnegative integer')
  if type(seed) is not int or not 0 <= seed < 2147483646:
    raise ValueError('seed must be in [0, 2147483646)')
  if (isinstance(time_limit, bool) or not isinstance(time_limit, (int, float)) or
      not math.isfinite(time_limit) or time_limit < 0):
    raise ValueError('time_limit must be nonnegative and finite')
  if type(max_device_bytes) is not int or max_device_bytes < 1:
    raise ValueError('max_device_bytes must be a positive integer')


def _candidate_batch(iterations, proposals, fixed_bytes, per_candidate,
                     max_device_bytes, multiprocessors=0):
  if not iterations:
    return 0
  if fixed_bytes + per_candidate > max_device_bytes:
    raise ValueError(f'CUDA x-stage memory budget exceeded: at least '
                     f'{fixed_bytes + per_candidate} bytes')
  return min(max(4096, multiprocessors * 512), iterations, proposals,
             (max_device_bytes - fixed_bytes) // per_candidate)


def refine_x(p, y, *, iterations=1000, time_limit=15., seed=4850,
             max_device_bytes=512*1024**2):
  """Refine on CUDA; wall deadlines are polled between bounded candidate batches.

  ``seed`` is validated for interchange compatibility; deterministic enumeration
  does not use it. Final physics and original local floors are checked on CPU.
  Batches target eight 64-thread blocks per hardware multiprocessor (at least
  4096 candidates), bounded by the remaining work and configured memory budget.
  """
  validate_x_options(iterations, time_limit, seed, max_device_bytes)
  start = time.perf_counter()
  p = replace(p)
  checked = p.check(y)
  if not math.isfinite(checked['welfare']):
    raise ValueError('incumbent welfare must be finite')
  x, y = p.x.copy(), np.asarray(y, dtype=np.int64).copy()
  floor = local_gains(p, x)
  initial = checked['welfare']
  trace = [dict(iteration=0, elapsed_s=time.perf_counter()-start, best_welfare=initial)]
  attempted = accepted = 0
  n, f = p.load.shape
  edges = len(p.src)
  proposals = n * f * (5 + 9 * f)
  # Arrays: immutable inputs, two plans/flows, floors/scores/counters, batch scratch.
  fixed_bytes = sum(a.nbytes for a in (p.load, p.capacity, p.memory, p.ram,
    p.alpha, p.gamma, p.src, p.dst, p.fun, p.beta, p.reward))
  fixed_bytes += 8 * (2 * n * f + 4 * edges + n + 3)
  # A 64-frame reduction stack covers every supported signed-int64 array size.
  per_candidate = 8 * (3 * n * f + edges + 1 + 64 + 72)
  batch = _candidate_batch(iterations, proposals, fixed_bytes, per_candidate,
                           max_device_bytes)
  device_bytes = fixed_bytes + batch * per_candidate if batch else 0
  reason = 'iteration_limit'
  if iterations and time.perf_counter() - start < time_limit:
    cuda_module, candidate_kernel, select_kernel, apply_kernel = kernels(np.getbufsize())
    if not cuda_module.is_available():
      raise RuntimeError('CUDA unavailable; CUDA x refinement has no CPU fallback')
    from numba import config
    multiprocessors = 0 if config.ENABLE_CUDASIM else int(
      cuda_module.get_current_device().MULTIPROCESSOR_COUNT)
    batch = _candidate_batch(iterations, proposals, fixed_bytes, per_candidate,
                             max_device_bytes, multiprocessors)
    device_bytes = fixed_bytes + batch * per_candidate
    transfer = cuda_module.to_device
    data = tuple(transfer(a) for a in (p.load, p.capacity, p.memory, p.ram,
      p.alpha, p.gamma, p.src, p.dst, p.fun, p.beta, p.reward,
      np.argsort(p.reward, kind='stable').astype(np.int64),
      np.argsort(-p.reward, kind='stable').astype(np.int64), floor))
    dx, dy, best_x, best_y = (transfer(a) for a in (x, y, x, y))
    # Neighboring candidate threads access adjacent scratch cells.
    tx = cuda_module.device_array((n, f, batch), dtype=np.int64)
    ty = cuda_module.device_array((edges, batch), dtype=np.int64)
    incoming = cuda_module.device_array((n, f, batch), dtype=np.int64)
    outgoing = cuda_module.device_array((n, f, batch), dtype=np.int64)
    values = cuda_module.device_array(batch, dtype=np.float64)
    lengths = cuda_module.device_array((64, batch), dtype=np.int64)
    partials = cuda_module.device_array((72, batch), dtype=np.float64)
    score = transfer(np.array([initial, initial], dtype=np.float64))
    accepted_device = transfer(np.zeros(1, dtype=np.int64))
    total_load = int(p.load.sum())
    while attempted < iterations:
      previous = accepted
      offset = 0
      while offset < proposals and attempted < iterations:
        if time.perf_counter() - start >= time_limit:
          reason = 'time_limit'
          break
        count = min(batch, proposals - offset, iterations - attempted)
        candidate_kernel[(count + 63)//64, 64](offset, count, dx, dy, tx, ty,
          incoming, outgoing, values, *data, total_load, lengths, partials)
        select_kernel[1, 64](count, tx, ty, values, best_x, best_y, score)
        cuda_module.synchronize()
        offset += count
        attempted += count
      apply_kernel[1, 64](dx, dy, best_x, best_y, score, accepted_device)
      accepted = int(accepted_device.copy_to_host()[0])
      if accepted > previous:
        trace.append(dict(iteration=attempted, elapsed_s=time.perf_counter()-start,
                          best_welfare=float(score.copy_to_host()[0])))
      if reason == 'time_limit':
        break
      if accepted == previous:
        reason = 'local_optimum' if offset == proposals else 'iteration_limit'
        break
    x, y = dx.copy_to_host(), dy.copy_to_host()
  elif iterations:
    reason = 'time_limit'
  checked = replace(p, x=x).check(y)
  gains = local_gains(p, x)
  if (not math.isfinite(checked['welfare']) or checked['welfare'] < initial - 1e-12 or
      np.any(gains < floor - 1e-12)):
    raise RuntimeError('CUDA x-stage failed independent final welfare/floor check')
  return dict(algorithm='VNS-RX', backend='cuda', x=x, y=y, **checked,
    local_gain_floor=floor, local_gain=gains, attempted=attempted, accepted=accepted,
    trace=trace, initial_welfare=initial, elapsed_s=time.perf_counter()-start,
    termination_reason=reason, execution=dict(variant='parallel-best-improvement',
      candidate_batch=batch, device_bytes=device_bytes, seed_used=False,
      iteration_unit='candidate slots including duplicates/infeasible; repairs are atomic',
      deadline_polling='between candidate batches; one batch and final check may overrun'))
