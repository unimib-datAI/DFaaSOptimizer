# VNS on Apple Metal

Run the same fixed-x VNS on the MacBook CPU and Apple GPU, preserving numerical
decisions at fixed iteration budgets. User requested continued implementation.

## Approach

Use optional MLX custom Metal kernels for integer candidate generation and
feasibility. Keep candidate gains, selection, archive and accepted state on the
CPU in float64, with the same operation and tie order as the reference. CPU
precomputes replica memory costs for integer processing values and insertion
spaces using the existing float64 formulas. No float32 approximation enters the
algorithm. Initialization remains the existing CPU DP.

A fully float32 backend could diverge; software double arithmetic would add
substantial complexity. The selected hybrid approach trades transfers and CPU
selection for reproducibility. It is not a fully GPU-resident implementation.

## Constraints

- Preserve CPU/CUDA behavior and optional imports; add `--backend metal`.
- Reject tables exceeding `max_device_bytes` before allocation; no fallback.
- Table costs saturate at RAM+1, since all costs are nonnegative.
- Check complete moves atomically, including repeated edges and exchanges.
- Preserve order of four float64 gain additions and existing tie breaking.
- Test actual GPU allocations, caches and archives against CPU each iteration.
- Benchmark cold initialization separately from warmed search; record fixed
  options, seeds, input hashes and equality, without modifying input data.
- Report actual performance even if the hybrid backend is slower.

## Scope

New metal.py and metal.metal, optional requirements, parity tests and a paired
benchmark script. Small changes to core.py, CLI and README. No dispatcher,
campaign, CUDA or existing DP changes.
