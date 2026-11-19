# VNS-Tree-BatchBounds on Apple Silicon

This standalone regional-bound stage reads a saved fixed-allocation VNS-Tree run. It checks the instance hashes, edge order, incumbent and covering prefixes, then selects up to 64 open leaves at evenly spaced positions in the saved frontier. Both methods receive the same column bounds, pair cuts, incumbent, time budget and root LP solution. HiGHS also reuses the root simplex basis. The stage writes a new JSON report.

Run from the optimizer directory in the Mac environment with highspy, SciPy and Metal-enabled MLX. For a five-minute budget per method use:

```sh
python -m centralized_vns_batch_bounds INSTANCE TREE_RESULT_DIRECTORY \
  --regions 64 --seconds 300 --iterations 1000000 --gap 0.001 \
  --output bounds-mac.json
```

HiGHS solves each regional LP to optimality when time permits. The Metal backend applies diagonally preconditioned primal–dual hybrid gradient to all the regional LPs together. Each sparse matrix is stored once, and each column of the state tensors represents one region. It starts from the root solution, clips primal variables to the regional boxes, and retains the lowest estimated dual objective seen for each region. The iteration limit is a second stopping condition for the GPU. The common LP build, root warm start and archive validation are timed separately. Per-method elapsed time includes verification.

## How a regional upper bound is accepted

For the maximization relaxation with rows `Ax <= b`, objective `base + c·x` and box `l <= x <= u`, every nonnegative row-price vector `y` gives the bound

```text
base + b·y + sum_j max(l_j * (c - Aᵀy)_j, u_j * (c - Aᵀy)_j)
```

Metal proposes prices in float32. The CPU checks the selected prices against the original binary64 LP data using exact dyadic arithmetic and rounds the upper bound towards positive infinity. HiGHS prices go through the same check. This certifies the numerical LP supplied to both solvers. The stage adopts the archived integer model, valid cuts and inherited bounds as inputs.

The report lists selected leaves in the same order as each method's raw LP bounds. A null value means the method produced no certificate within budget. For each selected leaf the effective bound is the smaller of the inherited and new bounds. Other leaves retain their archived values. The largest effective bound over the entire frontier determines the reported global gap on the coordinator objective. No DP allocation, forwarding flow or incumbent is changed by this stage.
