# Hierarchical MADeA Design

## Objective

Add a new hierarchical algorithm whose level-one behavior is the production
FaaS-MADeA auction in `run_faasmadea.py`. The existing `hierarchical` method is
retained unchanged as a legacy implementation. The new method is exposed as
`hierarchical-madea` and writes the result label `HierarchicalMADeA`.

The implementation may import existing functions but must not modify the
FaaS-MADeA functions or the existing hierarchical runner. This preserves both
baselines and makes the scientific comparison explicit.

## Architecture

Create `hierarchical_auction/madea_runner.py`. Its initialization, temporal
loop, persistence, and centralized solution construction follow the current
hierarchical runner. Its level-one iteration follows `run_faasmadea.run`:

1. compute residual capacity;
2. call MADeA `define_bids`, including stagnation-driven memory bids;
3. call MADeA `evaluate_bids`, including replacement and tentative replicas;
4. update cumulative offloading, replicas, fairness, and residual demand;
5. validate fixed offloading and solve LSPr.

After level one, recompute residual capacity from the updated state and call
`HierarchicalAuctionEngine.run_higher_levels`. Accepted higher-level allocations
are mapped into the same cumulative `y` matrix. If they change `y`, validate it
and solve LSPr again. Termination uses MADeA's stopping function with keyword
arguments and treats accepted hierarchical allocations as progress.

No imported MADeA or legacy hierarchical function is edited. Small new helpers
in the new runner may normalize the level-one `eta`, compute offloaded demand,
and preserve the loop when higher levels make progress.

## Integration

Register `hierarchical-madea` in `run.py`, remote job construction, and result
metadata. The module command is:

```bash
python -m hierarchical_auction.madea_runner -c config.json -j 0
```

Paper suite defaults use `hierarchical-madea` as the proposed model. Explicit
requests for the legacy `hierarchical` algorithm continue to work. The smoke
suite contains both so wiring regressions remain visible.

## Validation

Tests verify that the new runner imports the production MADeA helpers, passes
the correct signatures and options, invokes higher levels only after level one,
is accepted by local and remote dispatch, and is selected by paper defaults.
A Gurobi-backed planar smoke test verifies complete artifacts and centralized
feasibility. The full test suite protects all existing algorithms.
