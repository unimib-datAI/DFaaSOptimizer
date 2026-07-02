# FaaS-MALD Findings Remediation Design

**Date:** 2026-07-02

**Status:** approved design, pending implementation plan

## Goal

Correct the semantic and lifecycle defects found in the first FaaS-MALD
implementation, then remove MALD-specific dead paths and dense-graph overhead
without refactoring the existing coordinators.

## Scope

The work modifies only the MALD module, its tests, and its technical note:

- `decentralized_dual.py`
- `tests/test_dual_helpers.py`
- `tests/test_dual_e2e.py`
- `faas-mald-note/faas-mald.tex`
- `faas-mald-note/README.md`

Existing coordinators and shared helpers remain unchanged. Extracting a common
runner from MADiG, MABR, MAPoD, and MALD is explicitly out of scope.

## Phase 1: Correctness

### Cloud-relative coordination reward

The existing coordinators prefer horizontal offloading whenever its score is
better than Cloud rejection:

```text
s_ijf > -gamma_if
```

MALD currently applies that eligibility filter but later requires the raw
price-adjusted score to be positive. It therefore rejects negative horizontal
scores even when they are better than Cloud rejection.

MALD will optimize the incremental advantage over Cloud:

```text
a_ijf = s_ijf + gamma_if
```

The Cloud is the zero baseline. A pair participates when `a_ijf > 0`, which is
algebraically equivalent to `s_ijf > -gamma_if`. Seller prices are subtracted
from `a`, not from the raw score. The primal LB and dual UB use the same
Cloud-relative reward, so the certificate remains internally consistent.

The bid `utility` used by `evaluate_assignments` must also be the
Cloud-relative reward. This preserves the same buyer ordering as raw `s` for a
fixed `(i, f)` because `gamma_if` is constant across that buyer's sellers.

### Replica lifecycle and stopping

The runner currently computes `blackboard`, may create replicas through
`start_additional_replicas`, and then passes the old blackboard to
`check_stopping_criteria`. A globally empty old blackboard causes an immediate
`no capacity left` stop even when replicas have just created new capacity.

After any positive replica increment, MALD will recompute capacity, residual
capacity, effective load, and blackboard before evaluating stopping criteria.
The next outer iteration can then use the newly available capacity.

### Certificate identity and observability

`best_ub` can be obtained at an earlier price vector than the final `lam`.
Whenever `best_ub` improves, the round will retain a copy as `best_lam`.
`gap_info` will expose `best_lam`; it may optionally expose `final_lam` for
diagnostics, but a generic `lam` field must not ambiguously refer to one while
the bound refers to the other.

The certificate applies only to one fixed-residual-capacity inner
transportation LP. It does not certify the outer MALD solution, replica
decisions, or the LSPr re-optimization. The runner will write
`coordination_certificate.csv` with one row per outer coordination iteration:

- `timestep`
- `outer_iteration`
- `LB`
- `UB`
- `gap`
- `inner_iterations`
- `stop_reason`

`termination_condition.csv` remains the final per-timestep summary and labels
the reported gap as the final inner fixed-capacity LP gap.

### Configuration validation

Before starting the round, MALD validates:

- `max_inner_iterations` is an integer greater than or equal to one;
- `step_rule` is `sqrt` or `polyak`;
- `alpha0` is finite and strictly positive for `sqrt`;
- `theta` is finite and strictly positive for `polyak`;
- `gap_tolerance` is finite and non-negative;
- latency and fairness weights are finite.

Invalid settings raise `ValueError` with the option name in the message.

## Phase 2: Simplification and Sparse-Graph Performance

### Sparse neighbor traversal

`pair_scores` will iterate only over nonzero neighbors instead of all node
pairs. `buyer_price_response` will rank only eligible seller indices rather
than constructing a dense length-`Nn` adjusted-score vector for every active
buyer/function pair.

The resulting score and response work is proportional to eligible graph edges
plus ranking cost, rather than unconditionally `O(Nn^2 * Nf)`.

### Retain the best primal assignment

When a candidate improves the LB, the round will retain both `best_y` and
`best_bids`. Post-loop placed-demand calculation and the returned increment use
`best_y` directly. This removes duplicate calls to `evaluate_assignments` and
makes the identity between returned assignment and reported LB structural.

### Remove unreachable replica paths

MALD will have one replica-start mechanism: memory bids handled by the outer
runner. `dual_coordination_round` will no longer request tentative replica
starts or return `additional_replicas`. Its internal return becomes:

```text
(y_increment, memory_bids, gap_info, n_active)
```

If recovered demand is short, memory bids are produced for neighbors with
sufficient memory. The outer runner converts those bids into replica increments
and recomputes capacity before stopping.

### Remove unreachable forced-memory state

MALD increments are non-negative and its accepted-load total increases whenever
allocation changes. The copied equal-history stagnation condition cannot
normally become true. MALD will remove `force_memory_bids`, its accepted-load
deque, and the corresponding parameter from `dual_coordination_round`.

This change is MALD-only. Other coordinators keep their existing reassignment
and forced-memory behavior.

## Error Handling and Compatibility

The public `run(config, parallelism, log_on_file=False,
disable_plotting=False)` API and CLI remain unchanged. Existing configuration
files without a `solver_options.dual` section continue to use defaults.

The internal `dual_coordination_round` signature changes only in MALD and its
tests. No compatibility shim is required because GitNexus reports only the MALD
runner and MALD tests as callers.

## Testing Strategy

Development follows TDD. New regression coverage includes:

1. A negative horizontal score that is still better than Cloud is selected.
2. Cloud-relative LB/UB bracket a SciPy LP oracle built with `s + gamma`.
3. `best_lam` reproduces the retained UB.
4. Invalid numeric dual options raise clear `ValueError` exceptions.
5. New replicas created from an initially empty blackboard prevent premature
   stopping and are usable in the following iteration.
6. The round returns stored `best_y` without repeated evaluator calls.
7. Sparse graphs evaluate only neighbor pairs.
8. `coordination_certificate.csv` contains one row per outer iteration with the
   fixed-capacity scope fields.
9. Existing helper, E2E, sibling, and full repository suites remain green.

The E2E test continues to run with Gurobi when available. Pure runner lifecycle
logic introduced by this remediation must also have a focused test that does
not silently disappear when Gurobi is unavailable.

## Documentation

The LaTeX note and README will be updated to:

- formulate the LP with Cloud-relative advantage `a = s + gamma`;
- use the same quantity in buyer response, LB, and UB;
- describe `best_lam` and the per-outer-iteration certificate CSV;
- state explicitly that the certificate covers a fixed-capacity inner LP;
- remove tentative-replica and forced-memory pseudocode;
- state sparse traversal and ranking complexity matching the shipped code;
- distinguish the practical gap-based `polyak` rule from classical Polyak
  convergence theory.

The standalone LaTeX preview must compile without undefined references,
undefined citations, or overfull boxes, and all rendered pages must pass visual
inspection.

## Acceptance Criteria

- MALD makes the same node-versus-Cloud decision as the sibling coordinators.
- Replicas created from memory bids can affect the next outer iteration.
- Every certificate row identifies exactly one fixed-capacity inner LP.
- The price vector associated with the best UB is reported unambiguously.
- No MALD-only tentative-replica or forced-memory dead path remains.
- Sparse graphs are processed through neighbor/eligible indices.
- Public runner and CLI interfaces remain compatible.
- All MALD, sibling, and repository tests pass.
