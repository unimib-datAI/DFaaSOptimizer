# Centralized Feasibility Contract

## Goal

Every non-centralized algorithm must generate solutions that are feasible for
`LoadManagementModel`. The centralized optimum is therefore a valid upper bound
for every reported objective value.

Invalid solutions must fail immediately with a diagnostic naming the violated
constraint. This check is a safety and observability mechanism, not a repair or
projection step.

## Design

Add one canonical array-level validator for the constraints shared with
`LoadManagementModel`. It will check traffic conservation, direct-neighbor
offloading, no ping-pong, utilization bounds, and memory capacity using the same
tolerances as the solver-facing code.

All algorithms must invoke this validator before accepting a candidate as their
best centralized solution and before persisting it. Existing algorithm-specific
checks may delegate to the canonical validator instead of duplicating rules.
Validation failure raises an exception containing the constraint name and
indices involved.

The hierarchical auction must also prevent invalid moves by construction.
Higher-level requests may select a seller only when the concrete buyer and
seller nodes are direct neighbors. A request is excluded when it would make a
node both send and receive traffic for the same function. The final validator
remains active to catch regressions in this or any other algorithm.

## Non-goals

- Do not alter or clamp objective values.
- Do not project invalid solutions into the feasible region.
- Do not broaden the centralized model to support multi-hop offloading.
- Do not refactor unrelated auction or solver logic.

## Verification

Tests will first demonstrate the current failures: hierarchical allocation to a
non-neighbor, ping-pong creation, and acceptance of an invalid candidate. After
the change, focused unit tests must show that invalid moves are not generated
and that the validator reports each relevant constraint. Existing algorithm
tests and a small Gurobi comparison must confirm that every reported heuristic
objective is at most the centralized optimum within numerical tolerance.
