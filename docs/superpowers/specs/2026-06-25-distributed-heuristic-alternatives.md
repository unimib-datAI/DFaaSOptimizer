# Distributed coordination heuristics — alternatives considered

**Date:** 2026-06-25
**Status:** Reference / future work

This note records the distributed-coordination heuristics considered as
alternatives to the auction-based methods (FaaS-MADeA, HierarchicalAuction) for
the DiFRALB problem. The chosen method, **FaaS-MADiG** (greedy proportional
diffusion), is specified in
`2026-06-25-faas-madig-distributed-heuristic-design.md`. The two options below
were deliberately deferred.

All options share the same frame as the auctions: each node solves the local
`LSP`/`LSPr` MILP to fix replicas `r`, local load `x`, and residual load
`omega`; only the inter-node coordination (the replacement for `define_bids` +
`evaluate_bids` + price update) differs.

## Option B — Power-of-d-choices (randomized, lightweight)

**Mechanism.** For each function `f`, every overloaded node samples `d` random
neighbors (e.g. `d = 2`), queries their residual capacity `blackboard[j,f]`, and
forwards its residual load to the best of the sampled neighbors (most residual
capacity, or highest `score`). No global preference ranking; minimal
communication and state.

**Pros.**
- Models realistic stateless gossip / randomized load balancing.
- Very low computational and communication cost; trivially distributed.

**Cons.**
- Stochastic → requires averaging over multiple seeds for fair reporting.
- Typically sub-optimal; weakly tied to the objective weights `beta`.
- Harder to compare head-to-head with the deterministic auctions.

**When to revisit.** If we want to position FaaS-MADiG on a spectrum
*market → greedy → random*, B is the natural lightweight endpoint and a cheap
add-on once `define_assignments`/`evaluate_assignments` exist.

## Option C — Distributed best-response (Gauss-Seidel relaxation)

**Mechanism.** Treat neighbor residual capacity as a shared resource. In each
round, nodes (in a fixed order) claim capacity by computing their best response
to the current claims — i.e. re-optimize their local marginal cost given what
neighbors have taken — iterating toward a fixed point / equilibrium.

**Pros.**
- More "intelligent"; tends toward an equilibrium allocation.
- Can yield higher-quality solutions than plain greedy.

**Cons.**
- Heavier per iteration (repeated local re-optimization).
- Conceptually close to the auction and to the FaaS-MACrO decomposition →
  weak contrast; risks blurring the comparison the study aims to make.

**When to revisit.** If FaaS-MADiG proves too myopic and we need a stronger
solver-light competitor that still avoids an explicit market.

## Decision rationale (why A / FaaS-MADiG was chosen)

- **Cleanest ablation:** identical information and local plan as the auctions,
  differing *only* by removing the price signal → directly answers whether the
  market mechanism is worth its cost.
- **Maximal code reuse / lowest risk:** reuses `compute_residual_capacity`,
  `start_additional_replicas`, the restricted-problem re-solve, fairness, the
  objective, and all stopping criteria unchanged.
- **Deterministic / reproducible:** no seed averaging required.
