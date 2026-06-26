# Literature-Comparison Audit — `\subsection{Positioning with respect to the literature}`

Audited file: `faas-madig-note/faas-madig.tex` (subsection `sec:madig-related`).
Ground-truth algorithm: `decentralized_diffusion.py` (`define_assignments`, `evaluate_assignments`).
Evidence: downloaded PDFs (`bertsekas1988`, `mitzenmacher2001`, `jiang2018`, `nezami2021`) read directly via `mutool`; paywalled works (`cybenko1989`, `willebeek1993`, `leechoi2021`) assessed from abstracts + verification reports.
Date: 2026-06-26.

## Verdict table

| Citation | Note's claim (short) | Verdict | Evidence | Recommended fix |
|---|---|---|---|---|
| `cybenko1989` | "Diffusion paradigm: an overloaded node sends excess load to its less-loaded neighbours." | **Correct** | Cybenko studies diffusion schemes for dynamic load balancing on message-passing networks; nodes iteratively redistribute excess load to less-loaded neighbours (abstract; verification-classics §1). Accurate, textbook characterization. | None. (Optional: Cybenko's diffusion is *iterative to equilibrium*, not single-round; the note attributes "single-round" to Willebeek, not Cybenko, so no conflict.) |
| `willebeek1993` | "Sender-initiated, single-round diffusion is one of the dynamic strategies catalogued." | **Correct** (minor wording) | Paper compares five DLB strategies including **sender-initiated diffusion (SID)** and receiver-initiated diffusion (RID) (abstract; verification-classics §2). The "catalogues sender-initiated diffusion" claim is solid. | "single-round" is imprecise: SID is an iterative near-neighbour scheme, not single-round. Either drop "single-round" or attribute the single-round simplification to FaaS-MADiG itself ("the sender-initiated form we use, run as a single coordination round, is a simplification of the SID strategy catalogued by..."). |
| `nezami2021` | Underlies "serve locally first, then greedily offload residual to one-hop neighbours" / decentralized service placement & LB for IoT. | **Overstated / partly misattributed** | EPOS Fog's core mechanism is **I-EPOS collective learning** (agents generate candidate *plans*, then cooperatively/iteratively select one maximizing edge utilization) — not greedy residual offloading. The greedy step is only intra-plan generation, and it "**randomly chooses** the required number of available neighbouring nodes as candidate hosts" (nezami.txt L1283–1287), i.e. it does *not* scan the full one-hop set. It is decentralized edge-to-cloud LB for IoT (true), but the "greedy one-hop residual offload" control structure is not its mechanism. | Soften: cite as an example of *decentralized multi-agent edge LB* (true) but do not imply its mechanism is greedy one-hop residual offloading. Move out of the "serve-locally-then-greedily-offload" list, or qualify "uses a greedy candidate-host heuristic within a collective-learning framework." Note also it contradicts the later "full-neighbourhood deterministic scan" contrast (Nezami samples randomly). |
| `jiang2018` | Underlies "serve locally first, then greedily offload residual to one-hop neighbours" / greedy energy–delay–cost offloading in imbalanced edge–cloud. | **Wrong / misattributed** | System model is **mobile users (MUs) offloading tasks to *shared* edge-cloud servers (ECSs) through wireless access points (APs)** (jiang.txt L398–414); "all these computing-intensive tasks **must be offloaded**" (L412). It is a user→AP→ECS bipartite assignment solved by centralized greedy (CGA/MGA), a many-to-one **matching game** (ADMA), and a fairness greedy (FGA). There is **no** local-execution-first step and **no** peer-to-peer one-hop neighbour diffusion. | Remove `jiang2018` from the "serve-locally-first / one-hop neighbour offloading" claim. If kept at all, cite narrowly as "greedy heuristics for energy–delay–cost task offloading," without implying one-hop peer diffusion or serve-local-first. |
| `leechoi2021` | "Greedy load balancing tailored to FaaS platforms." | **Correct** | GRAF is a greedy LB algorithm specifically for FaaS, maximizing locality/cache-hit ratio while preserving load balance (verification-modern §3). The note's loose phrasing "greedy load balancing tailored to FaaS platforms" is accurate. | None required. (Caveat for honesty: it is a *dispatcher-level, cache-locality* greedy scheduler, not one-hop peer offloading; the note's wording is general enough to remain correct, but do not lean on it as evidence of the one-hop-diffusion structure.) |
| `bertsekas1988` | "Fixing all prices to zero in a Bertsekas-style auction reduces the bidding-and-clearing mechanism to a greedy (deferred-acceptance) assignment." | **Overstated** | See dedicated section below. The reduction is *directionally* defensible (greedy is a degenerate/single-round special case) but the phrasing conflates "prices = 0", "ε = 0", and "single round," and "deferred-acceptance" is the wrong term. | Reword — see below. |
| `mitzenmacher2001` | "Power-of-d-choices probes a random sample of d candidates," contrasted with FaaS-MADiG's deterministic full one-hop scan. | **Correct** | Supermarket model: each arrival samples d ≥ 2 servers **uniformly at random** and joins the shortest queue (abstract; verification-classics §3). "Probes a random sample of d candidates" is exact, and the determinism/full-set contrast with FaaS-MADiG is fair. | None. |

## Bertsekas "auction → greedy" reduction — detailed audit

**Note's sentence:** "fixing all prices to zero in a Bertsekas-style auction reduces the bidding-and-clearing mechanism to a greedy (deferred-acceptance) assignment."

**What Bertsekas's auction actually does (paper §2, eqs. 5–8):**
- Each unassigned person `i` computes object values `v_ij = a_ij − p_j` (eq. 5), picks best `j* = argmax_j (a_ij − p_j)`, and **bids** `b_ij* = p_j* + v_ij* − w_ij* + ε` where `w_ij*` is the second-best value (eq. 7).
- Each object goes to the highest bidder and **raises its price** (eq. 8).
- The algorithm is **iterative**: it runs multiple bidding/assignment rounds, **prices rise monotonically**, and **objects are re-awarded** across rounds (a person can lose an object it briefly held).
- The whole construction is driven by **ε-complementary slackness (ε-CS)** (eq. 4), and the theory needs **ε > 0** strictly; with ε = 0 the bid increment vanishes and the convergence/termination argument no longer applies.

**Why the note's phrasing is imprecise / overstated:**
1. **It is not "prices = 0" that yields greedy; it is "no price *update*" + a single round.** Setting `p_j ≡ 0` only makes the score equal `a_ij` (the raw utility). The auction would still iterate and re-award objects — that is *not* greedy. What actually reduces the mechanism to greedy is **freezing prices (no eq.-8 update) and disallowing re-assignment of already-served load**, i.e. running essentially one pass with each buyer grabbing its best available residual capacity. The code confirms this: `evaluate_assignments` has "no min_b tracking, no price update, no last_y replacement swap" — it is the *removal of the price-update/reassignment*, not zeroing a price level, that produces the greedy rule.
2. **"deferred-acceptance" is the wrong term.** Deferred acceptance (Gale–Shapley) is precisely characterized by *tentative* holds that can be *bumped* by later, more-preferred proposers — i.e. re-assignment. FaaS-MADiG explicitly **removes** reassignment ("no reassignment of previously served load," Alg. seller line 17; code: no `last_y` swap). So the resulting rule is **immediate-acceptance / first-come greedy**, the *opposite* of deferred acceptance. (Ironically, the *full* auction, with its price-driven re-awarding, is the more DA-like object.)
3. **ε is silently dropped.** "Fixing all prices to zero" ignores that the auction is parameterized by ε > 0 (the minimum bid increment / ε-CS slack). A faithful statement should mention that *both* the price feedback *and* the ε bidding increment are removed.

**Verdict:** Overstated. The underlying intuition — *greedy assignment is the degenerate, price-frozen, single-round special case of the auction* — is sound and is even stated in the verification report ("Setting all prices to zero and running a single bidding round reduces the auction to a greedy assignment"). But the published sentence (a) mislocates the reduction in "prices = 0," (b) uses "deferred-acceptance," which is technically false for a rule that forbids reassignment, and (c) omits the single-round / no-price-update condition that actually does the work.

**Recommended phrasing:**

> "FaaS-MADiG is the price-frozen special case of a Bertsekas-style auction~\citep{bertsekas1988}: when the cross-iteration price update (and the associated $\varepsilon$ bid increment and $\varepsilon$-complementary-slackness mechanism) is removed, the bidding-and-clearing cycle collapses to a single-pass greedy assignment in which each buyer claims its best available residual capacity and no previously served load is re-awarded. In this sense FaaS-MADiG *is* FaaS-MADeA with the market layer removed."

(If a matching-theory label is wanted, the correct one is **immediate-acceptance / serial-dictatorship-style greedy**, *not* deferred acceptance — so it is safer to drop the parenthetical term entirely.)

## FaaS-MADiG self-description vs `decentralized_diffusion.py`

The note's "Differences" and algorithm prose were checked line-by-line against the code. All three differentiators are **supported**:

| Note's claim | Code evidence | Status |
|---|---|---|
| (i) Determinism over the **full** one-hop neighbourhood; deterministic tie-break on buyer index. | `define_assignments` builds `candidate_sellers` from **all** capacity-bearing neighbours (`potential_capacity_sellers`, L65–82), sorts by descending `utility` via `np.argsort(...)[::-1]` (L86). `evaluate_assignments` sorts bids `by=["utility","i"], ascending=[False,True]` (L155–157) — deterministic buyer-index tie-break. No random sampling anywhere. | **Supported.** Genuinely contrasts with Mitzenmacher's random-d and with Nezami's *random* candidate selection. |
| (ii) Joint balancing + provisioning via memory-slack replica expansion (not on a fixed graph). | Buyer emits `memory_bids` to neighbours with `rho>0` (L111–122); seller's `tentatively_start_replicas` branch starts replicas gated by the **utilization test** `u <= max_utilization` with **no price** (L166–186); `start_additional_replicas` invoked in `run` (L344). | **Supported.** Matches "price-free replica expansion (Alg. 4)" and "coupling LB with provisioning." |
| (iii) Objective-aligned score (β, γ, latency, fairness), re-validated by P2 / restricted re-solve. | Score `ut = beta − latency_weight·latency − fairness_weight·fairness`, thresholded by `> −gamma` (L75–80) — exactly Eq. (score) and Eq. (threshold). Each round re-solves the restricted problem via `compute_social_welfare` (`run`, L331–336) and recomputes `omega` (L337–341). | **Supported.** Score matches the note's Eq.~\eqref{eq:madig-score} *including the sign* of the price term (the note's `s = u + p` with `p` removed equals `beta − w_lat·L − w_fair·φ`). |

Additional consistency checks on the "Removed/Kept" lists and Algorithms 2–3:
- **No price update / no bid-price tracking / no reassignment** — confirmed: `evaluate_assignments` docstring and body have "no min_b tracking, no price update, no last_y replacement swap." **Correct.**
- **Buyer greedy fill until `omega` met or neighbourhood saturated** — `while idx < ... and assigned < omega[i,f]` (L88). **Correct.**
- **`unit_bids` option** ("unit bids if enabled") — present (L90–101). **Correct.**
- **Seller fills to residual capacity in descending-(utility, i) order** — L155–164. **Correct.**
- One **wording nuance** (not in the Positioning subsection, but flagged for accuracy): the seller-side replica branch in code requires `remaining_capacity == 0` **and** existing bids, then iterates `a` upward until the utilization test passes (L166–186); the note's prose ("tentatively starts replicas ... gated by `κ ≤ 1`") is faithful.

No claim in the self-description is contradicted by the code.

## Final verdict

The comparison is **mostly sound but not publishable as written**. Two sentences need editing and one needs softening:

1. **Bertsekas sentence (lines 248–252)** — *must* be reworded. Drop "fixing all prices to zero" as the trigger (it is the price-*update*/reassignment removal plus single-round, not the price *level*), and **remove "deferred-acceptance"** (technically false for a no-reassignment rule). Use the recommended phrasing above.
2. **`jiang2018` (line 246)** — *must* be removed from, or detached from, the "serve locally first, then greedily offload residual to one-hop neighbours" claim: its model is users→shared-ECS offloading, not peer one-hop diffusion. Misattribution.
3. **`nezami2021` (line 245)** — *should* be softened: its mechanism is I-EPOS collective learning with *randomly* sampled candidate hosts, so it neither exemplifies greedy one-hop residual offloading nor the deterministic full-neighbourhood scan FaaS-MADiG claims to differ by.

Correct as written: **Cybenko, Mitzenmacher, Lee & Choi** (and **Willebeek-LeMair & Reeves** modulo dropping "single-round"). The FaaS-MADiG self-description faithfully matches `decentralized_diffusion.py`.
