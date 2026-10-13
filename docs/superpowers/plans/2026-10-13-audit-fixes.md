# Audit corrections implementation plan

> Use systematic debugging, test-driven development and final independent review.

**Goal:** Correct the confirmed defects from the 2026-10-13 audit of 1645396.
**Architecture:** Keep the existing models, shared runner and clock. Repair shared
metrics and boundaries once, preserve intentional algorithm variants, remove
production-unused private helpers and consolidate duplicated behavior.
**Stack:** Python 3.10, NumPy/Pandas, Pyomo, GLPK/HiGHS/Gurobi, pytest.
**Specification:** The audit discussed with the user, who authorized direct fixes.

## Constraints and review focus

- Work on feat/frank-wolfe-relaxed-replicas; no commit or push requested.
- GitNexus upstream impact before existing-symbol edits; report shared-code risks.
- Reproduce behavioral bugs before fixing; use existing fixtures and literal expectations.
- Preserve the five-column runtime contract and historical CSV compatibility.
- Verify zero loads, fractional window arrivals, nonzero/strided timesteps,
  infeasible/time-limited solvers and fixed-replica metrics explicitly.
- Intentional continuous/integer flow domains and PLASMA round budgets are documented,
  not silently replaced by different algorithms.

## Independent implementation groups

- [x] Shared models/metrics: run_faasmacro.combine_solutions; models/model.py;
  models/selfish.py and catalog_centralized.py. Test inbound CPU, final memory,
  fixed replicas, infeasible incumbents, HiGHS limit and exhausted catalog budget.
- [x] Decentralized algorithms: BR/PG budgets and observer exclusions, honest PG
  stopping reason, greedy capacity reuse and both no-ping-pong roles. Test each
  counterexample; remove unused MADEA helper implementations where appropriate.
- [x] PLASMA/RL: window-to-rate export, export timing, zero-delay heartbeats,
  reward sign/accumulation, neighbor indexing; share the feasibility checker.
  Test real tiny engine runs and reward calculations; preserve W=1 behavior.
- [x] Parent-owned data/orchestration: integer fixed_sum conservation, heterogeneous
  singleton weights, SCV/destination conversion, step-aware reference loading and
  runtime alignment. Add regressions for the documented failing inputs.
- [x] Cleanup/docs: remove proven dead helpers/models and tests solely exercising
  removed code; clarify algorithm names, timing scopes and flow-domain differences.
- [x] Integration: focused suites, full pytest, Ruff, GitNexus detect-changes and
  final independent diff review. Fix important findings, then rerun affected checks.

## Execution record

- Baseline verified in preceding audit: 1513 pass, 1 fail (fixed_sum), Ruff passes.
- Ruling: current clean branch is the requested target; work in place.
- Ruling: implement the corrections already proposed in the audit, without another
  approval gate. Preserve mathematical design differences and document them.
- GitNexus index lives in /tmp; no generated agent instructions or skills in the repo.
- MACrO baseline correction: regenerated only `faas-macro` and `faas-macro-v0`
  for both harmonization instances using unchanged `config_for`. On the first
  instance, substituting HEAD's metrics preserved the new objectives; substituting
  HEAD's coordinator restored the old objectives. Its first round forwarded 885
  requests and allocated 244 additional replicas, but simultaneously sent and
  received at zero-based node/function pairs (5, 3) and (23, 0). The corrected
  coordinator forwarded 821, allocated 218, and had no conflicting roles.
  Reverting only the sender-role guard reproduced both conflicts. No time budget
  bound these runs; changed feasible routing and stopping decisions explain the
  welfare differences. All other baseline method entries remain unchanged.

- Added regressions for physical/strided steps, historical reference loading,
  arbitrary node IDs, nonnegative fixed-sum conservation, SCV, actual converter
  destination, singleton gamma, historical hierarchical labels, count horizon
  union, and fractional rejection subtraction. Observed each defect before fixing.
- Independent review caught two further integration cases: Gurobi's empty timeout
  placeholder and positional consumers of reindexed solution frames. Both now
  have regression tests; historical method labels remain distinct.
- Real GLPK smoke: centralized at min=max=2 followed by MADEA with that fixed
  reference succeeds; both runtime and solution readers retain step 2.
- Historical raw artifact comparisons now require an explicit RUNNER_BASELINE
  capture, and fail on missing/ambiguous supplied captures. Arbitrary old files
  in /tmp no longer silently define the expected semantics of the current tree.
  Strict byte comparison itself is unchanged; both PLASMA variants still match
  the previous W=1 integer capture. Only the two MACrO numerical baseline entries
  were updated after coordinator attribution above.
- Ruff standard and expanded production F/E9 checks pass; git diff --check passes.
- GitNexus detect-changes succeeds (read-only escalation needed for its git
  subprocess); changed execution flows match the shared-model, runner, generation
  and postprocessing scope.
- Full-suite order check found a HiGHS global thread-pool conflict caused by
  the new solver tests using automatic threads before older one-thread tests.
  The new probes now consistently request one thread. The previously failing
  sequence passes (21 tests); no production scheduler reset was introduced.

## Final validation

- All 1,559 collected test cases covered on the integrated production code.
  The full run passed the first 596, then stopped at the old hierarchical alias
  expectation. After updating that test and the related legacy-resume test, the
  remaining 963 cases ran: 962 passed, with only an old unsuffixed-column
  expectation failing. That expectation was updated to the consistent method
  suffix and rerun successfully with the 32-test postprocessing suite.
  Hierarchical selection/resume checks also passed (6 + 2 focused tests).
  These final adjustments changed test expectations only, not production code.
- Numerical baselines: all 22 registered methods, both materialized instances,
  worker independence and tiny algorithm budgets validated. Only MACrO's two
  explicitly attributed baseline entries were changed.
- Final Ruff (repository configuration and production F/E9) and git diff --check
  pass. Existing Pandas/Matplotlib warnings remain informational.
- No remote jobs, full RL training, commits or pushes were performed.
