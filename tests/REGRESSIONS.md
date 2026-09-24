# Branch review regressions

These tests specify correct behavior. Unfixed findings fail normally: they are
not marked `xfail` and do not assert that the defective behavior is desirable.
The four `test_review_*_regressions.py` files originally added 39 cases: 38
failed on commit `2dd857d`, and the sequential detailed-price control passed.

Run the review cases with:

```sh
uv run --locked pytest -q tests/test_review_*_regressions.py
```

Run the objective safety checks with:

```sh
uv run --locked pytest -q tests/test_objective_gap.py tests/test_zero_load_*.py
```

Numerical model/runner cases use the optional system executable `glpsol`
(GLPK), and skip with an explicit reason if unavailable. No new dependency or
Gurobi license is required for those cases. Remote job execution is replaced at
the dispatcher boundary. Tests write generated data only under pytest's
temporary directories. The RL import smoke tests verify the installed training
dependencies without starting training.

## Finding-to-test mapping

File abbreviations: **F** = `test_review_faas_regressions.py`,
**O** = `test_review_optimization_regressions.py`,
**D** = `test_review_distributed_regressions.py`,
**R** = `test_review_orchestration_regressions.py`.

| Review finding | File | Test name (without `test_`) |
| --- | --- | --- |
| 1. Paper topology schema mismatch | O | `paper_suite_topology_can_be_generated` |
| 2. Nonpositive incumbent discarded/crashes | F | `runner_preserves_nonpositive_feasible_incumbent` |
| 3. PLASMA simultaneous sending/receiving | D | `plasma_current_window_cannot_both_send_and_receive` |
| 4. Failed campaign marked complete | R | `failed_campaign_suite_remains_pending_for_resume` |
| 5. Invalid screening ranking/coverage | R | `screening_does_not_promote_nan_objective`, `screening_prefers_higher_welfare_when_best_is_nonpositive`, `screening_does_not_reward_missing_difficult_instances` |
| 6. Missing/new reference mishandled | R | `fix_r_uses_newly_generated_centralized_solution`, `resume_can_add_method_without_a_centralized_reference` |
| 7. Missing greedy prior rejections | D | `greedy_adapter_accounts_for_prior_rejections` |
| 8. Missing campaign directory | R | `campaign_preparation_creates_batches_directory` |
| 9. Parallel execution ignores detailed prices | F | `detailed_prices_control_each_agents_offload_decision` |
| 10. Replica memory counted repeatedly | F | `madea_uses_each_available_replica_slot_once` |
| 11. Solver options leak between solves | O | `solver_options_do_not_leak_into_next_solve` |
| 12. Tight model changes primary optimum | O | `tight_model_preserves_primary_welfare_at_large_load` |
| 13. LSPr/LSPr_x lose rejected traffic | O | `local_reoptimization_accounts_for_all_rejected_load` |
| 14. Duplicate hierarchy token allocations | D | `overlapping_structures_do_not_spend_tokens_on_duplicate_demand` |
| 15. Negative welfare causes premature convergence | D | `hierarchical_madea_improves_a_negative_first_incumbent` |
| 16. Baselines retain failed resources/invent traffic | D | `failure_scenario_passes_the_same_failure_to_baselines` |
| 17. Negative-oracle adaptation lag | D | `adaptation_lag_is_zero_when_a_negative_oracle_is_matched` |
| 18. Unit bids exceed fractional demand | O | `powerd_unit_bids_never_exceed_fractional_demand` |
| 19. Shipped planar configurations unsupported | O | `shipped_planar_configuration_generates_connected_planar_graph` |
| 20. Integer fixed-sum traces lose load | O | `integer_fixed_sum_traces_preserve_system_workload` |
| 21. Short sinusoidal traces crash | O | `short_sinusoidal_traces_are_finite_and_within_bounds` |
| 22. Interrupted materialization blocks retries | R | `materialization_can_retry_after_interrupted_payload_write` |
| 23. Decentralized runtime reference fails | R | `runtime_comparison_accepts_decentralized_reference` |
| 24. Optional rejection CSV required accidentally | R | `custom_baseline_does_not_require_optional_rejections_csv` |
| 25. uv environment cannot import RL entrypoint dependencies | F | `uv_environment_can_import_training_environments` |

## Objective safety follow-up

The follow-up preserves normalization by each source/function's incoming load.
It does not adopt the total-load normalization from `origin/fog_nodes`, because
that changes the objective even for strictly positive loads. A zero denominator
uses a neutral divisor; positive fractional loads retain their exact divisor.
True arrival counts must not be changed to make an objective evaluable.

The incumbent and convergence corrections are covered by findings 2 and 15
above, plus `test_objective_gap.py`. Additional `test_zero_load_*` cases cover
idle systems, mixed active/idle sources/functions, incoming forwarded traffic,
and preservation of positive fractional-load utility.

RL reward tests replace only framework imports and execute the real reward,
feasibility, and scoring methods. Separate RL import smoke tests verify the
installed stack; a full training run is outside this regression suite.

## Verification after fixes

All 25 findings above have been corrected. Additional cases cover fractional
forwarding commitments, reuse of already allocated replicas, legacy failed
campaign checkpoints, and stopping-argument wiring in five decentralized runners.

The full suite yields **709 passed, 24 skipped**, with no failures. All review
and objective-safety regressions pass. The skipped integrations require Gurobi:
the sandbox cannot reach its license service, and a separate network-enabled
probe confirmed that the configured license has expired. GLPK numerical cases
run successfully. Ruff and mypy also pass with the repository configuration.
