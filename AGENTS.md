# Repository Guidelines

## Project Structure & Module Organization
This repository is script-driven (no `src/` package). Main entry points live at the top level:
- `run.py`: orchestrates batch experiment generation/execution and postprocessing.
- `run_centralized_model.py`, `run_faasmacro.py`, `run_faasmadea.py`: method-specific runners.
- `generate_data.py`, `postprocessing.py`, `logs_postprocessing.py`: data creation and analysis utilities.
- `models/`: Pyomo model definitions (`model.py`, `rmp.py`, `sp.py`, `auction_models.py`).
- `config_files/`: JSON experiment configs.
- `solutions/`: generated outputs, logs, and plots.

## Build, Test, and Development Commands
Use Python 3.10 in a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Key workflows:
```bash
python run_centralized_model.py -c config_files/manual_config.json
python run_faasmacro.py -c config_files/manual_config.json -j 0
python run.py -c config_files/config.json --methods centralized faas-macro --n_experiments 3
python compare_results.py --help
```
Use `--disable_plotting` for faster smoke runs when validating logic changes.

## Coding Style & Naming Conventions
Follow the existing style in this codebase:
- Python with type hints where practical.
- `snake_case` for variables/functions/files, `PascalCase` for classes.
- Keep the current 2-space indentation style used across scripts and `models/`.
- Prefer small, single-purpose functions and explicit argparse options for CLI scripts.

## Testing Guidelines
There is currently no dedicated automated test suite in-tree. Validate changes with focused smoke tests:
```bash
python run_centralized_model.py --help
python run_faasmacro.py --help
python run.py --help
```
For behavioral changes, run at least one small experiment and verify artifacts under `solutions/` (e.g., objective CSVs, termination logs, generated plots).

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (often lowercase), e.g., `avoid errors in postprocessing partial results`.
- Keep commit messages concise and action-oriented.
- Keep one logical change per commit.
- In PRs, include: purpose, config used, commands executed, and a short summary of result deltas (tables/plots when relevant).
- Link related issues and note solver assumptions (e.g., Gurobi vs GLPK) for reproducibility.

## Configuration & Data Hygiene
Do not commit large generated outputs from `solutions/` unless explicitly required for review. Keep environment-specific or absolute paths out of config JSONs.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **DFaaSOptimizer** (654 symbols, 1874 relationships, 55 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/DFaaSOptimizer/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/DFaaSOptimizer/context` | Codebase overview, check index freshness |
| `gitnexus://repo/DFaaSOptimizer/clusters` | All functional areas |
| `gitnexus://repo/DFaaSOptimizer/processes` | All execution flows |
| `gitnexus://repo/DFaaSOptimizer/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
