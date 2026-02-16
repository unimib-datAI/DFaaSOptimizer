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
