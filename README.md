# DFaaSOptimizer

The repository includes the implementation of the centralized 
LoadManagementModel (LMM) and the FaaS-MACrO (Multi-Agent Cooperative 
Orchestration) approach developed to tackle the Function Replica Allocation 
and Load Balancing problem (FRALB) problem (and its distributed version 
DiFRALB) in a FaaS-enabled Edge-Cloud continuum.

The repository includes scripts to run each approach independently, run both 
sequentially for comparison, post-process results, and perform “what-if” 
analyses regarding incomplete convergence in FaaS-MACrO.

Follow the instructions in the next sections to install and run the two 
methods.

### Installation instructions

We strongly recommend running the code in a dedicated Python virtual 
environment. To create and activate the environment, run:

```
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required dependencies in the new environment by running:

```
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> The provided code was tested under Python version 3.10.15

### How to run experiments

#### LoadManagementModel

The entrypoint to execute LMM is the 
[`run_centralized_model.py`](run_centralized_model.py) 
script. After 
properly defining the 
[experiment configuration](#how-to-configure-experiments), the intended usage 
is:

```
usage: run_centralized_model.py [-h] [-c CONFIG] [--disable_plotting]

Run LoadManagementModel (LMM)

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file (default: manual_config.json)
  --disable_plotting    True to disable automatic plot generation for each 
                        experiment (default: False)
```

#### FaaS-MACrO

The entrypoint to execute FaaS-MACrO is the 
[`run_faasmacro.py`](run_faasmacro.py) 
script. After 
properly defining the 
[experiment configuration](#how-to-configure-experiments), the intended usage 
is:

```
usage: run_faasmacro.py [-h] [-c CONFIG] [-j PARALLELISM] [--disable_plotting]

Run FaaS-MACrO

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file (default: manual_config.json)
  -j PARALLELISM, --parallelism PARALLELISM
                        Number of parallel processes to start 
                        (-1: auto, 0: sequential) (default: -1)
  --disable_plotting    True to disable automatic plot generation for each 
                        experiment (default: False)
```

#### Run a series of experiments with the two approaches

The [`run.py`](run.py) script allows to automatically run LMM and/or 
FaaS-MACrO on a series of experiments, generated according to the provided 
parameters and the information listed in the 
[configuration file](#sequences-of-multiple-experiments).

Intended usage:

```
usage: run.py [-h] [-c CONFIG] [--n_experiments N_EXPERIMENTS] 
[--run_centralized_only] [--run_faasmacro_only] [--postprocessing_only] 
[--generate_only] [--postprocessing_list] [--fix_r] [-j SP_PARALLELISM] 
[--enable_plotting] [--loop_over LOOP_OVER]

Run LMM and/or FaaS-MACrO on multiple experiments

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file (default: config.json)
  --n_experiments N_EXPERIMENTS
                        Number of experiments to run for each 
                        configuration (default: 3)
  --run_centralized_only
  --run_faasmacro_only
  --postprocessing_only
  --generate_only
  --postprocessing_list
                        To be used in conjunction with postprocessing_only. 
                        True if the base_solution_folder includes multiple 
                        subfolders to post-process (default: False)
  --fix_r               True to fix the number of replicas in FaaS-MACrO 
                        according to the optimal centralized solution 
                        (default: False)
  -j SP_PARALLELISM, --sp_parallelism SP_PARALLELISM
                        Number of parallel processes to start 
                        (-1: auto, 0: sequential) (default: -1)
  --enable_plotting
  --loop_over LOOP_OVER
                        Key to loop over (default: Nn)
```

### How to configure experiments

Each experiment can be configured by defining an appropriate configuration 
file (in JSON format). The file includes the following fields:
- `base_solution_folder`: path to the folder where experiments data and 
results should be stored (preferably, this should be a subfolder of 
[`solutions`](solutions) or provided as a complete path).
- `verbose`: verbosity level (integer). Expected values are 0 (no logging), 1 
(limited logging) or 2 (extended logging). If the provided value is greater 
than 0 when executing the 
[`run.py` script](#run-a-series-of-experiments-with-the-two-approaches), logs 
are saved on a file `<base_solution_folder>/<experiment directory>/out.log`
- `seed` for random number generation (integer).
- `max_steps`: how many workload samples to generate (integer).
- (optional) `min_run_time`: first workload sample to consider 
(integer, default: 0).
- (optional) `max_run_time`: last workload sample to consider 
(integer, default: `max_steps`).
- `solver_name`: name of the MILP solver.
- (optional) `solver_options`: dictionary of options to be passed as 
parameters to the solver. If provided, it may include:
  - `general`: dictionary of general options considered by all solvers. 
  Examples (for the Gurobi solver; see the solver 
  [documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html) 
  for further details):
    - `TimeLimit`: maximum time limit for the optimization (in seconds).
    - `MIPGap`: relative MIP optimality gap.
    - `FeasibilityTol`: primal feasibility tolerance.
  - `coordinator`: dictionary of specific options for the coordinator solver 
  in FaaS-MACrO. This section should be used to decide whether the coordinator 
  should solve the MILP coordination problem or apply the greedy coordination 
  algorithm. In particular:
    - the default approach is to solve the MILP model. Define the 
    `sorting_rule` option (providing value `product`, `beta` or `omega`) 
    to use the greedy algorithm.
    - an additional parameter `heuristic_only` (True by default) can be 
    set to False to use the greedy algorithm solution as initial solution for 
    the MILP coordinator model.
- (optional) `checkpoint_interval`: how often the current solution should be 
saved for fault tolerance.
- `max_iterations` for FaaS-MACrO.
- (optional) `plot_interval`: how often the current solution should be 
plot to check progress.
- `patience` for FaaS-MACrO convergence.
- `limits`: dictionary of parameters used to generate the problem instances. 
Further details are provided in the following.

> [!NOTE]
> LMM and FaaS-MACrO have been tested considering the 
> [Gurobi](https://www.gurobi.com) and GLPK solvers available in 
> [Pyomo](https://pyomo.readthedocs.io/en/6.8.0/index.html). When providing 
> additional options to the solver, you must follow the corresponding 
> syntax (as an example, the time limit option in Gurobi is provided through 
> a `TimeLimit` parameter, while it is provided as `tmlim` in GLPK).

Example of configuration file to run a single experiment with LMM:

```
{
  "base_solution_folder": "solutions/test_LMM",
  "verbose": 2,
  "seed": 4850,
  "max_steps": 100,
  "min_run_time": 0,
  "max_run_time": 10,
  "solver_name": "gurobi",
  "solver_options": {
    "general": {
      "TimeLimit": 120,
      "MIPGap": 1e-05
    }
  },
  "checkpoint_interval": 1,
  "plot_interval": 1
  "limits": {
    ...
  }
}
```

#### Parameters for instance generation

Each problem instance is generated by assigning values to parameters based 
on the ranges defined through the `limits` dictionary. The fields to set 
are:
- `Nn`: the number of nodes.
- `Nf`: the number of functions deployed on each node.
- `demand`: the service time of functions on the different nodes.
- `memory_requirement`: the memory demand of each function.
- `memory_capacity`: the memory capacity of each node.
- `neighborhood`: network topology.
- `max_utilization`: the maximum utilization level.
- `load`: incoming requests rate for each function and node.
- `weights`: objective function weights.

#### Sequences of multiple experiments

### Compare results

Intended usage:

```
usage: compare_results.py [-h] 
-i POSTPROCESSING_FOLDERS [POSTPROCESSING_FOLDERS ...] 
[--run {compare_results,compare_across_folders,compare_single_model}] 
[-o COMMON_OUTPUT_FOLDER] [--loop_over LOOP_OVER] 
[--loop_over_label LOOP_OVER_LABEL] [--models [MODELS ...]] 
[--filter_by FILTER_BY] [--keep_only KEEP_ONLY] [--drop_value DROP_VALUE] 
[--folder_parse_format FOLDER_PARSE_FORMAT] 
[--single_model_baseline SINGLE_MODEL_BASELINE]

Compare results

options:
  -h, --help            show this help message and exit
  -i POSTPROCESSING_FOLDERS [POSTPROCESSING_FOLDERS ...], 
  --postprocessing_folders POSTPROCESSING_FOLDERS [POSTPROCESSING_FOLDERS ...]
                        Results folder (or list of result folders) 
                        (default: None)
  --run {compare_results,compare_across_folders,compare_single_model}
                        What to do (default: compare_results)
  -o COMMON_OUTPUT_FOLDER, --common_output_folder COMMON_OUTPUT_FOLDER
                        Path to a folder where to save plots (default: None)
  --loop_over LOOP_OVER
                        Key to loop over (default: Nn)
  --loop_over_label LOOP_OVER_LABEL
                        Label to be attached to the loop-over key 
                        (default: Number of agents)
  --models [MODELS ...]
                        List of model names 
                        (default: ['LoadManagementModel', 'FaaS-MACrO'])
  --filter_by FILTER_BY
                        Key to filter (default: None)
  --keep_only KEEP_ONLY
                        Unique value to keep (default: None)
  --drop_value DROP_VALUE
                        Unique value to drop (default: None)
  --folder_parse_format FOLDER_PARSE_FORMAT
                        Format to parse the folder name (default: None)
  --single_model_baseline SINGLE_MODEL_BASELINE
                        Baseline for time comparison (default: None)
```
