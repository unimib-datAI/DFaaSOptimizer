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
methods:
- [Installation instructions](#installation-instructions)
- [How to run experiments](#how-to-run-experiments)
- [How to configure experiments](#how-to-configure-experiments)
- [Compare results](#compare-results)

## Installation instructions

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

## How to run experiments

### LoadManagementModel

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

### FaaS-MACrO

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

> [!CAUTION]
> The parallelism level determines how many local subproblems are solved in 
> parallel. When setting it, mind that MILP solvers as Gurobi usually 
> exploit multithreading. Keep `j` small to avoid issues with aggressive 
> over-commitment.

### Run a series of experiments with the two approaches

The [`run.py`](run.py) script allows to automatically generate multiple 
experiment instances, run LMM and/or FaaS-MACrO on a series of experiments, 
and perform results post-processing to compare the two approaches. Experiment 
instances are generated according to the provided parameters and the 
information listed in the 
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

> [!WARNING]
> The `loop_over` key must match the key for which multiple values are 
> provided in the configuration file (e.g., `Nn` or `Nf`).

## How to configure experiments

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

### Parameters for instance generation

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

#### Number of nodes and functions

For what concerns the number of nodes and functions in the system, those are 
generated uniformly at random from an interval specified by setting the 
parameters `min` and `max` in the corresponding dictionary field. As an 
example, a network with a random number of nodes selected between 2 and 10 
and exactly 3 functions deployed on each node will be generated if setting:

```
{
  ...,
  "limits": {
    "Nn": {
      "min": 2,
      "max": 10
    },
    "Nf": {
      "min": 3,
      "max": 3
    },
    ...
  }
}
```

#### Average service time

The demand (or average service time) for all functions and nodes can be 
generated according to different rules. The corresponding field in the 
`limits` dictionary can be populated by the following parameters:
- `demand_type`: specifies whether demands are `"homogeneous"` (default) 
or `"heterogeneous"`. In the first case, a different demand value is possibly 
generated for each function, but this is equivalent on all nodes where the 
function is deployed. In the second case, the demand value may be different 
also for the same function if this is deployed on different nodes.
- `values`: if provided, it should contain a list of values with length 
equal to `Nf`, reporting the average service time of each function (which is 
then equal on all nodes).
- in alternative to `values`, two fields `min` and `max` can be provided with 
the same semantic as for the nodes and functions generation: the demand 
values are extracted uniformly at random from the interval `[min,max]`. The 
generation process follows the rule set by `demand_type`.
- optionally, a `speedup_factors` dictionary can be provided to link the 
(heterogeneous) demand values to different classes of nodes. The dictionary 
should store key-value pairs whose key is a memory capacity and whose value 
is a multiplier such that the service demand of functions deployed on nodes 
with the given memory capacity is divided by the provided factor.

As an example, if the following dictionary is provided to configure a system 
with 4 functions and 3 nodes with memory capacity equal to 4Gb, 8Gb and 16Gb, 
respectively:

```
{
  ...,
  "limits": {
    ...,
    "demand": {
      "min": 0.2,
      "max": 0.5,
      "speedup_factors": {
        "4096": 0.5,
        "8192": 1.0,
        "16384": 1.2
      }
    },
    ...
  }
}
```

a demand vector with four random values between 0.2 and 0.5 is initially 
generated (the demand is homogeneous by default, thus the same value is 
applied to each function on all nodes), providing, e.g., 
`[0.3, 0.2, 0.4, 0.2]` for the four functions. Then, when the functions are 
associated with the existing nodes, these values are divided by the 
speedup factor providing:

```
first node (memory: 4Gb) --> [0.15, 0.1, 0.2, 0.1]
second node (memory: 8Gb) --> [0.3, 0.2, 0.4, 0.2]
third node (memory: 16Gb) --> [0.25, 0.167, 0.33, 0.167]
```

#### Functions memory requirement

The memory requirement of each function (in Megabytes) can be generated in 
three different ways:
- a `values` list can be provided associating a specific value to each 
function (as for the demand).
- the values can be extracted uniformly at random from an interval `[min,max]` 
if the corresponding parameters are configured.
- the values can be extracted uniformly at random from a list of given 
alternatives, provided by setting the parameter `values_from` with a list 
of admissible values (e.g., `values_from: [1024, 128, 256, 512]`).

#### Nodes memory capacity

The memory capacity (in Megabytes) of each node can be set:
- according to the same rules as the memory requirement, i.e., by setting the 
`values`, `values_from` or `min`...`max` parameters.
- assigning a given memory value to a given percentage of nodes. 

As an example, if the following dictionary is provided:

```
{
  ...,
  "limits": {
    ...,
    "memory_capacity": {
      "repeated_values": [
        [
          0.5,
          4096
        ],
        [
          0.125,
          8192
        ],
        [
          0.25,
          16384
        ],
        [
          0.125,
          24576
        ]
      ]
    },
    ...
  }
}
```

the 50% of the system nodes will have a memory capacity of 4Gb, the 12.5% 
will have a memory capacity of 8Gb, the 25% will have a memory capacity of 
16Gb, and the remaining 12.5% will have a memory capacity of 24Gb.

#### Network topology

The network topology can be generated in two different ways, setting in the 
`neighborhood` dictionary:
- a parameter `p` that represents the probability of creating a link between 
two nodes. If `p` is set to 1.0, the network is fully-connected.
- a parameter `k` that represent the degree of each node. In this case, the 
network is generated by providing `k` as parameter `d` to the 
[`random_regular_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.random_regular_graph.html) 
function of NetworkX.

> [!WARNING] 
> When setting a value `p < 1.0`, there is no guarantee that the resulting 
> graph is connected: there may exist subsets of nodes that are not linked to 
> one another.

#### Maximum utilization

The maximum utilization level is extracted uniformly at random from the 
interval `[min,max]` obtained by setting the corresponding parameters.

#### Incoming load

The `load` dictionary in the configuration file is used to generate the 
limits to be provided to a LoadGenerator object that defines the actual 
requests traces. It should include a `trace_type` field, with possible values:
- `load_existing` to reuse existing traces. In this case, an additional 
`path` field should be included providing the path to the existing trace 
files.
- `fixed_sum` if the sum of incoming load to all nodes for each function 
remains fixed across the whole trace.
- `clipped` if the generated trace should be clipped to specific minimum and 
maximum values.
- `sinusoidal` if the generated trace should follow a sinusoidal curve.

If the trace type is anything but `load_existing`, a dictionary of load limits 
is generated according to the provided parameters. In particular:
- A random dictionary of minimum and maximum values can be generated for 
each node and function if the `min` and `max` parameters are provided in 
association with the `clipped` or `sinusoidal` trace type. In this case, 
both the `min` and `max` field should further include `min` and `max` values 
as in the example below. 
- The `values` parameter should be provided in association with the 
`fixed_sum` trace type, to set the value that the total load arriving to all 
nodes should have for each function. In particular, the `values` field should 
be a list with:
  - the same length as the number of functions, or
  - a single element equal to `"auto"`. In this second case, the value of 
  total load arriving to the network for each function is automatically 
  computed to keep the nodes on the verge of saturation.

As an example, if the `load` dictionary is set to:

```
{
  ...,
  "limits": {
    ...,
    "load": {
      "trace_type": "sinusoidal",
      "values": {
        "min": {
          "min": 1,
          "max": 10
        },
        "max": {
          "min": 100,
          "max": 150
        }
      }
    }
  }
}
```

sinusoidal requests traces are generated considering, for each node and 
function, a minimum load extracted uniformly at random from the interval 
$[1,10] req/s$, and a maximum load extracted uniformly at random from the 
interval $[100,150] req/s$.

#### Objective function weights

The objective function weights can be generated following two different 
approaches. The first and simplest one requires to set ranges to randomly 
generate the value of each parameter as follows:
- the `alpha` values are extracted uniformly at random from a `[min,max]` 
interval.
- the values of the parameters `beta` are generated multiplying `alpha` by a 
`beta_multiplier`, extracted uniformly at random from a `[min,max]` interval.
- the values of the parameters `delta` are generated multiplying `beta` by a 
`delta_multiplier`, extracted uniformly at random from a `[min,max]` interval.
- the `gamma` values are extracted uniformly at random from a `[min,max]` 
interval.

Alternatively, parameters can be generated to reflect the initialization time 
of function replicas, and the communication overhead among Edge nodes and 
between Edge nodes and the Cloud. In particular, after setting, as an example:

```
{
  ...,
  "limits": {
    ...,
    "weights": {
      "initialization_time": {
        "min": 0.25,
        "max": 0.6
      },
      "input_data": {
        "avg": 0.0009765625,
        "std": 0.0009765625,
        "min": 9.765625e-05,
        "max": 5
      },
      "edge_network_latency": {
        "min": 0.005,
        "max": 0.005
      },
      "cloud_network_latency": {
        "values_from": [
          0.05,
          0.1,
          0.2
        ]
      },
      "edge_bandwidth": {
        "min": 100,
        "max": 100
      },
      "cloud_bandwidth": {
        "min": 10,
        "max": 10
      }
    }
  }
}
```

- `alpha` values are generated by extracting uniformly at random values of 
`initialization_time` from the `[min,max]` interval, normalized between 
0 and 1.
- `beta` values are generated by summing to the corresponding `alpha` a 
parameter encoding the network transfer time, computed as 
`edge_network_latency + input_data / edge_bandwidth`. Those are randomly 
generated in a `[min,max]` interval (for `edge_network_latency` and 
`edge_bandwidth`), and from a truncated normal distribution with the 
provided mean and standard deviation (for `input_data`). The resulting value 
is normalized between 0 and 1.
- `delta` is the average of `beta` over all receiving nodes.
- `gamma` is generated as 
`cloud_network_latency + input_data / cloud_bandwidth`.

### Sequences of multiple experiments

The `run.py` script can be used to run a 
[sequence of experiments](#run-a-series-of-experiments-with-the-two-approaches) 
considering, e.g., a variable network size `Nn` or number of functions `Nf`. 

To do so, the configuration file should include, for the `loop_over` 
parameter, a list of candidate values to consider. This can be provided either 
by setting a list of `values` or by providing a `[min,max]` interval 
associated with a `step` (by default, 1).

As an example, starting `run.py` with option `--loop_over Nn` and having 

```
{
  ...,
  "limits": {
    "Nn": {
      "values": [5, 10, 100]
    },
    ...
  }
}
```

entails that three experiments are executed, with `Nn` equal to 5, 10 and 100, 
respectively. Similarly, setting:

```
{
  ...,
  "limits": {
    "Nn": {
      "min": 5,
      "max": 10
    },
    ...
  }
}
```

corresponds to starting six experiments, with values of `Nn` ranging from 5 
to 10 (with step 1).

## Compare results

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

Examples (with the three `run` modes):

- `compare_results`: to be used when results obtained with different 
values of the `loop_over` parameter are provided in a unique 
`postprocessing_folders`.

```
python compare_results.py -i solutions/varyingN/light_load/3classes-0_10-greedy \
                          --run compare_results \
                          --loop_over Nn \
                          --loop_over_label "Number of agents"
```

- `compare_across_folders`: to be used when results obtained with different 
values of the `loop_over` parameter are provided in different subfolders 
of `postprocessing_folders`.

```
python compare_results.py -i solutions/varyingK \
                          -o solutions/varyingK/postprocessing_by_k \
                          --run compare_across_folders \
                          --loop_over k \
                          --loop_over_label "Node degree k" \
                          --folder_parse_format 3classes-0_10-{}_{}-greedy`
```

- `compare_single_model`: to be used to compare results obtained with a 
single model in variable conditions (identified by the `loop_over` parameter).

```
python compare_results.py -i solutions/centralized \
                          -o solutions/centralized/postprocessing_by_TL \
                          --run compare_single_model \
                          --loop_over TL \
                          --loop_over_label "Time limit [s]" \
                          --folder_parse_format 3classes-0_10-centralized-{}_{}`
```
