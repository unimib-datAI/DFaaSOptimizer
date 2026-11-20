# VNS-Tree RAM bounds

This stage replays the leaves in a saved batch-bounds report on the Mac CPU with HiGHS. It holds the incumbent, fixed local allocation, partition and original pair cuts constant. Run from DFaaSOptimizer:

```sh
.venv/bin/python -m centralized_vns_ram_bounds INSTANCE TREE_RESULT_DIR \
  BATCH_BOUNDS_RESULT.json NEW_OUTPUT.json --seconds 300
```

For a comparison using only the receiver formulation, add `--unconditional`. The default also propagates mandatory integer traffic in each leaf.

For each receiver and function, the formulation lists the maximum integer inflow served by each useful number of extra replicas. A convex combination of these alternatives limits both aggregate incoming flow and flow from subsets of senders. Up to six inbound edges are handled by enumerating every subset. At higher degree, singleton and whole-neighborhood inequalities are used. On nodes where functions compete for spare RAM, a layered flow network allows only joint choices that fit: a transition selects an integer replica increment and subtracts its memory cost. States with enough remaining RAM to buy every remaining option can be merged.

A feasible integer forwarding plan extends to this model by selecting one option per receiving function and following the resulting RAM path. In a leaf, positive compulsory flow fixes the source and receiver roles, forces their minimum replica counts, and excludes incompatible options. These implications are local to that leaf. The shared root upper bound applies globally. The earlier exact dyadic dual checker is used for both root and regional certificates. The stage writes the raw and effective leaf bounds in the order of the baseline report. The time budget covers regional solves and certificate checks; the report also includes setup time. The size of the extended model depends on the number of traffic levels and reachable RAM states.
