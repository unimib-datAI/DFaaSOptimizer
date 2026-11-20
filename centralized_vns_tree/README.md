# VNS-Tree

VNS-Tree optimizes integer forwarding flows at fixed local allocation `x`. It uses the CPU or native CUDA for local search and HiGHS on the CPU for upper bounds. From the DFaaSOptimizer directory, using the same environment and inputs as VNS-Frontier, run

```sh
python -m centralized_vns_tree INSTANCE --solution PREVIOUS/solution.npz \
  --backend cuda --pool-size 512 --candidates 512 --depth 20 \
  --refine-every 256 --time-limit 300 --relative-gap 0.001 --output NEW_DIRECTORY
```

With a saved solution the search starts from its allocation and flows. Omitting it selects the DP allocation and starts with zero flows. The output includes a checked sparse solution and a JSON report.

The root LP selects node-disjoint edges with fractional flow and a feasible integer split. Each binary decision constrains that edge to at most the threshold or at least the threshold plus one. Twenty eligible splits describe 1,048,576 possible full assignments. The search starts with one trajectory per coarse prefix, up to the largest power of two within the pool size. At each refinement, up to eight leading and eight rotating slots are considered. When a prefix splits, its children inherit its upper bound, the trajectory keeps its best solution's child, and its sibling waits for a future visit. A wait queue belongs to each coarse prefix.

The four move types insert flow, relocate it between destinations, replace another function's incoming flow at a receiver, or remove flow during a shake. The generator uses branch bounds, residual demand and receiver capacity to choose positive transfer quantities. A candidate slot can exhaust eight attempts and yield no move. Admissible proposals still undergo gain and feasibility checks. Disjoint active regions prevent cross-region duplicates within a search iteration. Repeats within one region remain possible.

HiGHS receives 80% of the time left after partitioning for the global solve, then up to three seconds per regional slice. The largest outstanding leaf upper bound covers the entire space, including waiting siblings. The relative gap uses the coordinator objective with fixed local utility removed and denominator `max(1, abs(incumbent))`. Up to four final cross-region path-relinking passes follow the timed search. The report records search time separately from total elapsed time. Set depth to nine with a 512-slot pool to compare the same move generator on a flat partition.
