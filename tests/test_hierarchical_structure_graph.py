import numpy as np
import pytest

from hierarchical_auction.structure_graph import StructureGraph


def chain_graph(size: int) -> np.ndarray:
  graph = np.zeros((size, size), dtype=float)
  for i in range(size - 1):
    graph[i, i + 1] = 1.0
    graph[i + 1, i] = 1.0
  return graph


def test_level1_structure_is_one_hop_neighborhood():
  sg = StructureGraph(chain_graph(4))
  structures = sg.build_level1(num_functions=2)
  assert structures[0].member_nodes == {0, 1}
  assert structures[1].member_nodes == {0, 1, 2}
  assert structures[2].member_nodes == {1, 2, 3}
  assert structures[3].member_nodes == {2, 3}


def test_adjacency_uses_physical_edge_between_members():
  sg = StructureGraph(chain_graph(5))
  structures = sg.build_level1(num_functions=1)
  # S_0 = {0,1}, S_2 = {1,2,3}: edge (1,2) exists → adjacent
  assert sg.are_adjacent(structures[0], structures[2]) is True
  # S_0 = {0,1}, S_4 = {3,4}: no edge between {0,1} and {3,4} → not adjacent
  assert sg.are_adjacent(structures[0], structures[4]) is False


def test_aggregation_rebuilds_adjacency_for_new_level():
  sg = StructureGraph(chain_graph(5))
  level1 = sg.build_level1(num_functions=1)
  level2 = sg.aggregate_to_next_level(level1, num_functions=1)
  assert all(s.level == 2 for s in level2.values())
  assert level1[0].member_nodes.issubset(level2[0].member_nodes)
  assert all(root not in s.adjacent_structures for root, s in level2.items())


@pytest.mark.parametrize("size, edges, expected_levels", [
  (3, [(0, 1), (1, 2), (0, 2)], 1),  # Already complete at level 1.
  (3, [(0, 1), (1, 2)], 2),  # One full structure must not stop the others.
  (9, [(i, i + 1) for i in range(8)], 3),
  (7, [(0, 1), (1, 2), (3, 4), (4, 5)], 2),  # Components + isolated node.
  (1, [], 1),
])
def test_aggregation_stops_when_no_structure_can_expand(size, edges, expected_levels):
  graph = np.zeros((size, size))
  for i, j in edges:
    graph[i, j] = graph[j, i] = 1
  sg = StructureGraph(graph)
  structures = sg.build_level1(num_functions=1)
  for level in range(2, expected_levels + 1):
    structures = sg.aggregate_to_next_level(structures, num_functions=1)
    assert structures
    assert all(s.level == level for s in structures.values())
  assert sg.aggregate_to_next_level(structures, num_functions=1) == {}
