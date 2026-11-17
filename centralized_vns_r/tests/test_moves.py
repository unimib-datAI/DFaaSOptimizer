import numpy as np
from centralized_vns.core import Problem,State,Options,evaluate,apply_move


def relocation_problem():
  return Problem(load=[[10,0],[0,10],[0,0],[0,0]],x=np.zeros((4,2)),
    capacity=np.full((4,2),10.),memory=[1,1],ram=[0,0,1,1],
    alpha=np.ones((4,2)),gamma=np.ones((4,2)),
    edges=[(0,2,0),(0,3,0),(1,2,1)],beta=[2,1,4])


def test_atomic_relocation_admits_a_previously_blocked_function():
  from centralized_vns_r.core import CPU
  p=relocation_problem();engine=CPU(p,Options(pool_size=1,candidates=256))
  apply_move(p,engine.state,0,np.array([[0,10],[-1,0],[-1,0],[-1,0]]))
  engine.level[0]=2
  engine.value[0]=engine.best[0]=p.check(engine.state.y[0])['welfare']
  engine.best_y[:]=engine.state.y
  before=engine.value[0]
  engine.step(0)
  # Moving A alone loses 1, inserting B alone is infeasible; together gain is 4.
  assert engine.value[0]==before+4
  np.testing.assert_array_equal(engine.state.y[0],[0,10,10])
  p.check(engine.state.y[0])


def test_rerouting_can_release_just_one_replica():
  from centralized_vns_r.moves import generate_move,incoming_index
  p=Problem(load=[[27],[0],[0]],x=np.zeros((3,1)),capacity=np.full((3,1),10.),
    memory=[1],ram=[0,3,3],alpha=np.ones((3,1)),gamma=np.ones((3,1)),
    edges=[(0,1,0),(0,2,0)],beta=[1,2])
  s=State.empty(p,1);apply_move(p,s,0,np.array([[0,27],[-1,0],[-1,0],[-1,0]]))
  moves=np.zeros((1,256,4,2),dtype=np.int64)
  for c in range(256):
    generate_move(4850,1,0,c,1,moves,s.y,s.incoming,s.rejected,
      p.x,p.capacity,p.memory,p.ram,p.src,p.dst,p.fun,p.starts,*incoming_index(p))
  expected=np.array([[0,-7],[1,7],[-1,0],[-1,0]])
  assert any(np.array_equal(m,expected) for m in moves[0])
  assert evaluate(p,s,0,expected)>0
