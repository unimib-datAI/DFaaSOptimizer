"""Buyer-local confirmation after assignment; messages retain their contract."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import run_faasmadea as madea


@pytest.mark.parametrize("valued", [False, True])
@pytest.mark.parametrize("residual", [0., 1e-6, 0.25, 2.])
def test_confirmation_preserves_messages_and_inputs(valued, residual):
  rows = dict(i=[0, 0, 0], j=[2, 1, 2], f=[0, 0, 1])
  if valued:
    rows.update(d=[5., 0., 7.], v=[3., 4., 5.])
  candidates = pd.DataFrame(rows, index=[4, 2, 9])
  before = candidates.copy(deep=True)
  omega = np.array([residual, 0.])
  expected = candidates.iloc[:2].copy() if residual > 1e-6 else candidates.iloc[:0].copy()
  if valued and residual > 1e-6:
    expected['d'] = residual
  actual = madea.confirm_memory_bids(candidates, omega)
  assert_frame_equal(actual, expected)
  assert_frame_equal(candidates, before)
  np.testing.assert_array_equal(omega, [residual, 0.])


@pytest.mark.parametrize('columns', [['i', 'j', 'f'], ['i', 'j', 'f', 'd', 'v']])
def test_confirmation_empty_preserves_schema(columns):
  candidates = pd.DataFrame(columns=columns)
  assert_frame_equal(madea.confirm_memory_bids(candidates, np.zeros(1)), candidates)


def test_confirmation_uses_runner_tolerance_and_no_persistent_withdrawal():
  candidates = pd.DataFrame(dict(i=[0, 0], j=[1, 1], f=[0, 0], d=[0., 0.], v=[7., 8.]))
  assert madea.confirm_memory_bids(candidates, np.array([0.25]), tolerance=0.3).empty
  assert_frame_equal(madea.confirm_memory_bids(candidates, np.array([2.])), candidates.assign(d=2.))


def three_node_case(demand=1., seller_capacity=1.):
  nodes = range(1, 4)
  data = {None: {
    'Nn': {None: 3}, 'Nf': {None: 1},
    'incoming_load': {(1, 1): demand, (2, 1): 1., (3, 1): 0.},
    'neighborhood': {(i, j): int(i != j) for i in nodes for j in nodes},
    'demand': {(i, 1): 1. for i in nodes}, 'max_utilization': {1: 1.},
    'memory_capacity': {1: 0, 2: seller_capacity + 1., 3: 4},
    'memory_requirement': {1: 1},
    'alpha': {(i, 1): 2. for i in nodes},
    'gamma': {(i, 1): 1. for i in nodes},
    'delta': {(i, 1): 0. for i in nodes},
    'beta': {(i, j, 1): 1. for i in nodes for j in nodes},
  }}
  x = np.array([[0.], [1.], [0.]])
  target = np.array([[demand], [0.], [0.]])
  state = madea.MadeaState(
    y=np.zeros((3, 3, 1)), omega=target.copy(), p=np.zeros((3, 1)),
    fairness=np.zeros((3, 1)), sp_r=np.array([[0.], [seller_capacity + 1.], [0.]]),
    sp_rho=np.array([0., 0., 4.]),
  )
  from hierarchical_auction.madea_runner import build_auction_options
  config = dict(solver_name='glpk', max_iterations=1, patience=10)
  kwargs = dict(sp_x=x, sp_omega=target, sp_data=data, data=data,
    agents=[0, 1, 2], loadt=data[None]['incoming_load'],
    neighborhood=np.ones((3, 3)) - np.eye(3), latency=np.zeros((3, 3)),
    config=config, auction_options=build_auction_options(config), parallelism=0, started_at=0.)
  return state, kwargs


@pytest.mark.parametrize('variant', ['madea', 'madea2', 'madea2-b'])
@pytest.mark.parametrize('demand,accepted,candidate_d', [(1., 1., 0.), (5., 3., 5.), (2., 0., 0.)])
def test_cycle_confirms_after_auction_with_real_allocator(monkeypatch, variant, demand, accepted, candidate_d):
  import madea2
  state, kwargs = three_node_case(demand, accepted)
  kwargs['auction_options']['unit_bids'] = False
  valued = variant != 'madea'
  rows = dict(i=[0], j=[2], f=[0])
  if valued:
    rows.update(d=[candidate_d], v=[2.])
  candidates = pd.DataFrame(rows)
  events, allocations, confirmed_at_stop = [], [], []
  allocator = madea2.start_additional_replicas if valued else madea.start_additional_replicas
  evaluate = madea.evaluate_bids
  stop = madea.check_stopping_criteria

  def define(*args, **kw):
    # A refused offer also had a zero-demand forced memory candidate.
    bids = pd.DataFrame(dict(i=[0], j=[1], f=[0], d=[max(accepted, 2.)], b=[2. if accepted else -100.]))
    if accepted:
      bids['d'] = accepted
    return bids, candidates, 1

  def auction(*args, **kw):
    events.append('auction')
    assert kw['tentatively_start_replicas'] is False
    return evaluate(*args, **kw)

  def allocate(bids, *args):
    events.append('allocate')
    allocations.append(bids.copy())
    return allocator(bids, *args)

  def stopping(*args, **kw):
    confirmed_at_stop.append(args[8].copy())
    return stop(*args, **kw)

  monkeypatch.setattr(madea, 'check_stopping_criteria', stopping)
  madea.run_madea_cycle(state, **kwargs, rules=dict(define_bids=define, evaluate_bids=auction, start_additional_replicas=allocate))
  assert state.y.sum() == accepted
  assert state.omega[0, 0] == demand - accepted
  if demand == accepted:
    assert events == ['auction']
    assert confirmed_at_stop[0].empty
    assert state.sp_r[2, 0] == 0
  else:
    assert events == ['auction', 'allocate']
    assert allocations[0]['j'].tolist() == [2]
    if valued:
      assert allocations[0]['d'].tolist() == [demand - accepted]
      assert allocations[0]['v'].tolist() == [2.]
    assert state.sp_r[2, 0] > 0
  np.testing.assert_array_equal(kwargs['sp_x'] + state.y.sum(axis=1) + state.omega, [[demand], [1.], [0.]])
  assert (state.sp_r[:, 0] <= [0., accepted + 1., 4.]).all()
  assert (state.y >= 0).all()


def test_cycle_confirmation_time_is_longest_buyer_and_skipped_when_empty(monkeypatch):
  from utils import step_time
  clock_value = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: clock_value[0])
  real_confirm = madea.confirm_memory_bids
  phases = []
  def confirm(candidates, residual_demand, **kwargs):
    assert step_time._phase.get() is not None
    assert not step_time._observing.get()
    phases.append(step_time._phase.get())
    clock_value[0] += 2. if candidates['i'].iloc[0] == 0 else 3.
    return real_confirm(candidates, residual_demand, **kwargs)
  monkeypatch.setattr(madea, 'confirm_memory_bids', confirm)
  for nonempty in [True, False]:
    state, kwargs = three_node_case()
    candidates = pd.DataFrame(dict(i=[0, 2], j=[2, 1], f=[0, 0]))
    if not nonempty:
      candidates = candidates.iloc[:0]
    def define(*args, **kw):
      return pd.DataFrame(columns=['i', 'j', 'f', 'd', 'b']), candidates, 0
    clock = step_time.start_step()
    try:
      madea.run_madea_cycle(state, **kwargs, rules=dict(madea.madea_rules(), define_bids=define))
      row = step_time.end_step(clock)
    finally:
      step_time.cancel_step(clock)
    assert row['algorithm_s'] == (3. if nonempty else 0.)
    assert row['wall_s'] == (5. if nonempty else 0.)
  assert len(phases) == 2 and phases[0] is phases[1]


def test_cycle_does_not_repeat_auction_or_allocate_after_direct_starts(monkeypatch):
  state, kwargs = three_node_case(2., 0.)
  calls = []
  def define(*args, **kw):
    return (pd.DataFrame(dict(i=[0], j=[1], f=[0], d=[2.], b=[2.])),
            pd.DataFrame(dict(i=[0], j=[2], f=[0])), 1)
  def evaluate(*args, **kw):
    calls.append('auction')
    assert kw['tentatively_start_replicas'] is False
    a = np.zeros((3, 1)); a[2, 0] = 1.
    return np.zeros((3, 3, 1)), state.p, a, 1
  def allocate(*args):
    pytest.fail('auction already started replicas')
  madea.run_madea_cycle(state, **kwargs, rules=dict(define_bids=define, evaluate_bids=evaluate, start_additional_replicas=allocate))
  assert calls == ['auction']


@pytest.mark.parametrize('response,target', [('greedy', 1.), ('reopt', 2.), ('reopt', 0.)])
def test_best_response_confirms_local_target_sequentially(response, target, monkeypatch):
  import decentralized_bestresponse as br
  from test_bestresponse_helpers import _base_data, _opts
  from utils import step_time
  data = _base_data(3, 1)
  clock_value = [0.]
  monkeypatch.setattr(step_time, 'perf_counter', lambda: clock_value[0])
  if hasattr(br, 'confirm_memory_bids'):
    original = br.confirm_memory_bids
    def confirm(candidates, residual, **kwargs):
      assert step_time._phase.get() is None
      clock_value[0] += 2.
      return original(candidates, residual, **kwargs)
    monkeypatch.setattr(br, 'confirm_memory_bids', confirm)
  clock = step_time.start_step()
  try:
    delta, memory, *_ = br.best_response_sweep(
      np.array([[1.], [0.], [0.]]), np.array([[0.], [2.], [0.]]), data,
      np.ones((3, 3))-np.eye(3), np.array([0., 4., 4.]), _opts(),
      np.zeros((3, 3)), np.zeros((3, 1)), True, order='fixed', response=response,
      reopt_fn=lambda i, ub: (np.array([target]), 0.),
    )
    timing = step_time.end_step(clock)
  finally:
    step_time.cancel_step(clock)
  assert delta.sum() == target
  assert memory.empty
  assert timing['parallel_phases'] == 0
  assert timing['algorithm_s'] == timing['wall_s'] == 2.


def runner_context(kwargs):
  from types import SimpleNamespace
  return SimpleNamespace(
    Nn=3, Nf=1, tolerance=1e-6, agents=[0, 1, 2], solver_name='glpk',
    general_options={}, parallelism=0, patience=10, config=kwargs['config'],
    neighborhood=kwargs['neighborhood'], latency=kwargs['latency'],
    dense_latency=kwargs['latency'], reference=None, max_iterations=1,
    time_limit=float('inf'), step_data=lambda data, t: data,
  )


@pytest.mark.parametrize('runner', ['diffuse', 'powd', 'legacy'])
@pytest.mark.parametrize('accepted', [True, False])
def test_other_runners_withdraw_only_served_candidates(monkeypatch, runner, accepted):
  import decentralized_diffusion as diffuse
  from hierarchical_auction import madea_runner as legacy
  state, kwargs = three_node_case(1., 1. if accepted else 0.)
  owner = legacy if runner == 'legacy' else diffuse
  def solve(data, *args, **kw):
    return (data, kwargs['sp_x'], None, None, kwargs['sp_omega'], state.sp_r,
      state.sp_rho, np.zeros((3, 1)), {'tot': 0.}, {'tot': 'optimal'}, {'tot': 0.})
  monkeypatch.setattr(owner, 'solve_subproblem', solve)
  def define(*args, **kw):
    bids = pd.DataFrame(dict(i=[0], j=[1], f=[0], d=[1.], b=[2.], utility=[2.]))
    return bids, pd.DataFrame(dict(i=[0], j=[2], f=[0])), 1
  allocated = []
  original = owner.start_additional_replicas
  def allocate(bids, *args):
    allocated.append(bids.copy())
    return original(bids, *args)
  monkeypatch.setattr(owner, 'start_additional_replicas', allocate)
  ctx = runner_context(kwargs)
  if runner == 'legacy':
    monkeypatch.setattr(legacy, 'define_bids', define)
    result = legacy._solve_step(ctx, 0, kwargs['data'], auction_options=kwargs['auction_options'], first_level_options=kwargs['auction_options'])
  else:
    # PowD delegates to the same assignment_step with its own buyer rule.
    if runner == 'powd':
      import decentralized_powerd as powd
      assert powd.assignment_step is diffuse.assignment_step
    result = diffuse.assignment_step(ctx, 0, kwargs['data'], options={'latency_weight': 0., 'fairness_weight': 0.}, assign=define)
  assert len(allocated) == (0 if accepted else 1)
  assert result.solutions['LSPc']['y'].sum() == (1. if accepted else 0.)


@pytest.mark.parametrize('variant', ['madea', 'madea2', 'madea2-b', 'classic'])
@pytest.mark.parametrize('demand', [0., 1.])
def test_one_shot_and_classic_keep_memory_only_without_capacity_bids(monkeypatch, variant, demand):
  import decentralized_auction as one_shot
  import madea2
  from hierarchical_auction import runner as classic
  state, kwargs = three_node_case(demand, 0.)
  owner = classic if variant == 'classic' else one_shot
  def solve(data, *args, **kw):
    return (data, kwargs['sp_x'], None, None, kwargs['sp_omega'], state.sp_r,
      state.sp_rho, np.zeros((3, 1)), {'tot': 0.}, {'tot': 'optimal'}, {'tot': 0.})
  monkeypatch.setattr(owner, 'solve_subproblem', solve)
  ctx = runner_context(kwargs)
  ctx.solver_options = {'auction': kwargs['auction_options']}
  ctx.solver_options['auction']['memory_bid_value'] = 'bid' if variant.endswith('-b') else 'beta'
  allocations = []
  valued = variant.startswith('madea2')
  original = madea2.start_additional_replicas if valued else madea.start_additional_replicas
  def allocate(bids, *args):
    allocations.append(bids.copy())
    return original(bids, *args)
  if variant == 'classic':
    monkeypatch.setattr(classic, 'start_additional_replicas', allocate)
    classic._solve_step(ctx, 0, kwargs['data'], auction_options=kwargs['auction_options'], engine_class=classic.HierarchicalAuctionEngine, refinement=None)
  else:
    one_shot._solve_step(ctx, 0, kwargs['data'], refinement=None,
      rules=dict(define_bids=madea2.define_bids if valued else madea.define_bids,
                 evaluate_bids=madea.evaluate_bids, start_additional_replicas=allocate))
  assert len(allocations) == 1
  assert allocations[0].empty == (demand == 0.)
  assert state.sp_r[2, 0] == (0. if demand == 0. else 4. if not valued else 1.)


@pytest.mark.parametrize('capacity', [0., 1.])
def test_dual_memory_messages_reflect_actual_assignment(capacity):
  from decentralized_dual import dual_coordination_round, _dual_options
  from test_dual_helpers import make_data, DUAL_OPTIONS
  target = np.array([[1.], [0.], [0.]])
  delta, bids, *_ = dual_coordination_round(
    target, np.array([[0.], [capacity], [0.]]), make_data(),
    np.ones((3, 3)) - np.eye(3), np.array([0., 2., 2.]), _dual_options({}),
    np.zeros((3, 3)), np.zeros((3, 1)),
  )
  assert delta.sum() == capacity
  assert bids.empty == (capacity == 1.)


def test_pg_rejected_proposal_keeps_its_own_scarcity_messages():
  from decentralized_potentialgame import node_move, compute_z
  from test_potentialgame_sweep import _data_2n_1f, _state
  data = _data_2n_1f()
  x, y, r = _state()
  before = [a.copy() for a in (x, y, r)]
  def proposal(i, ub):
    return np.array([0.]), np.array([0.]), np.array([0.]), 0.
  accepted, _, bids, _ = node_move(
    0, x, y, r, data, np.array([[0, 1], [1, 0]]), np.array([0., 10.]),
    1e-6, proposal, 1e-9,
  )
  assert not accepted
  assert compute_z(x, y, data)[0, 0] == 0.
  assert bids == dict(i=[0], j=[1], f=[0])
  for got, expected in zip((x, y, r), before):
    np.testing.assert_array_equal(got, expected)


def test_best_response_mixed_empty_confirmations_keep_integer_message_ids():
  from decentralized_bestresponse import best_response_sweep
  from test_bestresponse_helpers import _base_data, _opts
  _, memory, *_ = best_response_sweep(
    np.array([[1.], [1.], [0.]]), np.array([[0.], [0.], [1.]]), _base_data(3, 1),
    np.ones((3, 3))-np.eye(3), np.array([0., 0., 4.]), _opts(),
    np.zeros((3, 3)), np.zeros((3, 1)), False, order='fixed', response='greedy',
  )
  assert memory['i'].tolist() == [1]
  assert all(pd.api.types.is_integer_dtype(memory[c]) for c in ['i', 'j', 'f'])


def test_served_buyer_can_request_memory_again_after_reassignment(monkeypatch):
  state, kwargs = three_node_case(1., 1.)
  kwargs['config']['max_iterations'] = 2
  seen = []
  original_allocate = madea.start_additional_replicas
  original_evaluate = madea.evaluate_bids
  def define(*args, **kw):
    return (pd.DataFrame(dict(i=[0], j=[1], f=[0], d=[1.], b=[2.])),
            pd.DataFrame(dict(i=[0], j=[2], f=[0])), 1)
  def evaluate(*args, **kw):
    if kw['it'] == 0:
      return original_evaluate(*args, **kw)
    # The next auction revokes the buyer's previous assignment.
    return -state.y.copy(), state.p, np.zeros((3, 1)), 1
  def allocate(bids, *args):
    seen.append(bids.copy())
    return original_allocate(bids, *args)
  outcomes = iter([(False, None), (True, 'test completed')])
  monkeypatch.setattr(madea, 'check_stopping_criteria', lambda *a, **k: next(outcomes))
  madea.run_madea_cycle(state, **kwargs, rules=dict(define_bids=define, evaluate_bids=evaluate, start_additional_replicas=allocate))
  assert len(seen) == 1
  assert seen[0]['i'].tolist() == [0]
  assert state.omega[0, 0] == 1.
  assert state.sp_r[2, 0] > 0
