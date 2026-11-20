"""Build the SCIP model in the coordinator objective scale."""

import numpy as np
import pyscipopt as sc
from centralized_vns_highs.core import Record
from centralized_vns_r.io import integers


def read_flow(solver, variables, p):
  sol = solver.getBestSol()
  flow = integers([solver.getSolVal(sol, v) for v in variables[: len(p.src)]], "SCIP flow")
  record = Record("scip", flow, p.check(flow))
  np.testing.assert_allclose(record.welfare, solver.getSolObjVal(sol), atol=1e-6, rtol=0)
  return record


def solution(solver, variables, values, heur=None):
  sol = solver.createOrigSol(heur)
  for i in np.flatnonzero(values):
    solver.setSolVal(sol, variables[i], values[i])
  return sol


def build(shared, log_file, gap):
  p = shared.p
  lp = shared.model.lp_
  solver = sc.Model()
  solver.setLogfile(str(log_file))
  solver.setIntParam("parallel/maxnthreads", 1)
  solver.setIntParam("lp/threads", 1)
  solver.setIntParam("randomization/randomseedshift", 4850)
  solver.setRealParam("limits/gap", gap)
  solver.setMaximize()
  solver.addObjoffset(p.base_welfare)
  types = np.asarray(lp.integrality_, dtype=int)
  assert set(np.unique(types)) <= {0, 1}
  variables = [
    solver.addVar(vtype="I" if t else "C", lb=lo, ub=hi, obj=-c)
    for t, lo, hi, c in zip(types, lp.col_lower_, lp.col_upper_, lp.col_cost_)
  ]
  assert len(variables) == shared.A.shape[1]
  for row, rhs in enumerate(shared.b):
    sl = slice(shared.A.indptr[row], shared.A.indptr[row + 1])
    solver.addCons(
      sc.quicksum(float(v) * variables[i] for i, v in zip(shared.A.indices[sl], shared.A.data[sl]))
      <= float(rhs)
    )
  return solver, variables
