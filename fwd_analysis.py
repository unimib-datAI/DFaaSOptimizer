from run_centralized_model import encode_solution
from utilities import load_requests_traces
from postprocessing import load_solution
from load_generator import LoadGenerator

import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import json
import os


def count_requests(
    solution_folder: str, model_name: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  # count nodes and functions
  Nn, Nf = None, None
  with open(
      os.path.join(solution_folder, "base_instance_data.json"), "r"
    ) as istream:
    data = json.load(istream)
    Nn = int(data["None"]["Nn"]["None"])
    Nf = int(data["None"]["Nf"]["None"])
  # load solution
  solution, replicas, detailed_fwd_solution, _, _ = load_solution(
    solution_folder, model_name
  )
  # count local/fwd/rejected requests
  all_local = pd.DataFrame()
  all_sentrecv = pd.DataFrame()
  all_rej = pd.DataFrame()
  for t in range(len(detailed_fwd_solution)):
    x, y, z, _, _ = encode_solution(
      Nn, Nf, solution, detailed_fwd_solution, replicas, t
    )
    local = {
      "all": []
    }
    sentrecv = {
      "sent": [],
      "recv": []
    }
    rej = {
      "all": []
    }
    for n in range(Nn):
      # -- total
      local["all"].append(x[n,:].sum())
      sentrecv["sent"].append(y[n,:,:].sum())
      sentrecv["recv"].append(y[:,n,:].sum())
      rej["all"].append(z[n,:].sum())
      # -- by function
      for f in range(Nf):
        if f"f{f}" not in local:
          local[f"f{f}"] = []
        local[f"f{f}"].append(x[n,f])
        if f"f{f}_sent" not in sentrecv:
          sentrecv[f"f{f}_sent"] = []
          sentrecv[f"f{f}_recv"] = []
        sentrecv[f"f{f}_sent"].append(y[n,:,f].sum())
        sentrecv[f"f{f}_recv"].append(y[:,n,f].sum())
        if f"f{f}" not in rej:
          rej[f"f{f}"] = []
        rej[f"f{f}"].append(z[n,f])
    # add info
    local = pd.DataFrame(local)
    local["node"] = range(Nn)
    local["t"] = t
    sentrecv = pd.DataFrame(sentrecv)
    sentrecv["node"] = range(Nn)
    sentrecv["t"] = t
    rej = pd.DataFrame(rej)
    rej["node"] = range(Nn)
    rej["t"] = t
    # merge
    all_local = pd.concat([all_local, local], ignore_index = True)
    all_sentrecv = pd.concat([all_sentrecv, sentrecv], ignore_index = True)
    all_rej = pd.concat([all_rej, rej], ignore_index = True)
  return all_local, all_sentrecv, all_rej


def filter_traces(
    all_sentrecv: pd.DataFrame, all_rej: pd.DataFrame
  ) -> Tuple[list, list]:
  only_local = []
  with_offloading = []
  with_reject = []
  sentrecv_sum = all_sentrecv.groupby("t")["sent"].sum()
  reject_sum = all_rej.groupby("t")["all"].sum()
  for t, s, r in zip(sentrecv_sum.index, sentrecv_sum, reject_sum):
    if s <= 0.0 and r <= 0.0:
      only_local.append(t)
    elif r > 0.0:
      with_reject.append(t)
    else:
      with_offloading.append(t)
  return only_local, with_offloading, with_reject


def plot_filtered_traces(
    solution_folder, only_local, with_offloading, with_reject
  ):
  traces, mt, Mt, ts  = load_requests_traces(solution_folder)
  for f, f_traces in traces.items():
    # f_traces_array = {
    #   a: np.array(tr)[with_offloading] for a, tr in f_traces.items()
    # }
    f_traces_array = {
      a: np.empty_like(np.array(tr)) for a, tr in f_traces.items()
    }
    for a in f_traces_array:
      v = np.array(f_traces[a])
      last_valid = np.maximum.accumulate(
        np.where(
          np.isin(np.arange(len(v)), with_offloading),
          np.arange(len(v)),
          -1
        )
      )
      f_traces_array[a] = v[last_valid]
    LoadGenerator.plot_input_load(
      f_traces_array, 
      plot_filename = os.path.join(solution_folder, "load", f"f{f}_fwd.png")
    )


if __name__ == "__main__":
  solution_folder = "solutions/for_dfaas_instances/2026-01-27_12-07-13.163152"
  model_name = "LoadManagementModel"
  all_local, all_sentrecv, all_rej = count_requests(
    solution_folder, model_name
  )
  only_local,with_offloading,with_reject = filter_traces(all_sentrecv, all_rej)
  with open(
      os.path.join(solution_folder, "load", "trace_filtered.json"), "w"
    ) as ost:
    ost.write(
      json.dumps(
        {
          "only_local": only_local, 
          "with_offloading": with_offloading,
          "with_reject": with_reject
        }, 
        indent = 2
      )
    )
  plot_filtered_traces(
    solution_folder, only_local, with_offloading, with_reject
  )
  # plot all
  t = 0
  all_sentrecv[all_sentrecv["t"] == t][["sent","recv"]].plot.bar(logy = True)
  plt.grid(which = "both", axis = "y")
  plt.savefig(
    os.path.join(solution_folder, f"sentrecv_t{t}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  # #
  # _, axs = plt.subplots(nrows = Nf, ncols = 1, figsize = (30,2*Nf))
  # for f in range(Nf):
  #   all_sentrecv[
  #     all_sentrecv["t"] == 0
  #   ].loc[:,all_sentrecv.columns.str.startswith(f"f{f}")].plot.bar(
  #     ax = axs[f],
  #     logy = True
  #   )
  # plt.show()
