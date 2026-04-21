


def get_current_load(
    input_requests_traces: dict, agents: list, t: int
  ) -> dict:
  incoming_load = {
    (a+1, f+1): input_requests_traces[f][a][t] \
      for a in agents for f in input_requests_traces
  }
  return incoming_load
