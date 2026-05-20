import pandas as pd

from logs_postprocessing import parse_faasmadea_log_file


def test_parse_faasmadea_log_file_reads_sp_runtime(tmp_path):
  (tmp_path / "out.log").write_text(
    "\n".join([
      "t = 0",
      "    it = 0",
      "        sp: DONE  (ok; obj = 12.0; runtime = 2.5)",
      "        define_bids: DONE; runtime = 0.5)",
      "        evaluate_bids: DONE; runtime = 0.25)",
      "    TOTAL RUNTIME [s] = 3.25 (wallclock: 4.0)",
      "All solutions saved",
      "",
    ]),
    encoding = "utf-8",
  )

  logs_df, _ = parse_faasmadea_log_file(
    str(tmp_path),
    "exp0",
    pd.DataFrame(),
    {},
    Nn = 2,
  )

  assert logs_df.loc[0, "sp_runtime"] == 2.5
