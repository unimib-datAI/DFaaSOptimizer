import run_faasmacro


def test_parse_arguments_default_v0_is_false(monkeypatch):
  monkeypatch.setattr("sys.argv", ["run_faasmacro.py"])
  args = run_faasmacro.parse_arguments()
  assert args.v0 is False


def test_parse_arguments_accepts_v0_flag(monkeypatch):
  monkeypatch.setattr("sys.argv", ["run_faasmacro.py", "--v0"])
  args = run_faasmacro.parse_arguments()
  assert args.v0 is True
