from models.sp import LSP, LSP_fixedr, LSP_capped, LSP_capped_fixedr


def test_lsp_capped_subclasses_lsp_and_adds_cap():
  m = LSP_capped()
  assert isinstance(m, LSP)
  assert m.model.component("omega_ub") is not None
  assert m.model.component("cap_offloading") is not None


def test_lsp_capped_fixedr_subclasses_fixedr_and_adds_cap():
  m = LSP_capped_fixedr()
  assert isinstance(m, LSP_fixedr)
  assert m.model.component("omega_ub") is not None
  assert m.model.component("cap_offloading") is not None
  assert m.model.component("fix_r") is not None
