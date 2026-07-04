# FaaS-MAGCAA technical note

This directory contains a paper-ready technical note for the FaaS-MAGCAA
baseline, aligned with the implementation in `decentralized_gcaa.py`.

## Contents

- `faas-magcaa.tex`: the section to include in a paper. It contains a removable
  notation recap, the implementation mapping, pseudocode, invariants, stopping
  conditions, and positioning relative to GCAA and the other baselines.
- `main.tex`: a minimal standalone wrapper for compiling and reviewing the note.
- `references.bib`: the bibliography entries used by the standalone wrapper.
- `.gitignore`: excludes LaTeX build products, including the generated PDF.

## Build

```bash
cd faas-magcaa-note && latexmk -pdf main.tex
```

Optionally remove generated files with `latexmk -C main.tex`.

## Integrate into the paper

1. Copy the directory or place `faas-magcaa.tex` on the paper's TeX input path,
   then add `\input{faas-magcaa}` at the intended location.
2. Remove the fenced “self-contained notation recap” block when the paper
   already defines the FRALB notation and capacity model.
3. Replace section/equation/algorithm labels and plain cross-references where
   needed to match the host paper's naming and numbering conventions.
4. Merge the required entry from `references.bib` into the paper bibliography.
5. Ensure the host preamble loads `amsmath`, `amssymb`, `amsthm`, `booktabs`,
   `algorithm`, `algpseudocode`, and `natbib`, and defines the `proposition`
   theorem environment. The standalone wrapper also loads `inputenc` and
   `geometry`, but the included section does not require them specifically.

## Runtime artifacts

For each prefix `LSP` and `LSPc`, the implementation writes
`<prefix>_solution.csv`, `<prefix>_offloaded.csv`,
`<prefix>_utilization.csv`, `<prefix>_replicas.csv`,
`<prefix>_detailed_fwd_solution.csv`, and
`<prefix>_residual_capacity.csv`. It also writes `obj.csv`,
`termination_condition.csv`, `runtime.csv`, and `config.json`. The file
`out.log` is written only when `log_on_file=True`.

When `t % checkpoint_interval == 0` or `t == max_steps - 1`, component CSVs
are checkpointed under `LSP/<t>/` and `LSPc/<t>/`; no checkpoint directory is
written for other periods. The only plot artifact is `sp.png`, written only
when plotting is enabled and both `Nf <= 10` and `Nn <= 10`.
