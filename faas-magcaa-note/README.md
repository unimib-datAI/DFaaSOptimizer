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

The implementation saves final `LSP` and `LSPc` solution outputs and writes
`obj.csv`, `termination_condition.csv`, and `runtime.csv`. It also writes
periodic `LSP`/`LSPc` checkpoints according to `checkpoint_interval`; these are
separate from the final outputs. Plot files are optional and depend on the run
configuration and problem size.
