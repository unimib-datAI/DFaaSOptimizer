# FaaS-MALD technical note

A focused, paper-ready LaTeX section describing **FaaS-MALD**, the
Lagrangian-dual coordination method with a primal--dual optimality certificate,
in the notation and style of `Decentralized_FaaS_coordination.pdf`.

## Files

- `faas-mald.tex` — **the deliverable**: a self-contained `\section` covering
  the transportation LP, dual decomposition, primal recovery, certificate,
  properties, and literature positioning. Delete the fenced notation recap
  when inserting it into a paper that already defines the notation.
- `references.bib` — bibliography for the literature-positioning subsection.
- `main.tex` — minimal standalone preview wrapper.
- `.gitignore` — ignores LaTeX build artifacts and the preview PDF.

## Compile the preview

```bash
cd faas-mald-note
latexmk -pdf main.tex      # produces main.pdf
latexmk -c                 # clean aux files (keep main.pdf)
```

## Insert into the paper

Copy `faas-mald.tex` next to the paper sources and add `\input{faas-mald}` at
the desired position. Merge `references.bib` into the paper bibliography and
replace standalone equation and algorithm references with the host paper's
labels. The inserted section requires `amsmath`, `amssymb`, `algorithm`,
`algpseudocode`, `booktabs`, and `natbib`. The standalone `main.tex` wrapper
additionally loads `geometry` for preview margins and `hyperref` for PDF links;
those two packages are wrapper conveniences rather than host-section
requirements.
