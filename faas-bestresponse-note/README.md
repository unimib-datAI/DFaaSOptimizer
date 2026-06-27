# FaaS-MABR note

A paper-ready LaTeX section explaining the **FaaS-MABR** family --- three
sequential (Gauss-Seidel) best-response heuristics (S: fixed-order greedy, R:
randomized-order greedy, O: capped local best response) --- in the notation of
`Decentralized_FaaS_coordination.pdf`.

## Files
- `faas-mabr.tex` --- the `\section{}` to `\input{}` (or paste) into the paper.
  Remove the self-contained "Notation and capacity model" subsection on
  insertion.
- `main.tex` --- standalone preview wrapper.
- `references.bib` --- cited works (Bertsekas & Tsitsiklis 1989 Gauss-Seidel
  anchor; Cybenko diffusion; Bertsekas auction).
- `.gitignore` --- LaTeX build artifacts.

## Build a preview
```bash
cd faas-bestresponse-note
latexmk -pdf main.tex
```

## Insert into the paper
1. `\input{faas-mabr}` (or paste the section).
2. Delete the "Notation and capacity model" subsection.
3. Convert plain-text cross-references to the host paper's `\ref{}` labels.
4. Merge `references.bib` into the paper's bibliography.
