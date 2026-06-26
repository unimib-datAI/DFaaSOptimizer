# FaaS-MAPoD note

A paper-ready LaTeX section explaining **FaaS-MAPoD**, the randomized
power-of-d-choices sibling of FaaS-MADiG, in the notation of
`Decentralized_FaaS_coordination.pdf`.

## Files
- `faas-mapod.tex` — the `\section{}` to `\input{}` (or paste) into the paper.
  Remove the self-contained "Notation and capacity model" subsection on
  insertion (the host paper already defines that notation and those equations).
- `main.tex` — standalone preview wrapper (compile this to review the note).
- `references.bib` — cited works (shared, already-verified set; Mitzenmacher is
  the direct basis).
- `.gitignore` — LaTeX build artifacts.

## Build a preview
```bash
cd faas-mapod-note
latexmk -pdf main.tex
```

## Insert into the paper
1. `\input{faas-mapod}` (or paste the section).
2. Delete the "Notation and capacity model" subsection.
3. Convert the plain-text cross-references ("Eq.~(7)", "Alg.~4",
   "Section~\ref{sec:madig-greedy}") to the host paper's `\ref{}` labels.
4. Merge `references.bib` into the paper's bibliography (or re-key the
   `\citep{}` commands).
