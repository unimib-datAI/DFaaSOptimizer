# FaaS-MAPG technical note

A paper-ready LaTeX section describing **FaaS-MAPG** (Multi-Agent Potential
Game) --- the sequential better-response coordination scheme in
`decentralized_potentialgame.py` --- as an exact potential game with a
certified $\varepsilon$-Nash equilibrium stopping condition, in the notation
of `Decentralized_FaaS_coordination.pdf`.

## Files
- `faas-mapg.tex` --- the `\section{}` to `\input{}` (or paste) into the
  paper. Covers the game formulation, the exact-potential proposition, the
  proposal-MILP + greedy-split + $\varepsilon$-acceptance move rule, the
  finite-termination / equilibrium-certificate theorem (explicitly scoped to
  the implemented move class), the reused memory market, the S/R variants,
  the three stopping reasons, and positioning against FaaS-MABR, FaaS-MALD,
  and the potential-game literature. Remove the self-contained "Notation and
  capacity model" subsection on insertion.
- `main.tex` --- standalone preview wrapper.
- `references.bib` --- cited works: Monderer & Shapley 1996 (potential
  games), Rosenthal 1973 (congestion games), plus the auction/dual-
  decomposition references shared with the FaaS-MADeA/FaaS-MALD notes.
- `.gitignore` --- LaTeX build artifacts.

## Build a preview
```bash
cd faas-mapg-note
latexmk -pdf main.tex
```

## Insert into the paper
1. `\input{faas-mapg}` (or paste the section).
2. Delete the "Notation and capacity model" subsection.
3. Convert plain-text cross-references to the host paper's `\ref{}` labels.
4. Merge `references.bib` into the paper's bibliography.
5. The section requires `amsmath`, `amssymb`, `amsthm` (for the
   `proposition`/`theorem`/`proof` environments), `booktabs`, `algorithm`,
   `algpseudocode`, and `natbib`; define the `proposition`/`theorem`
   environments with `\newtheorem` if the host paper does not already have
   them (or rename to match the paper's own theorem environments).
