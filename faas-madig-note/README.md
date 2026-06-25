# FaaS-MADiG technical note

A focused, paper-ready LaTeX section describing **FaaS-MADiG** — the price-free
greedy-diffusion ablation of the **FaaS-MADeA** decentralized auction — in the
notation and style of `Decentralized_FaaS_coordination.pdf`.

## Files

- `faas-madig.tex` — **the deliverable**: a `\section` to be inserted into the
  paper. Concentrated on the algorithm and its rationale (no abstract/intro).
  It is **self-contained**: the `Notation and capacity model` subsection
  reproduces the relevant rows of the paper's Tables 2–3 and Eqs. (1),(2),(12),
  (13), so the note can be read without the original paper. That subsection is
  fenced by `BEGIN/END self-contained notation recap` comments — **delete it
  when inserting into the paper**, where the notation is already defined.
- `main.tex` — a minimal standalone wrapper used only to compile a preview PDF.
- `.gitignore` — ignores LaTeX build artifacts.

## Compile the preview

```bash
cd faas-madig-note
latexmk -pdf main.tex      # produces main.pdf
latexmk -c                 # clean aux files (keep main.pdf)
```

## Insert into the paper

1. Copy `faas-madig.tex` next to the paper sources and add `\input{faas-madig}`
   at the desired position (after the FaaS-MADeA solution-approach section), or
   paste its body directly.
2. Replace the plain-text cross-references (`Eq.~(7)`, `Eq.~(12)`, `Eq.~(13)`,
   `Alg.~2`, `Alg.~3`, `Alg.~4`, `(P2)`) with `\ref{}`/`\eqref{}` to the host
   paper's labels.
3. Align the latency/fairness symbols of Eq.~`\eqref{eq:madig-score}`
   (`\ell_{ij}`, `\Phi_i^f`, `w_lat`, `w_fair`) with whatever the paper uses in
   its Eq.~(7); the note derives the score as "Eq.~(7) without the `-p_j^f`
   term", so only the symbol names need matching.
4. Required packages (already in the paper for the auction algorithms):
   `amsmath`, `amssymb`, `algorithm`, `algpseudocode`.
