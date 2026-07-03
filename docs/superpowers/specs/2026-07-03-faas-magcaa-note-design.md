# FaaS-MAGCAA Technical Note Design

## Goal

Create an English, paper-ready LaTeX technical note for FaaS-MAGCAA that
matches the structure and standalone usability of `faas-mapg-note`. The note
must describe the implemented algorithm exactly, distinguish inherited GCAA
ideas from FRALB-specific adaptations, and avoid unsupported convergence or
optimality claims.

## Deliverables

Create `faas-magcaa-note/` containing:

- `faas-magcaa.tex`: the section intended for `\input{}` into the paper;
- `main.tex`: a minimal standalone preview wrapper;
- `references.bib`: verified primary references;
- `README.md`: build and integration instructions;
- `.gitignore`: LaTeX build artifacts.

## Content

The main section will introduce FaaS-MAGCAA and position it against FaaS-MADeA,
FaaS-MALD, FaaS-MABR, and FaaS-MAPG. A removable notation recap will define
FRALB load, residual demand, seller capacity, utility, routing, and the
no-ping-pong constraint. The method description will then map GCAA agents and
tasks to buyer--function and seller--function pairs, formalize pure-utility
bids, and present the single-winner consensus rule.

The pseudocode will match `decentralized_gcaa.py`: unit bids, zero price
adaptation, no replica bidding, one preferred seller per buyer--function pair,
one accepted winner per seller--function pair per round, residual-capacity
checks, and dynamic sending/receiving guards that preserve no-ping-pong both
across previous allocations and within a round. It will also describe the
re-optimization and artifact-generation path reused from the existing runners.

The properties section will state only defensible results: per-round capacity
feasibility, single-winner consensus, and inductive preservation of the
no-ping-pong invariant. It will explicitly state that the original GCAA round
bound does not automatically transfer to this FRALB adaptation, whose practical
guards are maximum iterations and wall-clock time. The observed possibility of
termination by the maximum-iteration guard will not be presented as a
convergence certificate.

## References and Verification

Bibliographic metadata and algorithm claims will be checked against the primary
Braquet--Bakolas publication and other primary sources used for positioning.
The implementation-facing statements will be checked against
`decentralized_gcaa.py`, `run_faasmadea.py`, `run_faasmacro.py`, and the GCAA
tests. The standalone document must compile with `latexmk -pdf main.tex`; the
log must contain no undefined references or citations.

## Scope

No production code, configuration, or other algorithm note will be modified.
No experimental performance claims or generated plots will be added because
the request is for an algorithmic technical note and the repository does not
yet contain a controlled MAGCAA comparison dataset.
