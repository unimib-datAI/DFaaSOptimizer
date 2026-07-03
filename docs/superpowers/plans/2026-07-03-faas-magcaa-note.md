# FaaS-MAGCAA Technical Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an English, paper-ready and standalone-compilable LaTeX technical note that documents the implemented FaaS-MAGCAA algorithm precisely.

**Architecture:** Add a self-contained `faas-magcaa-note/` directory matching `faas-mapg-note`. The main section will separate the original GCAA result from the FRALB adaptation, derive only properties supported by the implementation, and include pseudocode that matches `decentralized_gcaa.py` after the no-ping-pong fix.

**Tech Stack:** LaTeX, BibTeX, `latexmk`, `algorithm`, `algpseudocode`, `amsmath`, `amssymb`, `amsthm`, `booktabs`, `natbib`.

---

### Task 1: Verified bibliography and note scaffold

**Files:**
- Create: `faas-magcaa-note/references.bib`
- Create: `faas-magcaa-note/main.tex`
- Create: `faas-magcaa-note/.gitignore`

- [ ] **Step 1: Add the primary GCAA reference**

Add a BibTeX entry with the verified metadata: Martin Braquet and Efstathios Bakolas, “Greedy Decentralized Auction-based Task Allocation for Multi-Agent Systems,” `IFAC-PapersOnLine`, volume 54, issue 20, pages 675--680, 2021, DOI `10.1016/j.ifacol.2021.11.249`.

- [ ] **Step 2: Add only positioning references actually cited**

Reuse the verified Bertsekas auction references already present in `faas-mapg-note/references.bib`. Do not add uncited or secondary sources.

- [ ] **Step 3: Create the standalone wrapper**

Create `main.tex` with `article`, UTF-8 input, `amsmath`, `amssymb`, `amsthm`, `booktabs`, `algorithm`, `algpseudocode`, numerical `natbib`, `geometry`, the theorem environments used by the note, `\input{faas-magcaa}`, and the local bibliography.

- [ ] **Step 4: Ignore LaTeX build products**

Create `.gitignore` covering `*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.log`, `*.out`, and `main.pdf`.

- [ ] **Step 5: Commit the scaffold**

```bash
git add faas-magcaa-note/main.tex faas-magcaa-note/references.bib faas-magcaa-note/.gitignore
git commit -m "scaffold FaaS-MAGCAA technical note"
```

### Task 2: Paper-ready algorithm section

**Files:**
- Create: `faas-magcaa-note/faas-magcaa.tex`

- [ ] **Step 1: Write the introduction and positioning**

Explain that FaaS-MAGCAA adapts the GCAA selection-and-consensus mechanism of Braquet and Bakolas to FRALB as an experimental baseline. State explicitly that it has no adaptive prices and no replica market, unlike FaaS-MADeA, and does not claim the equilibrium certificates of FaaS-MALD or FaaS-MAPG.

- [ ] **Step 2: Write the removable notation recap**

Define nodes `\mathcal N`, functions `\mathcal F`, residual buyer load `\omega_i^f`, routing `y_{ij}^f`, seller residual capacity `C_j^f`, neighbourhood `N_i`, utility `u_{ij}^f`, and the sending/receiving predicates. Include the capacity equation

```latex
C_j^f(h)=\max\!\left\{0,\frac{r_j^f(h)U_{\max}^f}{D_j^f}-x_j^f(h)-\sum_i y_{ij}^f(h)\right\}.
```

and the implemented utility

```latex
u_{ij}^f(h)=\beta_{ij}^f-w_L\ell_{ij}-w_F q_i^f(h),
```

noting that repository configurations set `w_L=w_F=0`, so bids reduce to `\beta_{ij}^f`.

- [ ] **Step 3: Describe the GCAA-to-FRALB mapping**

State that each active buyer--function pair `(i,f)` is a GCAA agent and each feasible seller--function pair `(j,f)` is a task. Unit bids are mandatory. A null assignment occurs when no convenient, capacity-feasible, no-ping-pong-safe seller exists.

- [ ] **Step 4: Formalize proposal and consensus**

Define the best proposal as the maximum-utility candidate after capacity and neighbourhood filtering. For every contested `(j,f)`, select at most one valid winner, test `C_j^f\ge1`, and update the within-round sending/receiving state immediately. Explain that a lower-ranked contender may win only when a higher-ranked proposal is invalid under no-ping-pong or capacity constraints.

- [ ] **Step 5: Add implementation-faithful pseudocode**

Provide one algorithm covering initial local solves, zero price vector, disabled replica sellers, repeated `define_bids`, `resolve_gcaa_round`, restricted social-welfare re-optimization, residual-load update, and the existing stopping guards. The pseudocode must show both checks `R_i^f=0` for a buyer and `S_j^f=0` for a seller, with state updates immediately after acceptance.

- [ ] **Step 6: State and prove only supported properties**

Add propositions for: one winner per contested seller--function task per round; no capacity over-allocation from unit acceptance; and inductive preservation of no ping-pong from `y=0`. Do not claim global optimality, strategy-proofness, an equilibrium certificate, or transfer of the original paper's at-most-`n` bound.

- [ ] **Step 7: Document stopping and practical limits**

Describe residual-demand exhaustion, residual-capacity exhaustion, lack of admissible bids, maximum iterations, and time limit. State that an execution reaching `max iterations reached` remains feasible but does not carry a convergence certificate.

- [ ] **Step 8: Commit the main section**

```bash
git add faas-magcaa-note/faas-magcaa.tex
git commit -m "document FaaS-MAGCAA algorithm"
```

### Task 3: README and semantic cross-check

**Files:**
- Create: `faas-magcaa-note/README.md`
- Verify: `decentralized_gcaa.py`
- Verify: `run_faasmadea.py`
- Verify: `run_faasmacro.py`

- [ ] **Step 1: Write README integration instructions**

Describe every file, the `latexmk -pdf main.tex` preview command, insertion through `\input{faas-magcaa}`, required packages, bibliography merging, and removal of the self-contained notation recap when embedded in the host paper.

- [ ] **Step 2: Cross-check every implementation claim**

Verify the note against the actual code for: mandatory `unit_bids`; zero price; zero replica-seller vector; highest utility per `(i,f)`; at most one winner per `(j,f)`; residual-capacity comparison against `d`; `current_y` no-ping-pong guards; restricted re-optimization; and saved `LSP`/`LSPc`, objective, runtime, and termination artifacts.

- [ ] **Step 3: Scan for unsupported language**

Reject or qualify every occurrence of “optimal,” “converges,” “guarantees,” “equilibrium,” “strategy-proof,” and the original GCAA round bound unless the surrounding sentence explicitly scopes it to the cited original paper rather than FaaS-MAGCAA.

- [ ] **Step 4: Commit README and corrections**

```bash
git add faas-magcaa-note/README.md faas-magcaa-note/faas-magcaa.tex
git commit -m "complete FaaS-MAGCAA note guidance"
```

### Task 4: Compile and verify the deliverable

**Files:**
- Verify: `faas-magcaa-note/main.tex`
- Verify: `faas-magcaa-note/faas-magcaa.tex`
- Verify: `faas-magcaa-note/references.bib`

- [ ] **Step 1: Compile twice through latexmk**

Run:

```bash
cd faas-magcaa-note
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 and `main.pdf` generated.

- [ ] **Step 2: Check warnings**

Run:

```bash
rg -n "Undefined|Citation.*undefined|Reference.*undefined|LaTeX Error|Overfull" main.log
```

Expected: no undefined citations/references, LaTeX errors, or material overfull boxes. Fix prose or tables if any are reported and rebuild.

- [ ] **Step 3: Perform final source checks**

Run:

```bash
rg -n "TODO|TBD|PLACEHOLDER|\\?\\?\\?" README.md faas-magcaa.tex main.tex references.bib
git diff --check
```

Expected: no placeholders and no whitespace errors.

- [ ] **Step 4: Commit compilation fixes**

```bash
git add faas-magcaa-note
git commit -m "verify FaaS-MAGCAA LaTeX note"
```
