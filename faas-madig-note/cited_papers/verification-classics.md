# Verification Report -- Classic References

Generated: 2026-06-26

---

## 1. `cybenko1989` -- Cybenko, "Dynamic load balancing for distributed memory multiprocessors"

### Existence

**Yes.** Confirmed via CrossRef (DOI resolves) and Semantic Scholar.

### Metadata verification

| Field | BibTeX value | Authoritative value (CrossRef) | Status |
|-------|-------------|-------------------------------|--------|
| Author | George Cybenko | George Cybenko | OK |
| Title | Dynamic load balancing for distributed memory multiprocessors | Dynamic load balancing for distributed memory multiprocessors | OK |
| Journal | Journal of Parallel and Distributed Computing | Journal of Parallel and Distributed Computing | OK |
| Volume | 7 | 7 | OK |
| Number | 2 | 2 | OK |
| Pages | 279--301 | 279-301 | OK |
| Year | 1989 | 1989 (published October 1989) | OK |
| DOI | 10.1016/0743-7315(89)90021-X | 10.1016/0743-7315(89)90021-x | OK |

All fields match.

### Download status

**Paywalled.** Semantic Scholar listed an open-access PDF at `http://www.dartmouth.edu/~gvc/Cybenko_JPDP.pdf` (GREEN status), but the URL returns HTTP 404 as of 2026-06-26. No alternative open-access copy was found. The canonical publisher URL is:

- DOI: <https://doi.org/10.1016/0743-7315(89)90021-X>
- ScienceDirect: <https://www.sciencedirect.com/science/article/abs/pii/074373158990021X>

### Core claim

Cybenko studies diffusion schemes for dynamic load balancing on message-passing multiprocessor networks, proving convergence conditions and convergence rates for arbitrary topologies via the eigenstructure of the iteration matrices; he also proposes and analyzes a "dimension exchange" method for hypercube networks and shows it is superior to the diffusion approach. Yes, the paper is about diffusion-based load balancing where overloaded nodes iteratively redistribute excess load to less-loaded neighbours.

Source: abstract reproduced in multiple citing works and confirmed via web search; DOI <https://doi.org/10.1016/0743-7315(89)90021-X>.

---

## 2. `willebeek1993` -- Willebeek-LeMair & Reeves, "Strategies for dynamic load balancing on highly parallel computers"

### Existence

**Yes.** Confirmed via CrossRef (DOI resolves) and Semantic Scholar.

### Metadata verification

| Field | BibTeX value | Authoritative value (CrossRef) | Status |
|-------|-------------|-------------------------------|--------|
| Author | Marc H. Willebeek-LeMair and Anthony P. Reeves | M.H. Willebeek-LeMair and A.P. Reeves | OK |
| Title | Strategies for dynamic load balancing on highly parallel computers | Strategies for dynamic load balancing on highly parallel computers | OK |
| Journal | IEEE Transactions on Parallel and Distributed Systems | IEEE Transactions on Parallel and Distributed Systems | OK |
| Volume | 4 | 4 | OK |
| Number | 9 | 9 | OK |
| Pages | 979--993 | 979-993 | OK |
| Year | 1993 | 1993 | OK |
| DOI | 10.1109/71.243526 | 10.1109/71.243526 | OK |

All fields match.

### Download status

**Paywalled.** Semantic Scholar reports CLOSED access. No open-access author copy, institutional repository version, or preprint was found. The canonical publisher URL is:

- DOI: <https://doi.org/10.1109/71.243526>
- IEEE Xplore: <https://ieeexplore.ieee.org/document/243526/>

### Core claim

The paper presents and experimentally compares five dynamic load balancing (DLB) strategies on a hypercube multicomputer (Intel iPSC/2): sender-initiated diffusion (SID), receiver-initiated diffusion (RID), hierarchical balancing method (HBM), gradient model (GM), and dimension exchange method (DEM). These strategies illustrate the tradeoff between knowledge (accuracy of each balancing decision) and overhead (communication/processing cost). Yes, it catalogs sender-initiated and receiver-initiated diffusion strategies among others.

Source: abstract and descriptions confirmed across multiple citing works; DOI <https://doi.org/10.1109/71.243526>.

---

## 3. `mitzenmacher2001` -- Mitzenmacher, "The power of two choices in randomized load balancing"

### Existence

**Yes.** Confirmed via CrossRef (DOI resolves) and Semantic Scholar.

### Metadata verification

| Field | BibTeX value | Authoritative value (CrossRef) | Status |
|-------|-------------|-------------------------------|--------|
| Author | Michael Mitzenmacher | M. Mitzenmacher | OK |
| Title | The power of two choices in randomized load balancing | The power of two choices in randomized load balancing | OK |
| Journal | IEEE Transactions on Parallel and Distributed Systems | IEEE Transactions on Parallel and Distributed Systems | OK |
| Volume | 12 | 12 | OK |
| Number | 10 | 10 | OK |
| Pages | 1094--1104 | 1094-1104 | OK |
| Year | 2001 | 2001 | OK |
| DOI | 10.1109/71.963420 | 10.1109/71.963420 | OK |

All fields match.

### Download status

**Downloaded.** Author copy from Harvard EECS.

- Local path: `faas-madig-note/cited_papers/mitzenmacher2001.pdf`
- Source URL: <https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf>
- Verified: PDF document, version 1.3, 11 pages, 242 KB

### Core claim

Mitzenmacher analyzes the "supermarket model" where each arriving customer samples d >= 2 servers uniformly at random and joins the shortest queue. He proves that having d = 2 choices yields an exponential improvement in expected customer time over d = 1 (random placement), while d = 3 is only a constant factor better than d = 2. Yes, this is the randomized power-of-d-choices result.

Source: author copy abstract; <https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf>.

---

## 4. `bertsekas1988` -- Bertsekas, "The auction algorithm: A distributed relaxation method for the assignment problem"

### Existence

**Yes.** Confirmed via CrossRef (DOI resolves) and Semantic Scholar.

### Metadata verification

| Field | BibTeX value | Authoritative value (CrossRef) | Status |
|-------|-------------|-------------------------------|--------|
| Author | Dimitri P. Bertsekas | D. P. Bertsekas | OK |
| Title | The auction algorithm: A distributed relaxation method for the assignment problem | The auction algorithm: A distributed relaxation method for the assignment problem | OK |
| Journal | Annals of Operations Research | Annals of Operations Research | OK |
| Volume | 14 | 14 | OK |
| Number | 1 | 1 | OK |
| Pages | 105--123 | 105-123 | OK |
| Year | 1988 | 1988 (published December 1988) | OK |
| DOI | 10.1007/BF02186476 | 10.1007/bf02186476 | OK |

All fields match.

### Download status

**Downloaded.** Author copy from Bertsekas's MIT page.

- Local path: `faas-madig-note/cited_papers/bertsekas1988.pdf`
- Source URL: <https://www.mit.edu/~dimitrib/TheAuctionAP.pdf>
- Verified: PDF document, version 1.2, 3.3 MB

### Core claim

Bertsekas proposes a massively parallelizable auction algorithm for the classical assignment problem: unassigned "persons" bid simultaneously for "objects," raising object prices; objects are then awarded to the highest bidder. The algorithm can also be interpreted as a Jacobi-like relaxation method for solving a dual problem. Setting all prices to zero and running a single bidding round reduces the auction to a greedy assignment (each person grabs its best available object), so greedy assignment is a degenerate special case of the auction mechanism.

Source: abstract and description confirmed via ASU/Springer listing and MIT-hosted PDF; DOI <https://doi.org/10.1007/BF02186476>.
