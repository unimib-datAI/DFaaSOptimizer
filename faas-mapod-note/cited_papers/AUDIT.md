# Citation Existence & Metadata Audit — FaaS-MAPoD note

Audited file: `faas-mapod-note/faas-mapod.tex` (citations) and `faas-mapod-note/references.bib` (metadata).
Scope: exactly the 5 works cited in `faas-mapod.tex` via `\citep{...}` — `bertsekas1988`, `cybenko1989`, `leechoi2021`, `mitzenmacher2001`, `willebeek1993`.
(Note: `references.bib` also contains `nezami2021`, but it is NOT cited in `faas-mapod.tex`, so it is out of scope for this audit.)
Method: re-confirmed every DOI against CrossRef (`https://api.crossref.org/works/<DOI>`) on this run; reused the prior FaaS-MADiG audit (`faas-madig-note/cited_papers/`) as baseline. PDFs copied/verified locally.
Date: 2026-06-26.

## Summary table

| Key | Exists | Metadata correct | Fix needed | PDF status |
|---|---|---|---|---|
| `bertsekas1988` | Yes | Yes | None | Downloaded (reused) — `bertsekas1988.pdf`, 3.3 MB |
| `cybenko1989` | Yes | Yes | None | Downloaded (user-provided) & content-verified — `cybenko1989.pdf` |
| `leechoi2021` | Yes | Yes | None | Downloaded (user-provided) & content-verified — `leechoi2021.pdf` |
| `mitzenmacher2001` | Yes | Yes | None | Downloaded (reused) — `mitzenmacher2001.pdf`, 242 KB |
| `willebeek1993` | Yes | Yes | None | Downloaded (user-provided) & content-verified — `willebeek1993.pdf` |

**Verdict: all 5 works exist; all metadata fields in `references.bib` are correct. No `references.bib` corrections are required.**

## Per-work confirmation

### `bertsekas1988` — Bertsekas, "The auction algorithm: A distributed relaxation method for the assignment problem"
CrossRef (DOI `10.1007/BF02186476`) returns author D. P. Bertsekas, title verbatim, journal *Annals of Operations Research*, volume 14, issue 1, pages 105–123, year 1988. Every field in the `.bib` entry (lines 38–47) matches exactly. PDF is the author copy from MIT (`https://www.mit.edu/~dimitrib/TheAuctionAP.pdf`), copied here from the prior MADiG audit; md5 verified identical to the source copy. **`.bib` entry is correct as written.**

### `cybenko1989` — Cybenko, "Dynamic load balancing for distributed memory multiprocessors"
CrossRef (DOI `10.1016/0743-7315(89)90021-X`) returns author George Cybenko, title verbatim, journal *Journal of Parallel and Distributed Computing*, volume 7, issue 2, pages 279–301, year 1989. Every field in the `.bib` entry (lines 5–14) matches exactly (CrossRef lowercases the DOI suffix to `...90021-x`; this is case-insensitive and not an error). No legitimate open-access PDF exists: the Dartmouth author URL from the prior audit (`http://www.dartmouth.edu/~gvc/Cybenko_JPDP.pdf`) still returns HTTP 404; only paywalled ScienceDirect and non-publisher mirrors (ResearchGate/Academia) were found. **`.bib` entry is correct as written.** PDF subsequently provided by the user and content-verified: page 1 reads "JOURNAL OF PARALLEL AND DISTRIBUTED COMPUTING 7, 279-301 (1989) / Dynamic Load Balancing for Distributed Memory Multiprocessors / GEORGE CYBENKO, Tufts University" — matches title, author, venue, volume, pages, year.

### `leechoi2021` — Lee & Choi, "A Greedy Load Balancing Algorithm for FaaS Platforms"
CrossRef (DOI `10.1145/3481646.3481657`) returns authors Youngsoo Lee and Sunghee Choi, title verbatim, container *2021 5th International Conference on Cloud and Big Data Computing (ICCBDC)*, pages 63–69, year 2021. The `.bib` entry (lines 59–66) matches, **and notably it already includes `pages = {63--69}`** — the missing-pages issue flagged for this key in the prior MADiG audit is already fixed in this note's `.bib`. No legitimate open-access PDF exists: the author's old preprint host `get.prev.kr` no longer resolves (DNS failure), and the author's current page (`https://prev.github.io/about/`) links only to the paywalled ACM DL and a code repo (`github.com/Prev/HotFunctions`), not a PDF. ACM `doi/fullHtml` returns HTTP 403. **`.bib` entry is correct as written.** PDF subsequently provided by the user and content-verified: page 1 reads "A Greedy Load Balancing Algorithm for FaaS Platforms / Youngsoo Lee … Sunghee Choi … KAIST" — matches title and both authors.

### `mitzenmacher2001` — Mitzenmacher, "The power of two choices in randomized load balancing"
CrossRef (DOI `10.1109/71.963420`) returns author M. Mitzenmacher, title verbatim, journal *IEEE Transactions on Parallel and Distributed Systems*, volume 12, issue 10, pages 1094–1104, year 2001. Every field in the `.bib` entry (lines 27–36) matches exactly. PDF is the author copy from Harvard EECS (`https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf`), copied here from the prior MADiG audit; md5 verified identical to the source copy. **`.bib` entry is correct as written.**

### `willebeek1993` — Willebeek-LeMair & Reeves, "Strategies for dynamic load balancing on highly parallel computers"
CrossRef (DOI `10.1109/71.243526`) returns authors M.H. Willebeek-LeMair and A.P. Reeves (the `.bib` spells out first names "Marc H." / "Anthony P." — consistent, not an error), title verbatim, journal *IEEE Transactions on Parallel and Distributed Systems*, volume 4, issue 9, pages 979–993, year 1993. Every field in the `.bib` entry (lines 16–25) matches exactly. No legitimate open-access PDF exists: only paywalled IEEE Xplore and non-publisher mirrors (ResearchGate/Academia) were found; no author copy, institutional repository, or preprint. **`.bib` entry is correct as written.** PDF subsequently provided by the user and content-verified: page 1 reads "IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS, VOL. 4, NO. 9, SEPTEMBER 1993, 979 / Strategies for Dynamic Load Balancing on Highly Parallel Computers / Marc H. Willebeek-LeMair … Anthony P. Reeves" — matches title, both authors, venue, volume, issue, page, year.

## `references.bib` corrections needed

**None.** All five in-scope entries (`bertsekas1988`, `cybenko1989`, `leechoi2021`, `mitzenmacher2001`, `willebeek1993`) are metadata-correct against CrossRef. In particular, `leechoi2021` already carries `pages = {63--69}`, so the pages gap noted in the prior MADiG audit does not apply here.

## Download tally

- Reused (copied from `faas-madig-note/cited_papers/`, md5-verified identical): `bertsekas1988.pdf`, `mitzenmacher2001.pdf` — 2 PDFs.
- User-provided (the agents found no legal open-access copy; the user supplied them via institutional/subscription access, no paywall bypassed by the agents): `cybenko1989.pdf`, `willebeek1993.pdf`, `leechoi2021.pdf` — 3 PDFs, each content-verified against its first page (title/authors/venue match the `.bib`).

**Tally: all 5 cited works now have a content-verified PDF in `cited_papers/` (2 reused + 3 user-provided).**
