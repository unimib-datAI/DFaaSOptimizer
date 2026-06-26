# Verification Report: Modern Cited Works

Generated: 2026-06-26

---

## 1. `nezami2021` — Nezami et al., IEEE Access 2021

### Existence
**Confirmed.** Published in IEEE Access, indexed on IEEE Xplore as document 9418552. arXiv preprint 2005.00270 is the same paper (same authors, title, content).

### Metadata Verification

| Field | BibTeX value | Authoritative source | Status |
|-------|-------------|---------------------|--------|
| author | Zeinab Nezami, Kamran Zamanifar, Karim Djemame, Evangelos Pournaras | IEEE Xplore / arXiv | OK |
| title | Decentralized Edge-to-Cloud Load Balancing: Service Placement for the Internet of Things | IEEE Xplore: identical; arXiv uses "Load-balancing" (hyphenated) | OK (matches IEEE published version) |
| journal | IEEE Access | IEEE Xplore | OK |
| volume | 9 | IEEE Xplore | OK |
| year | 2021 | IEEE Xplore | OK |
| pages | (omitted) | 64983--65000 per IEEE Xplore | OK (acceptable omission, noted in task) |
| doi | 10.1109/ACCESS.2021.3074962 | IEEE Xplore | OK |

### Download Status
**Downloaded.** Local path: `faas-madig-note/cited_papers/nezami2021.pdf`
Source: arXiv (https://arxiv.org/pdf/2005.00270). Verified: PDF document, 4.5 MB.

### Core Claim
The paper proposes EPOS Fog, a decentralized multi-agent system for IoT service placement across the edge-to-cloud continuum. Agents locally generate possible assignments of requests to resources and then cooperatively select an assignment that maximizes edge utilization while minimizing service execution cost — this is collective learning-based optimization, not greedy/diffusion-style one-hop neighbour offloading. The approach reduces service execution delay up to 25% and improves load-balance up to 90% vs. First Fit and cloud-only baselines.

Source: https://arxiv.org/abs/2005.00270

---

## 2. `jiang2018` — Jiang et al., arXiv 2018

### Existence
**Confirmed.** Available on arXiv as preprint 1805.02006 (submitted 2018-05-05). Not published in a peer-reviewed journal as of this check.

### Metadata Verification

| Field | BibTeX value | Authoritative source | Status |
|-------|-------------|---------------------|--------|
| author | Weiheng Jiang, Yi Gong, Yang Cao, Xiaogang Wu, Qian Xiao | arXiv | OK |
| title | Energy-delay-cost Tradeoff for Task Offloading in Imbalanced Edge Cloud Based Computing | arXiv | OK |
| journal | arXiv preprint arXiv:1805.02006 | arXiv | OK |
| year | 2018 | arXiv | OK |

### Download Status
**Downloaded.** Local path: `faas-madig-note/cited_papers/jiang2018.pdf`
Source: arXiv (https://arxiv.org/pdf/1805.02006). Verified: PDF document, 537 KB, 36 pages.

### Core Claim
The paper addresses task offloading in an edge cloud architecture where mobile users offload tasks to shared edge cloud servers (ECS) through wireless access points. It introduces an "ECS access-cost" metric incorporating transmission delay, energy consumption, and resource usage costs, then solves NP-hard optimization problems (minimizing aggregate cost for efficiency, or minimizing maximum individual cost for fairness) via centralized and distributed heuristic algorithms. This is a centralized/hierarchical offloading model (users to shared ECS), not a peer-to-peer diffusion/gossip-style one-hop neighbour offloading scheme.

Source: https://arxiv.org/abs/1805.02006

---

## 3. `leechoi2021` — Lee & Choi, ICCBDC 2021

### Existence
**Confirmed.** Published in Proceedings of the 2021 5th International Conference on Cloud and Big Data Computing (ICCBDC), indexed in ACM Digital Library with DOI 10.1145/3481646.3481657. Also indexed on dblp.

### Metadata Verification

| Field | BibTeX value | Authoritative source | Status |
|-------|-------------|---------------------|--------|
| author | Youngsoo Lee, Sunghee Choi | ACM DL / dblp | OK |
| title | A Greedy Load Balancing Algorithm for {FaaS} Platforms | ACM DL: "A Greedy Load Balancing Algorithm for FaaS Platforms" | OK |
| booktitle | Proc.\ 2021 5th International Conference on Cloud and Big Data Computing (ICCBDC) | ACM DL: "Proceedings of the 2021 5th International Conference on Cloud and Big Data Computing" | OK (abbreviated form acceptable) |
| year | 2021 | ACM DL / dblp | OK |
| pages | (omitted) | 63--69 per dblp | MISMATCH — pages not present in BibTeX (consider adding `pages = {63--69}`) |
| doi | 10.1145/3481646.3481657 | ACM DL | OK |

### Download Status
**Paywalled.** The ACM Digital Library requires a subscription. The author's preprint was hosted at `https://get.prev.kr/papers/preprint/faas_lb.pdf` but the domain does not currently resolve (DNS failure). No other open-access copy was found.

DOI URL: https://doi.org/10.1145/3481646.3481657

### Core Claim
The paper proposes GRAF (GReedy Algorithm for FaaS platforms), a load balancing algorithm specifically designed for FaaS platforms that maximizes locality (cache-hit ratio) while preserving load balance. The algorithm uses a tabular data structure to greedily route function invocations to workers that already have warm containers/cached state, reducing cold starts. This is indeed a greedy, locality/cache-driven LB specifically for FaaS — it is not diffusion-style one-hop neighbour offloading but rather a dispatcher-level scheduling algorithm that optimizes for container reuse.

Source: https://dl.acm.org/doi/10.1145/3481646.3481657 (search result summary), https://prev.github.io/about/

---

## Summary

| Paper | Exists | Metadata | Download |
|-------|--------|----------|----------|
| `nezami2021` | Yes | All fields OK (pages omitted, acceptable) | Downloaded (`nezami2021.pdf`, 4.5 MB) |
| `jiang2018` | Yes | All fields OK | Downloaded (`jiang2018.pdf`, 537 KB) |
| `leechoi2021` | Yes | Pages missing (should be 63--69) | Paywalled; preprint host unreachable |
