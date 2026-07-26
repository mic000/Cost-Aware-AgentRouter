# Cost-Aware AgentRouter (Project 13)

**Course:** Summer 2026 CSC 504 & SENG 404
**Team:** Ziming Dong, Nhan Huynh, Ming Chen, Jason Thomo

An offline study of whether the choice of AI coding agent (Codex, Devin, Copilot,
Cursor, Claude Code) for a GitHub task can be made better, framed as a **cost-aware
router**. A single XGBoost regressor learns `f(task features, agent) → success`;
routing scores all five agents for a task and picks either the success-only choice
`argmax_a P(success | x, a)` or a cost-aware choice
`argmax_a [ P(success | x, a) − λ · cost(a) ]`.

**Headline finding (see [interim_report.md](interim_report.md)):** on success alone,
no learned router beats *always-best-agent* (Codex) — and that baseline is largely a
selection-bias artifact (always-best ≈ always-most-popular, both route ~99% to Codex).
The router's value is confined to the **cost–quality frontier**, where a
*cell cost-gap* router keeps ~93% of always-Codex's success at ~29% lower cost.

> **Scope note.** AIDev records a single observed agent per PR — only ~5 issues are
> attempted by two or more agents — so there is no per-issue counterfactual. The
> router is trained and evaluated *within the observed (issue, agent) distribution*
> using a held-out subgroup direct-method estimator; we make no causal or
> counterfactual optimality claim.

See [proj_proposal.md](proj_proposal.md) for the original study design and
[ethics.md](ethics.md) for the data-ethics statement (Menlo Report framing).

## Data

We use the [`hao-li/AIDev`](https://huggingface.co/datasets/hao-li/AIDev) dataset,
read from `hf://datasets/hao-li/AIDev`. The full-dump tables are read via **DuckDB**
because `all_pull_request.parquet` trips a pyarrow repetition-level bug. No raw PR
text, usernames, or identifiable repository metadata are redistributed here — only
derived/aggregate artifacts under `data/` and figures under `fig/`.

## Repository structure

| Path | Purpose |
|---|---|
| `00_inspect_data.py` | EDA / schema sanity-check over the raw AIDev tables. |
| `01_clean_transform.py` | Cleans `pull_request` + `repository`, derives `outcome`/`merged`, writes `data/01_clean_parquet`. |
| `02_undersampling.py` | Down-samples the dominant Codex class to balance the agent distribution. |
| `build_router_dataset.py` | Builds the balanced 25,580-PR router corpus from the **full** dump (5,116/agent), with leakage-safe title/body cleaning → `data/router_dataset.jsonl`. |
| `05_router_pretreatment.ipynb` | The reduced-leakage (v2) router pipeline: features, XGBoost model, routing. |
| `make_figures.py` | Regenerates Figures 1–6 and the reported numbers (single source of truth for the report). |
| `interim_report.md` | **Interim report** — methods, findings, discussion. |
| `proj_proposal.md` | Original project proposal. |
| `ethics.md` | Data-ethics statement. |
| `data/`, `fig/` | Derived artifacts and generated figures. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Build the balanced router corpus from the full AIDev dump (reads HF via DuckDB)
python build_router_dataset.py          # -> data/router_dataset.jsonl

# 2. Run the reduced-leakage router pipeline
jupyter notebook 05_router_pretreatment.ipynb

# 3. Regenerate all figures and the reported numbers
python make_figures.py                  # -> fig/fig1..6 + printed metrics
```

`make_figures.py` also needs the precomputed title embeddings
(`data/emb_title_all-MiniLM-L6-v2_25580.npy`); Figure 5 additionally reads the AIDev
`pr_reviews` table and is skipped gracefully if it is unavailable.

The earlier exploratory scripts (`00`–`02`) operate on the smaller cleaned subset and
pop up matplotlib windows summarizing the agent × outcome distribution; close each
window to let the script continue.

## Status

Dataset build, the reduced-leakage v2 model, the held-out-cell off-policy estimator,
the cost-aware frontier, and Figures 1–6 are implemented and reported in the interim
report. Planned next: time-based split with IPW + bootstrap CIs, a censored/binary
treatment of `still_open`, a cost-budget sensitivity sweep, and quality labels beyond
merge (review friction, reverts, tests). See [interim_report.md](interim_report.md#L335).
