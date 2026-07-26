# Proposal review — Option C applied

> **Status: RESOLVED — Option C (hybrid) was chosen and applied to
> [proj_proposal.md](proj_proposal.md).** This file is kept as the record of what changed and why.
> The only item still needing the main contributor is the dangling `image2` figure (item 11).

**What Option C did:** *retrofit* the items that would read as errors next to the interim report
(model type, metrics, dataset, cost values, imbalance method, the PR-vs-issue proxy, temporal-holdout
status, encoder names), and *leave the genuine scope-evolution as forecast* with a one-line
"*(Revised in the interim report: …)*" pointer (the λ-grid, the 5pp prediction, Stage 2 triage, and
SHAP). The reasoning: a proposal is a forecast written before the work, so scope that genuinely
evolved should stay as forecast-with-a-pointer, while plain factual mismatches should be corrected.

Resolution per item (details below):

| # | Item | Resolution |
|---|---|---|
| 1 | Model type classifier→regressor | **Retrofitted** (RQ2, §4, §5, §6 W2) |
| 2 | Metrics Macro-F1/AUC→MAE + policy value | **Retrofitted** (RQ2, RQ3, §5, §8) |
| 3 | Dataset ~33k pop→25,580 full dump | **Retrofitted** (§4, §6 W1, §6 Tools) |
| 4 | λ-grid → budget sweep | **Forecast + pointer** (RQ4) |
| 5 | Cost values $0.02–$2 → $1.10–$4.80 | **Retrofitted** (§5 cost model) |
| 6 | Imbalance class-weighting → down-sampling | **Retrofitted** (§4, §7) |
| 7 | 5pp prediction falsified | **Recast as hypothesis + pointer** (§8.2) |
| 8 | SHAP not delivered | **Forecast + pointer** (§8.3) |
| 9 | Stage 2 triage orphaned | **Forecast + pointer** (§1 fixed earlier; §5 pointer added) |
| 10 | Issue-vs-PR proxy | **Retrofitted** (§5 Stage 1) |
| 11 | Temporal holdout / encoders / `image2` | Holdout→"planned"; encoders updated; **`image2` still needs your caption** |

The original analysis for each item is preserved below for reference.

---

## Decisions needed (forecast vs. implemented)

Each item lists what the proposal says, what was actually built, and the decision.

### 1. Model type: classifier → regressor  *(high impact, pervasive)*
- **Proposal:** "classifier (Random Forest / XGBoost)" — RQ2 (§3), §4 preprocessing, §5 Stage 1,
  §6 timeline W2.
- **Built:** a single **XGBoost regressor** predicting a continuous success target
  `{merged 1.0, still_open 0.3, closed 0.0}` ([build_router_dataset.py](build_router_dataset.py)
  line 26; interim §3.2, §3.4). Random Forest was never used.
- **Decision:** retrofit "classifier → regressor", drop Random Forest, and state the regression
  target? Or leave as forecast?

### 2. Metrics: Macro-F1 / AUC → MAE + policy value  *(high impact, pervasive)*
- **Proposal:** Macro-F1 and AUC throughout — RQ2, RQ3, §4, §5 ("Primary metrics are Macro-F1,
  AUC"), §6 W2.
- **Built:** **MAE** for prediction quality and **success-at-cost** for routing
  (interim §4.2–4.3). Macro-F1/AUC never reported. (Note: AUC is natively binary anyway, and
  Macro-F1 is arguably the wrong objective for a *routing* problem, where policy value — success at
  a given cost — is what matters.)
- **Decision:** retrofit the metric framing to MAE + policy value? This is the single most
  pervasive divergence and the one most likely to read as an error if left.

### 3. Dataset: ~33k pop subset → 25,580 from the full dump  *(high impact)*
- **Proposal:** "AIDev-pop subset (~33k PRs)" — §4, and the 33k figure is baked into the timeline
  (W1) and §6 Tools & Compute.
- **Built:** the pop subset was **explicitly rejected** (too few minority-agent PRs:
  Claude_Code=459, Cursor=1541) in favor of a balanced **25,580-PR** corpus drawn from the full
  ~932k dump ([build_router_dataset.py](build_router_dataset.py) lines 4–6; interim §3.1).
- **Decision:** retrofit to "balanced 25,580-PR corpus from the full ~932k dump" with the
  one-sentence rationale? (Recommended even under Option C — "~33k pop" otherwise looks like a
  factual error.)

### 4. λ-sweep: five-agent grid → Codex→Copilot budget sweep  *(medium impact)*
- **Proposal:** RQ4 and §5 promise a literal grid, λ ∈ {0, 0.1, 0.5, 1, 5, 10}.
- **Built:** the λ objective is described as "conceptual"; the reported frontier is a
  **Codex-to-Copilot budget sweep**, because Copilot is the only agent cheaper than Codex
  (interim §3.4, §4.3).
- **Decision:** keep the λ grid as the design and add a note that it collapses to a budget sweep in
  practice, or replace it with the budget-fraction sweep that was actually run?

### 5. Cost model values: $0.02–$2 → $1.10–$4.80  *(medium impact)*
- **Proposal:** §5 says costs range from ~$0.02 (Copilot) to ~$2 (Devin).
- **Built:** a normalized per-PR scenario of Copilot $1.10, Codex/Cursor $3.85, Devin $4.50,
  Claude $4.80 (interim §3.5) — Devin is near the *top* of the range, not the cheap end, and the
  absolute values are ~2× higher. The proposal's cost bullet also mixes denominators ("per
  completion-equivalent" vs "per task" vs "amortised per-PR") and never says how flat-rate
  subscription pricing converts to a marginal per-PR number.
- **Decision:** replace the numbers with the implemented scenario (or relabel them "illustrative"),
  and state the subscription→per-PR conversion assumption?

### 6. Imbalance handling: class weighting → down-sampling  *(medium impact)*
- **Proposal:** §4 and §7 ("Data Imbalance") say class weighting + Macro-F1.
- **Built:** **down-sampling** each agent to the minimum count
  ([build_router_dataset.py](build_router_dataset.py) lines 129–136). The interim report also notes
  the per-agent base rate *is* the routing signal, so balancing trades against that signal — a
  tension the proposal never examines.
- **Decision:** swap to down-sampling (and note the base-rate-vs-balancing tension), or present both?

### 7. Expected Result §8.2 — the 5pp prediction was falsified  *(medium impact)*
- **Proposal:** predicts "at least a 5 pp improvement … dominating both baselines."
- **Built:** the headline finding is the **opposite** — no learned router beats always-best-agent on
  success; the value is only on the cost axis (interim §1, §4.3, §4.5). The selection-bias framing
  in §8.2 was already softened in an earlier pass, but the confident "≥5 pp / dominating" claim
  remains.
- **Decision:** recast §8.2 as a hypothesis and allow for the strong-baseline outcome? (For a
  proposal a forecast that turned out wrong is acceptable, but the confident phrasing jars next to
  the interim result.)

### 8. SHAP interpretability — RQ-less and not delivered  *(medium impact)*
- **Proposal:** §6 W4 and §8.3 call SHAP "the main practical contribution," but no RQ covers
  interpretability.
- **Built:** no SHAP analysis; the interpretability finding is the MAE ablation showing task text
  contributes little (interim §4.2).
- **Decision:** give interpretability its own RQ and keep SHAP as planned, or replace the SHAP claim
  with the ablation-based finding?

### 9. Stage 2 (PR Triage) — orphaned design  *(medium impact; the §1/§5 wording was fixed, the design question wasn't)*
- The §1↔§5 wording contradiction is fixed (both now say Stage 1 = router incl. selection, Stage 2 =
  triage). But the deeper problem stands: **the triage stage has no RQ, no Expected Result, and —
  since it runs after a PR exists — cannot inform the pre-action routing decision.** The interim
  report does not build it (the closest artifact is the descriptive review-friction analysis in
  §4.4).
- **Decision:** (a) give Stage 2 its own RQ + Expected Result and explain how its output is used,
  (b) cut it, or (c) keep it as forecast with a "revised in interim report" pointer. Recommend (b)
  or (c).

### 10. "Issue" vs "PR" / post-treatment proxy  *(medium impact; conceptual)*
- The proposal routes "GitHub issues" and talks about "issue text" as a routing-time feature. In
  reality the data is **PR-level**, and the task proxy is the (cleaned) **PR title**, which is
  agent-authored and therefore post-treatment — a point the interim report had to confront head-on
  (interim §3.3; [build_router_dataset.py](build_router_dataset.py) lines 8–10).
- **Decision:** add a sentence acknowledging that issue text is not available at routing time and
  that the PR title is a reduced-leakage proxy? (Worth doing under any option — it's central to the
  design's validity.)

### 11. Minor leftover divergences
- **Temporal holdout** (§5 "hold out the most recent calendar month") is listed as a method but is
  *planned, not done* in the interim report (§3.6, §6). Downgrade to "planned"?
- **Encoder names**: proposal lists CodeBERT / UniXcoder / bge-code; the implementation used
  MiniLM / mpnet / st-codesearch (interim §4.2). Update if you want them to match.
- **`image2`** (referenced in §6, just after Tools & Compute) is a **dangling, uncaptioned figure**.
  Only you know what it depicts — please add a caption or remove the reference. I left it untouched
  rather than guess.

---

## Already fixed (mechanical, direction-neutral — no decision needed)

These were applied directly because they are internal-consistency or formatting defects, not
forecast-vs-reality calls:

1. **§1 Stage 2 contradiction** — §1 said "Stage 2 maximises a cost-adjusted utility" while §5
   defined Stage 2 as PR triage. §1 now matches §5 (Stage 1 = router incl. cost-aware selection;
   Stage 2 = triage). *(The deeper design question is item 9 above.)*
2. **§5 pricing dates** — removed the stray "as of 2026-05" that conflicted with the two stated
   snapshots (2024-Q4 / 2026-Q2).
3. **RQ1 parenthetical** — it listed four task-category axes but the stated z-test only covers
   language × agent; reworded so the confirmatory test (language) and descriptive axes are
   distinguished.
4. **§6 "Tools & Compute"** — split the ~1,900-character single paragraph into paragraphs with the
   encoder contingencies as a bullet list. Wording (incl. the stale "~33k", "Random Forest",
   "Macro-F1" mentions) was preserved verbatim — those are decisions 1–3 above.
5. **§7 limitation bullets** — converted the non-Markdown `•`+tab glyphs to standard Markdown list
   syntax.

---

*Credit: the divergence analysis was produced by an independent review pass over the proposal,
interim report, and dataset-build script.*
