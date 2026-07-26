"""Experiment #3 (exploratory, NOT yet wired into the report).

Question: under a review-quality success metric instead of raw merge, do the
cost-dominated middle agents (especially Cursor, priced same as Codex) re-enter
the cost-success Pareto set?

Caveat: review coverage is sparse (~11% of pop PRs are reviewed, and only ~3.9%
of our balanced corpus), so this is an AGGREGATE per-agent comparison on the full
pop set, not a per-task quality router (the data cannot support cell-level quality
estimates). Writes exp3_quality_results.md only.
"""
import json
import numpy as np
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download

COST = {"Copilot": 1.10, "OpenAI_Codex": 3.85, "Cursor": 3.85, "Claude_Code": 4.80, "Devin": 4.50}
AG = list(COST)

pp = hf_hub_download("hao-li/AIDev", "pull_request.parquet", repo_type="dataset")
rr = hf_hub_download("hao-li/AIDev", "pr_reviews.parquet", repo_type="dataset")
con = duckdb.connect()
con.execute(f"""CREATE VIEW rv AS SELECT pr_id,
  count(*) FILTER (WHERE state='CHANGES_REQUESTED') n_cr, count(*) n_rev
  FROM read_parquet('{rr}') GROUP BY pr_id""")
agents_sql = ",".join("'" + a + "'" for a in AG)
q = con.execute(f"""SELECT p.agent, (p.merged_at IS NOT NULL) merged,
  coalesce(v.n_cr,0) n_cr, coalesce(v.n_rev,0) n_rev
  FROM read_parquet('{pp}') p LEFT JOIN rv v ON p.id=v.pr_id
  WHERE p.agent IN ({agents_sql})""").df()

rows = {}
for a, g in q.groupby("agent"):
    rev = g[g.n_rev > 0]
    rows[a] = {
        "n": int(len(g)),
        "cost": COST[a],
        "merge": float(g.merged.mean()),
        "reviewed_frac": float((g.n_rev > 0).mean()),
        # quality metric 1: among reviewed PRs, fraction merged with no changes requested
        "clean_among_reviewed": float(((rev.merged) & (rev.n_cr == 0)).mean()) if len(rev) else np.nan,
        # quality metric 2: over ALL PRs, fraction that merged AND passed review with no changes (penalizes self-merge)
        "reviewed_clean_all": float(((g.n_rev > 0) & g.merged & (g.n_cr == 0)).mean()),
    }
T = pd.DataFrame(rows).T[["n", "cost", "merge", "reviewed_frac", "clean_among_reviewed", "reviewed_clean_all"]]


def pareto(col):
    s = T[col].astype(float)
    return [a for a in T.index if not any(
        b != a and s[b] >= s[a] and T.cost[b] <= T.cost[a]
        and (s[b] > s[a] or T.cost[b] < T.cost[a]) for b in T.index)]


metrics = ["merge", "clean_among_reviewed", "reviewed_clean_all"]
pareto_sets = {m: pareto(m) for m in metrics}
spd = {m: {a: round(float(T[m][a]) / T.cost[a], 3) for a in AG} for m in metrics}

print(T.round(3).to_string())
print("\nPareto-efficient agents under each success metric:")
for m in metrics:
    print(f"  {m:22} -> {pareto_sets[m]}")
print("\nsuccess-per-dollar by metric:")
for m in metrics:
    print(f"  {m:22} -> {spd[m]}")
cursor_back = any("Cursor" in pareto_sets[m] for m in ["clean_among_reviewed", "reviewed_clean_all"])
print(f"\n=> Cursor re-enters the Pareto set under a quality metric: {cursor_back}")

with open("exp3_quality_results.md", "w") as f:
    f.write("# Experiment #3: quality-metric Pareto (exploratory, not in report)\n\n")
    f.write("Per-agent (full pop set, reviewed where applicable):\n\n```\n")
    f.write(T.round(3).to_string())
    f.write("\n```\n\n")
    f.write("```\n" + json.dumps({
        "pareto_by_metric": pareto_sets,
        "success_per_dollar_by_metric": spd,
        "cursor_reenters_under_quality": bool(cursor_back),
        "coverage_note": "reviewed_frac is small and agent-dependent; quality metrics are descriptive, not a router",
    }, indent=2) + "\n```\n")
print("\nwrote exp3_quality_results.md")
