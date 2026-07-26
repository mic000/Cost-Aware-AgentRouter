"""
Build a clean, agent-balanced router dataset from the FULL AIDev dump.

Why full (not the pop subset): the pop subset has too few minority-agent PRs to train a
5-agent router (Claude_Code=459, Cursor=1541). The full dump has Claude_Code=5137,
Cursor=32941, so we take all minority agents and downsample Codex to match.

Why PR title+body (not linked issue text): only ~4.9k pop PRs link an issue and 84% of them
are Copilot, so an issue-text dataset is effectively Copilot-only. PR title ("Fix X"/"Add Y")
is the most defensible task proxy available at routing time; `has_issue` is kept as a feature.

NOTE: AIDev's all_pull_request.parquet cannot be read by pyarrow 19 ("Repetition level
histogram size mismatch") -- this is an upstream parquet-format issue -- so we read via duckdb.

Output: data/router_dataset.jsonl
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from huggingface_hub import hf_hub_download

SEED = 42
AGENTS = ["OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code"]
SUCCESS_MAP = {"merged": 1.0, "still_open": 0.3, "closed_unmerged": 0.0}
OUT = Path("data/router_dataset.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


def hf(name):
    return hf_hub_download("hao-li/AIDev", f"{name}.parquet", repo_type="dataset")


# ---- load full PR + repository + issue-link flag via duckdb ----
con = duckdb.connect()
con.execute(f"CREATE VIEW pr   AS SELECT * FROM read_parquet('{hf('all_pull_request')}')")
con.execute(f"CREATE VIEW repo AS SELECT * FROM read_parquet('{hf('all_repository')}')")
con.execute(f"CREATE VIEW ri   AS SELECT DISTINCT pr_id FROM read_parquet('{hf('related_issue')}')")

df = con.execute("""
    SELECT p.id, p.number, p.title, p.body, p.agent, p.state,
           p.created_at, p.closed_at, p.merged_at, p.repo_id,
           r.language, r.forks, r.stars,
           (p.id IN (SELECT pr_id FROM ri)) AS has_issue
    FROM pr p
    LEFT JOIN repo r ON p.repo_id = r.id
    WHERE p.agent IN ('OpenAI_Codex','Copilot','Devin','Cursor','Claude_Code')
""").df()
print(f"loaded full agent PRs: {len(df):,}")

# ---- label: 3-class outcome + regression success ----
merged = df["merged_at"].notna()
df["outcome"] = np.select(
    [merged, df["state"].eq("closed") & ~merged, df["state"].eq("open")],
    ["merged", "closed_unmerged", "still_open"],
    default="unknown",
)
df = df[df["outcome"] != "unknown"].copy()
df["success"] = df["outcome"].map(SUCCESS_MAP)
df["language"] = df["language"].fillna("Unknown")

# ---- quality filter: need a repo (for grouped splits) and task text (title or body) ----
df = df[df["repo_id"].notna()].copy()          # 75 rows lack repo metadata (language=Unknown)
df["title"] = df["title"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)
df = df[df["title"].str.strip().ne("") | df["body"].str.strip().ne("")].copy()

# ---- clean body and title: strip agent footer lines, URLs, emails, agent tokens (leakage). ----
URL = re.compile(r"https?://\S+")
# Agent identity leaks heavily into the body (footers + trailers + self-references), which
# would invalidate counterfactual routing. Strip footer/trailer lines, then mask any
# residual agent-identity tokens so body_clean is leakage-safe for modeling.
FOOTER_LINE = re.compile(
    r"^.*("
    r"copilot coding agent|you can make copilot smarter|share your feedback on|"
    r"this pull request was (created|opened) by|generated (with|by)|"
    r"original prompt|requested by|run by devin|devin ai|link to devin run|"
    r"co-authored-by|"
    r"🤖|💡|💬|✨"
    r").*$",
    re.I | re.M,
)
AGENT_TOKENS = re.compile(r"\b(copilot|devin|cursor|codex|claude|openai|anthropic)\b", re.I)
def clean_body(t):
    t = FOOTER_LINE.sub(" ", str(t))     # strip footer/trailer lines
    t = URL.sub(" ", t)                  # strip URLs
    t = AGENT_TOKENS.sub(" ", t)         # mask residual agent-identity tokens (leakage)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()
df["body_clean"] = df["body"].map(clean_body)

# clean the EXPORTED title too (agent tokens / footers / emails are leakage, not issue-time text)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
def clean_title(t):
    t = FOOTER_LINE.sub(" ", str(t)); t = URL.sub(" ", t); t = EMAIL.sub(" ", t)
    t = AGENT_TOKENS.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()
df["title"] = df["title"].map(clean_title)

# ---- task_type from the CLEANED title (conventional-commit prefix, else keyword) ----
CC = re.compile(r"^\s*(feat|fix|docs|test|refactor|chore|build|ci|perf|style|revert)\b", re.I)
KW = [
    ("docs", r"\bdoc(s|umentation)?\b|readme"),
    ("test", r"\btest(s|ing|case)?\b"),
    ("ci", r"\bci\b|workflow|github action|pipeline"),
    ("build", r"\bbuild|dependenc|deps|bump|upgrade|version\b"),
    ("perf", r"\bperf|performance|optimi[sz]e|speed up\b"),
    ("refactor", r"\brefactor|cleanup|rename|simplif"),
    ("fix", r"\b(fix|bug|issue|error|crash|broken|resolve|patch)\b"),
    ("feat", r"\b(add|implement|support|introduce|new|create|enable)\b"),
]
def task_type(title):
    m = CC.match(title)
    if m:
        return m.group(1).lower()
    for t, pat in KW:
        if re.search(pat, title, re.I):
            return t
    return "other"
df["task_type"] = df["title"].map(task_type)

# ---- tidy numeric fields ----
df["stars"] = df["stars"].fillna(0).astype(int)
df["forks"] = df["forks"].fillna(0).astype(int)
df["success"] = df["success"].round(1)

# ---- balance: take min agent count, sample equally per agent ----
n_per = int(df["agent"].value_counts().min())
balanced = (
    df.groupby("agent", group_keys=False)
      .sample(n=n_per, random_state=SEED)
      .sample(frac=1, random_state=SEED)   # shuffle
      .reset_index(drop=True)
)

# raw `body` dropped to keep the file small (regenerate from this script if needed);
# `body_clean` (footer/URL-stripped) is the text column for modeling.
cols = ["id", "number", "repo_id", "agent", "title", "body_clean",
        "task_type", "language", "stars", "forks", "has_issue",
        "state", "outcome", "success", "created_at", "closed_at", "merged_at"]
balanced[cols].to_json(OUT, orient="records", lines=True,
                       force_ascii=False, date_format="iso")

# ---- summary ----
print(f"\nbalanced to {n_per:,}/agent  ->  {len(balanced):,} rows  ->  {OUT}")
print("\n=== agent x outcome (within-agent %) ===")
print((pd.crosstab(balanced["agent"], balanced["outcome"], normalize="index") * 100).round(1))
print("\n=== merge rate per agent ===")
print(balanced.groupby("agent")["success"].agg(["mean", "count"]).round(3))
print("\n=== task_type distribution ===")
print(balanced["task_type"].value_counts())
print(f"\nhas_issue: {balanced['has_issue'].mean()*100:.1f}% | "
      f"languages: {balanced['language'].nunique()} | "
      f"overall success mean: {balanced['success'].mean():.3f}")
print("\ntop languages:")
print(balanced["language"].value_counts().head(10))
