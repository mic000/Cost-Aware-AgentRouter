import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

out_path = Path("data/01_clean_parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)

df_req = pd.read_parquet("hf://datasets/hao-li/AIDev/pull_request.parquet"); df_req.info()
df_req["agent"].value_counts().plot(kind="bar", title="agent Distribution")
plt.show()
df_repo = pd.read_parquet("hf://datasets/hao-li/AIDev/repository.parquet"); df_repo.info()
print("pull request: \n", df_req.isna().any()[df_req.isna().any()])
print("repository: \n", df_repo.isna().any()[df_repo.isna().any()])

DROP_PR   = ["repo_url", "html_url"]
DROP_REPO = ["url", "license", "full_name"]
df_req   = df_req.drop(columns=[c for c in DROP_PR   if c in df_req.columns])
df_repo = df_repo.drop(columns=[c for c in DROP_REPO if c in df_repo.columns])

for col in ["created_at", "closed_at", "merged_at"]:
    df_req[col] = pd.to_datetime(df_req[col], utc=True, errors="coerce")

before = len(df_req)
df_req = df_req[df_req["body"].notna() & (df_req["body"].str.strip() != "")]
after  = len(df_req)

df_repo = df_repo.rename(columns={"id": "repo_id"})
df = df_req.merge(df_repo[["repo_id", "language", "forks", "stars"]],
    on="repo_id", how="left", validate="many_to_one")

print(sum(df["language"].isna()))
df["language"] = df["language"].fillna("Unknown")
print(f"join coverage: {df['language'].ne('Unknown').mean()*100:.1f}% PRs have a language")

df["outcome"] = np.select(
    condlist=[
        (df["state"] == "closed") & df["merged_at"].notna(),
        (df["state"] == "closed") & df["merged_at"].isna(),
        (df["state"] == "open"),
    ],
    choicelist=["merged", "closed_unmerged", "still_open"],
    default="unknown",
)
df["merged"] = (df["outcome"] == "merged").astype(int)
df.to_parquet(out_path)
print(f"\nfinal shape: {df.shape}")
print(df[["agent", "outcome"]].value_counts())


############################################################
agent_outcome_counts = (
    df.groupby(["agent", "outcome"])
      .size()
      .unstack(fill_value=0)
)
# make sure these three columns exist
for col in ["merged", "closed_unmerged", "still_open"]:
    if col not in agent_outcome_counts.columns:
        agent_outcome_counts[col] = 0

# add total PR count for each agent
agent_outcome_counts["total_prs"] = agent_outcome_counts.sum(axis=1)
# reorder columns
plot_df = agent_outcome_counts[
    ["total_prs", "merged", "closed_unmerged", "still_open"]
].sort_values("total_prs", ascending=False)

print(plot_df)
# plot grouped bar chart
ax = plot_df.plot(
    kind="bar",
    figsize=(18, 8),
    color=["gray", "green", "orange", "red"],
    width=0.8
)
plt.title("Agent Distribution and Outcome Distribution")
plt.xlabel("Agent")
plt.ylabel("Number of Pull Requests")
plt.xticks(rotation=45, ha="right")
plt.legend(
    title="Distribution Type",
    labels=["Agent total PRs", "Merged", "Closed unmerged", "Still open"]
)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%d",
        label_type="edge",
        fontsize=9,
        padding=3
    )

plt.ylim(0, plot_df.max().max() * 1.15)
plt.tight_layout()
plt.show()
