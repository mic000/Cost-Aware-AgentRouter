import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("data/01_clean_parquet")
out_path = Path("data/02_clean_parquet")
out_path.parent.mkdir(parents=True, exist_ok=True)

print(df["agent"].value_counts())
print("distribution of OpenAI_Codex:", df[df["agent"] == "OpenAI_Codex"]["outcome"].value_counts())

codex_name = "OpenAI_Codex"
target_agent = "Copilot"
target_n = np.ceil((df["agent"] == target_agent).sum())
codex_df = df[df["agent"] == codex_name].copy()
non_codex_df = df[df["agent"] != codex_name].copy()
print("\nOriginal Codex outcome percentage:")
print(codex_df["outcome"].value_counts(normalize=True))

codex_outcome_counts = codex_df["outcome"].value_counts()
codex_outcome_pct = codex_outcome_counts / codex_outcome_counts.sum()
target_counts_float = codex_outcome_pct * target_n
target_counts = np.floor(target_counts_float).astype(int)

print("\nTarget Codex outcome counts after downsampling:")
print(target_counts)
print("Total target Codex:", target_counts.sum())

sampled_codex_list = []
for outcome, n in target_counts.items():
    sampled_part = codex_df[codex_df["outcome"] == outcome].sample(n=n, random_state=123)
    sampled_codex_list.append(sampled_part)
codex_downsampled = pd.concat(sampled_codex_list, axis=0)
df_balanced = pd.concat([non_codex_df, codex_downsampled], axis=0)
df_balanced = df_balanced.sample(frac=1, random_state=123).reset_index(drop=True)

print("\nAfter Codex downsampling:")
print(df_balanced["agent"].value_counts())
print("\nCodex outcome counts after downsampling:")
print(df_balanced[df_balanced["agent"] == codex_name]["outcome"].value_counts())
print("\nCodex outcome percentage after downsampling:")
print(df_balanced[df_balanced["agent"] == codex_name]["outcome"].value_counts(normalize=True))

df_balanced.to_parquet(out_path, index=False)
# df_balanced.to_csv("02_clean_csv", index=False)

############################################################
agent_outcome_counts = (
    df_balanced.groupby(["agent", "outcome"])
      .size()
      .unstack(fill_value=0)
)

for col in ["merged", "closed_unmerged", "still_open"]:
    if col not in agent_outcome_counts.columns:
        agent_outcome_counts[col] = 0

agent_outcome_counts["total_prs"] = agent_outcome_counts.sum(axis=1)

plot_df = agent_outcome_counts[
    ["total_prs", "merged", "closed_unmerged", "still_open"]
].sort_values("total_prs", ascending=False)

print("\nPlot table:")
print(plot_df)

ax = plot_df.plot(
    kind="bar",
    figsize=(18, 8),
    color=["gray", "green", "orange", "red"],
    width=0.8
)

plt.title("Agent Distribution and Outcome Distribution after Codex Downsampling")
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
############################################################