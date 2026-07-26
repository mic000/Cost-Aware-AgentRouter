import pandas as pd

df_req = pd.read_parquet("hf://datasets/hao-li/AIDev/all_pull_request.parquet")
df_repo = pd.read_parquet("hf://datasets/hao-li/AIDev/all_repository.parquet")
df_user = pd.read_parquet("hf://datasets/hao-li/AIDev/all_user.parquet")

print("shape of pull_request file:", df_req.shape)
print("shape of repository file:", df_repo.shape)
print("shape of user file:", df_user.shape)
df_req.info(show_counts=True)
df_repo.info(show_counts=True)
df_user.info(show_counts=True)

print(df_req.sample(5, random_state=0))
print(df_repo.sample(5, random_state=0))
print(df_user.sample(5, random_state=0))


miss_req = (df_req.isna().mean() * 100).round(1).sort_values(ascending=False)
miss_repo = (df_repo.isna().mean() * 100).round(1).sort_values(ascending=False)
miss_user = (df_user.isna().mean() * 100).round(1).sort_values(ascending=False)
print("\nmissing of req%:\n", miss_req)
print("\nmissing of repo%:\n", miss_repo)
print("\nmissing of user%:\n", miss_user)


print("\nn_unique per column:\n", df_req.nunique().sort_values())
print("\nn_unique per column:\n", df_repo.nunique().sort_values())
print("\nn_unique per column:\n", df_user.nunique().sort_values())


for col in ["agent", "state"]:
    if col in df_req.columns:
        print(f"\n{col}:\n", df_req[col].value_counts(dropna=False))

if "body" in df_req.columns:
    print("\nbody length (chars):")
    print(df_req["body"].str.len().describe())
if "title" in df_req.columns:
    print("\ntitle length (chars):")
    print(df_req["title"].str.len().describe())