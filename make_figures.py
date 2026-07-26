"""Reproducible figures + reported numbers for the interim report (single source of truth).
v2 (reduced-leakage) model, FLAT per-agent cost. Routing success is scored with an INDEPENDENT
validation-cell estimator (cells estimated on val), so the cell-based router (which routes using
the train-cell signal) is evaluated without circularity. Writes fig/fig1..5 and prints the numbers
used in the report. Needs data/router_dataset.jsonl + data/emb_title_all-MiniLM-L6-v2_25580.npy."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

os.makedirs("fig", exist_ok=True)
AG=["OpenAI_Codex","Copilot","Devin","Cursor","Claude_Code"]
SM={"merged":1.0,"still_open":0.3,"closed_unmerged":0.0}
COST={"Copilot":1.10,"OpenAI_Codex":3.85,"Cursor":3.85,"Claude_Code":4.80,"Devin":4.50}
NAT={"OpenAI_Codex":814522,"Copilot":50447,"Cursor":32941,"Devin":29744,"Claude_Code":5137}
SEED=42; short={"OpenAI_Codex":"Codex","Claude_Code":"Claude","Copilot":"Copilot","Cursor":"Cursor","Devin":"Devin"}
df=pd.read_json("data/router_dataset.jsonl",lines=True).reset_index(drop=True)
df["merged_i"]=(df.outcome=="merged").astype(int); df["success"]=df.outcome.map(SM)
order=df.groupby("agent").merged_i.mean().sort_values(ascending=False).index.tolist()

# ---- FIG 1: dataset composition ----
ct=pd.crosstab(df.agent,df.outcome)[["merged","still_open","closed_unmerged"]].loc[order]
ax=ct.plot(kind="bar",stacked=True,color=["#2ca02c","#ff7f0e","#d62728"],figsize=(7.6,4.5))
ax.set_xticklabels([short[a] for a in order],rotation=0); ax.set_ylabel("number of PRs"); ax.set_xlabel("")
ax.set_title("Dataset composition by agent and outcome")
ax.legend(["merged","still open","closed unmerged"],title="outcome",loc="center left",bbox_to_anchor=(1.01,0.5))
plt.tight_layout(); plt.savefig("fig/fig1_dataset_composition.png",dpi=130,bbox_inches="tight"); plt.close()

# ---- FIG 2: merge rate by stars ----
df["sb"]=pd.cut(df.stars,[-1,500,5000,1e12],labels=["<500*","500-5k*",">5k*"])
piv=df.pivot_table("merged_i","agent","sb",observed=False).loc[order]
ax=piv.plot(kind="bar",figsize=(7.5,4.5),color=["#4c72b0","#dd8452","#c44e52"])
ax.set_xticklabels([short[a] for a in order],rotation=0); ax.set_ylabel("merge rate"); ax.set_xlabel(""); ax.set_ylim(0,1)
ax.set_title("Merge rate by agent and repository popularity"); ax.legend(title="repo stars")
plt.tight_layout(); plt.savefig("fig/fig4_selection_bias_stars.png",dpi=130); plt.close()

# ---- FIG 3: agent x task_type heatmap ----
tt=df.task_type.value_counts().head(8).index.tolist()
hm=df[df.task_type.isin(tt)].pivot_table("merged_i","task_type","agent",observed=False)[order].loc[tt]
fig,ax=plt.subplots(figsize=(7,5))
im=ax.imshow(hm.values,cmap="RdYlGn",vmin=0.3,vmax=0.95,aspect="auto")
ax.set_xticks(range(len(order))); ax.set_xticklabels([short[a] for a in order])
ax.set_yticks(range(len(tt))); ax.set_yticklabels(tt)
for i in range(len(tt)):
    for j in range(len(order)): ax.text(j,i,f"{hm.values[i,j]:.2f}",ha="center",va="center",fontsize=8)
ax.set_title("Merge rate by agent and task type"); fig.colorbar(im,label="merge rate")
plt.tight_layout(); plt.savefig("fig/fig2_agent_tasktype_heatmap.png",dpi=130); plt.close()

# ---- v2 model ----
df["log_stars"]=np.log1p(df.stars); df["log_forks"]=np.log1p(df.forks); df["has_issue_i"]=df.has_issue.astype(int)
NUM=["log_stars","log_forks","has_issue_i"]; EMB=np.load("data/emb_title_all-MiniLM-L6-v2_25580.npy")
def gs(d,ts):
    a,b=next(GroupShuffleSplit(1,test_size=ts,random_state=SEED).split(d,groups=d.repo_id)); return d.iloc[a].copy(),d.iloc[b].copy()
tr,tmp=gs(df,0.30); va,te=gs(tmp,0.50)
lang=OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=50,sparse_output=False).fit(tr[["language"]])
task=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(tr[["task_type"]])
ag=OneHotEncoder(categories=[AG],handle_unknown="ignore",sparse_output=False).fit(tr[["agent"]])
def base(d,txt=True):
    bl=[d[NUM].to_numpy(float),lang.transform(d[["language"]]),task.transform(d[["task_type"]])]
    return np.hstack(([EMB[d.index.values]]+bl) if txt else bl)
def wa(b,a): return np.hstack([b,ag.transform(pd.DataFrame({"agent":a}))])
def fit(txt):
    m=xgb.XGBRegressor(n_estimators=800,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,
        min_child_weight=5,reg_lambda=1.0,early_stopping_rounds=40,random_state=SEED,n_jobs=-1)
    m.fit(wa(base(tr,txt),tr.agent),tr.success.to_numpy(),eval_set=[(wa(base(va,txt),va.agent),va.success.to_numpy())],verbose=False); return m
m=fit(True); bte=base(te,True); m2=fit(False); bte2=base(te,False)
amean=tr.groupby("agent").success.mean()
print("[MAE] baseline %.3f | v2 %.3f | repo-only %.3f"%(
    mean_absolute_error(te.success,te.agent.map(amean)),
    mean_absolute_error(te.success,m.predict(wa(bte,te.agent))),
    mean_absolute_error(te.success,m2.predict(wa(bte2,te.agent)))))
sc=np.column_stack([m.predict(wa(bte,[a]*len(te))) for a in AG])     # instance per-agent success preds

# ---- cells: q_train routes, q_val evaluates (independent) ----
TOPL=set(tr.language.value_counts().head(12).index)
def cell(d):
    l=d.language.where(d.language.isin(TOPL),"Other"); return (l+"|"+d.task_type).values
for d in (tr,va,te): d["cell"]=cell(d)
def qtab(d):
    gm=d.groupby("agent").success.mean(); q=d.groupby(["cell","agent"]).success.mean().unstack().reindex(columns=AG)
    for a in AG: q[a]=q[a].fillna(gm[a])
    return q,gm
q_tr,gm_tr=qtab(tr); q_va,gm_va=qtab(va); tc=te.cell.values
def Q(tab,gm,cells,agents): return np.array([tab.loc[c,a] if c in tab.index else gm[a] for c,a in zip(cells,agents)])
def V(agents): return float(Q(q_va,gm_va,tc,agents).mean())   # INDEPENDENT evaluator (val cells)
def C(agents): return float(np.mean([COST[a] for a in agents]))
# per-test-task value of each agent under the evaluator + train-cell routing scores
val_by_agent={a:Q(q_va,gm_va,tc,[a]*len(te)) for a in AG}
qtr_by_agent={a:Q(q_tr,gm_tr,tc,[a]*len(te)) for a in AG}

print("\n[SUCCESS-ONLY routing, evaluated on val cells]")
rng=np.random.default_rng(SEED)
cellbest=q_tr.idxmax(1); valbest=q_va.idxmax(1)
raw_mode=tr.groupby(["cell","agent"]).size().unstack().reindex(columns=AG).fillna(0).mul(pd.Series({a:NAT[a]/5116 for a in AG})).idxmax(1)
suite={
 "Always-cheapest (Copilot)": ["Copilot"]*len(te),
 "Random":                    list(rng.choice(AG,len(te))),
 "Instance router":           list(np.array(AG)[sc.argmax(1)]),
 "Cell argmax router":        [cellbest.get(c,gm_tr.idxmax()) for c in tc],
 "Most-popular":              [raw_mode.get(c,"OpenAI_Codex") for c in tc],
 "Always-best (Codex)":       ["OpenAI_Codex"]*len(te),
 "Cell oracle (val, opt.)":   [valbest.get(c,gm_va.idxmax()) for c in tc],
}
for n,ch in suite.items():
    print(f"  {n:26} success={V(ch):.3f} cost=${C(ch):.2f} %codex={np.mean([a=='OpenAI_Codex' for a in ch])*100:.0f}")

# ---- COST FRONTIER: send a fraction to Copilot, picked by gap (cell vs instance vs random) ----
gap_cell = qtr_by_agent["Copilot"]-qtr_by_agent["OpenAI_Codex"]      # train-cell gap (new method)
gap_inst = sc[:,AG.index("Copilot")]-sc[:,AG.index("OpenAI_Codex")]  # instance-model gap
def frontier(score):
    o=np.argsort(-score); xs=[];ys=[]
    for p in np.linspace(0,1,11):
        k=int(round(p*len(te))); sel=np.zeros(len(te),bool); sel[o[:k]]=True
        ag_ch=np.where(sel,"Copilot","OpenAI_Codex")
        ys.append(np.where(sel,val_by_agent["Copilot"],val_by_agent["OpenAI_Codex"]).mean())
        xs.append(C(list(ag_ch)))
    return np.array(xs),np.array(ys)
xr,yr=frontier(rng.standard_normal(len(te))); xi,yi=frontier(gap_inst); xc,yc=frontier(gap_cell)
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.plot(xr,yr,"o--",color="#999999",label="random mix",ms=5)
ax.plot(xi,yi,"s--",color="#dd8452",label="instance router",ms=5)
ax.plot(xc,yc,"o-",color="#4c72b0",label="cell cost-gap router",ms=6,lw=2)
ax.scatter([C(["OpenAI_Codex"]*len(te))],[V(["OpenAI_Codex"]*len(te))],color="#d62728",marker="*",s=260,zorder=5,label="always-Codex")
ax.set_xlabel("mean cost ($ / task)"); ax.set_ylabel("estimated success (val-cell estimator)")
ax.set_title("Cost-quality frontier: routing more tasks to the cheap agent")
ax.grid(alpha=.3); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig("fig/fig3_cost_quality_pareto.png",dpi=130); plt.close()

# ---- FIG 5: success-only baseline bars (val estimator) ----
names=["Always-cheapest (Copilot)","Random","Instance router","Most-popular","Cell argmax router","Always-best (Codex)"]
succ=[V(suite[n]) for n in names]; cost=[C(suite[n]) for n in names]; pcx=[np.mean([a=='OpenAI_Codex' for a in suite[n]])*100 for n in names]
o2=np.argsort(succ); names=[names[i] for i in o2]; succ=[succ[i] for i in o2]; cost=[cost[i] for i in o2]; pcx=[pcx[i] for i in o2]
fig,ax=plt.subplots(figsize=(9,4.8))
colors=["#2ca02c" if "Codex" in n else "#4c72b0" if "Cell" in n or "Instance" in n else "#c44e52" if "cheapest" in n else "#999999" for n in names]
ax.bar(range(len(names)),succ,color=colors)
for i,(s,c) in enumerate(zip(succ,cost)):
    ax.text(i,s+0.003,f"{s:.3f}",ha="center",fontsize=9,fontweight="bold"); ax.text(i,0.655,f"${c:.2f}",ha="center",fontsize=8,color="#333")
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" (","\n(") for n in names],fontsize=8)
ax.set_ylabel("estimated success (val-cell estimator)"); ax.set_ylim(0.65,0.92)
ax.axhline(V(["OpenAI_Codex"]*len(te)),ls="--",c="green",alpha=0.4)
ax.set_title("Success-only routing: no method beats always-Codex"); plt.tight_layout()
plt.savefig("fig/fig6_baselines.png",dpi=130); plt.close()

print("\n[COST FRONTIER at matched cost — success of random / instance / cell(new)]")
for p,xrr,yrr,yii,ycc in zip(np.linspace(0,1,11),xr,yr,yi,yc):
    if round(p,1) in (0.2,0.4,0.6,0.8): print(f"  cost=${xrr:.2f}: random={yrr:.3f} instance={yii:.3f} cell={ycc:.3f}")
print("  always-Codex: success=%.3f cost=$%.2f"%(V(['OpenAI_Codex']*len(te)),C(['OpenAI_Codex']*len(te))))

# ---- FIG 6: quality dimension beyond merge (pop subset; needs pr_reviews) ----
try:
    import duckdb
    from huggingface_hub import hf_hub_download
    pp=hf_hub_download("hao-li/AIDev","pull_request.parquet",repo_type="dataset")
    rr=hf_hub_download("hao-li/AIDev","pr_reviews.parquet",repo_type="dataset")
    con=duckdb.connect()
    cm=hf_hub_download("hao-li/AIDev","pr_commits.parquet",repo_type="dataset")
    con.execute(f"CREATE VIEW rv AS SELECT pr_id, count(*) FILTER (WHERE state='CHANGES_REQUESTED') n_cr, count(*) n_rev FROM read_parquet('{rr}') GROUP BY pr_id")
    con.execute(f"CREATE VIEW ck AS SELECT pr_id, count(*) n_commits FROM read_parquet('{cm}') GROUP BY pr_id")
    qd=con.execute(f"SELECT p.agent, (p.merged_at IS NOT NULL) merged, coalesce(v.n_cr,0) n_cr, coalesce(v.n_rev,0) n_rev, coalesce(k.n_commits,0) n_commits FROM read_parquet('{pp}') p LEFT JOIN rv v ON p.id=v.pr_id LEFT JOIN ck k ON p.id=k.pr_id").df()
    qd["cr"]=(qd.n_cr>0); allr=qd.groupby("agent").cr.mean(); revr=qd[qd.n_rev>0].groupby("agent").cr.mean()
    unrev=qd.assign(u=qd.n_rev==0).groupby("agent").u.mean(); o=revr.sort_values().index.tolist()
    xq=np.arange(len(o)); w=0.38; fig,ax=plt.subplots(figsize=(8,4.8))
    ax.bar(xq-w/2,[allr[a] for a in o],w,label="all PRs",color="#bcbddc")
    ax.bar(xq+w/2,[revr[a] for a in o],w,label="reviewed PRs only",color="#4c72b0")
    for i,a in enumerate(o):
        ax.text(i-w/2,allr[a]+0.004,f"{allr[a]:.3f}",ha="center",fontsize=8)
        ax.text(i+w/2,revr[a]+0.004,f"{revr[a]:.3f}",ha="center",fontsize=8,fontweight="bold")
        ax.text(i,-0.022,f"{unrev[a]*100:.0f}% unrev.",ha="center",fontsize=7.5,color="#666")
    ax.set_xticks(xq); ax.set_xticklabels([short[a] for a in o]); ax.set_ylim(0,0.26)
    ax.set_ylabel("changes-requested rate")
    ax.set_title("Review friction: Codex's edge is a self-merge artifact (reviewed-only levels out)")
    ax.legend(loc="upper left"); plt.tight_layout(); plt.savefig("fig/fig5_quality.png",dpi=130); plt.close()
    print("[FIG6] reviewed-only changes-req:",{short[a]:round(revr[a],3) for a in o})
    qd["clean"]=(qd.merged & (qd.n_cr==0))
    qm=qd.groupby("agent").agg(clean_merge=("clean","mean"),changes_req=("cr","mean"),
        review_burden=("n_rev","mean"),commits=("n_commits","mean")).reindex(o).round(3)
    print("[FIG6] alt quality metrics, all PRs (backs the Section 4.4 claim):\n"+qm.to_string())
except Exception as e:
    print("[FIG6] skipped (needs duckdb + HF pr_reviews):",repr(e)[:80])
print("\nsaved fig/fig1..6")
