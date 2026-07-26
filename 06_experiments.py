"""Final-report experiments built on the reproduced v2 pipeline (router_lib.py).

Experiments:
  A featimp    - which features the router actually uses (XGBoost gain importance)
  B modelclass - does any model class beat XGBoost on these features?
  C ci         - bootstrap confidence intervals for the routing claims
  D binary     - do the conclusions hold under a binary 'merged' label?

Run: python 06_experiments.py
Writes figures to fig/ and a results summary to experiments_results.md.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import router_lib as R

RESULTS = {}


def group_of(name):
    if name.startswith("emb_"):
        return "title (MiniLM, 384d)"
    if name.startswith("language_"):
        return "language"
    if name.startswith("task_type_"):
        return "task_type"
    if name.startswith("agent_"):
        return "agent"
    return name  # log_stars, log_forks, has_issue_i


def grouped_permutation_importance(p, n_repeats=5, seed=R.RANDOM_STATE):
    """Block-level permutation importance on the test set: MAE increase when each
    feature group is jointly shuffled. Unlike gain, this is not biased by how many
    columns a group has, so it measures real out-of-sample reliance."""
    from sklearn.metrics import mean_absolute_error
    names = p.feature_names
    groups = {}
    for i, nm in enumerate(names):
        groups.setdefault(group_of(nm), []).append(i)
    base = mean_absolute_error(p.yte, p.model.predict(p.Xte))
    rng = np.random.default_rng(seed)
    out = {}
    for g, idx in groups.items():
        idx = np.array(idx)
        deltas = []
        for _ in range(n_repeats):
            Xp = p.Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xp[np.ix_(perm, idx)]
            deltas.append(mean_absolute_error(p.yte, p.model.predict(Xp)) - base)
        out[g] = float(np.mean(deltas))
    return base, pd.Series(out).sort_values(ascending=False)


def exp_a_featimp(p):
    booster = p.model.get_booster()
    gain = booster.get_score(importance_type="gain")  # {'f<idx>': gain}, only used features
    names = p.feature_names
    per_feat = {names[int(k[1:])]: v for k, v in gain.items()}
    total = sum(per_feat.values())

    # grouped gain totals + counts + per-feature mean (gain is biased toward many-column groups)
    rows, counts = {}, {}
    for nm in names:
        g = group_of(nm)
        rows[g] = rows.get(g, 0.0) + per_feat.get(nm, 0.0)
        counts[g] = counts.get(g, 0) + 1
    grouped = pd.DataFrame({
        "gain_%": {g: 100 * v / total for g, v in rows.items()},
        "n_feat": counts,
    })
    grouped["gain_per_feat_%"] = (grouped["gain_%"] / grouped["n_feat"]).round(3)
    grouped = grouped.sort_values("gain_%", ascending=False).round(2)
    top = (pd.Series(per_feat).sort_values(ascending=False) / total * 100).head(15).round(2)

    # unbiased cross-check: grouped permutation importance (out-of-sample MAE increase)
    base_mae, perm = grouped_permutation_importance(p)

    print("\n[A] feature importance")
    print("grouped GAIN (in-sample, count-biased):\n", grouped)
    print("\ntop-15 individual features by gain (%):\n", top)
    print(f"\ngrouped PERMUTATION importance (test MAE increase; base MAE={base_mae:.3f}):")
    print(perm.round(4))

    # figure: grouped permutation importance is the honest, count-unbiased view
    fig, ax = plt.subplots(figsize=(7, 4))
    perm[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("test-set MAE increase when group is permuted")
    ax.set_title("Feature-group importance (permutation, out-of-sample)")
    fig.tight_layout()
    fig.savefig("fig/fig7_feature_importance.png", dpi=150)
    plt.close(fig)

    RESULTS["A_featimp"] = {
        "grouped_gain": grouped.reset_index().rename(columns={"index": "group"}).to_dict("records"),
        "top15_individual_gain_pct": top.to_dict(),
        "title_block_gain_pct": round(grouped.loc["title (MiniLM, 384d)", "gain_%"], 2)
        if "title (MiniLM, 384d)" in grouped.index else 0.0,
        "base_test_mae": round(base_mae, 4),
        "grouped_permutation_mae_increase": {k: round(v, 4) for k, v in perm.items()},
    }


def exp_b_modelclass(p):
    """Capacity ladder on the v2 (title+repo) features: if a constant, a linear
    model, a small MLP, and a tuned GBDT all land at the same MAE, then model
    capacity is not the bottleneck and no model is likely to do better on these
    features. Linear/MLP are standardized; XGBoost uses raw features."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    Xtr, ytr, Xva, yva, Xte, yte = p.Xtr, p.ytr, p.Xva, p.yva, p.Xte, p.yte
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    amean = p.train_df.groupby("agent").success.mean()
    res = {"per-agent-mean baseline": mean_absolute_error(yte, p.test_df.agent.map(amean))}

    # linear: pick Ridge alpha on val (parameter optimization, recorded)
    alphas = [0.1, 1.0, 10.0, 100.0]
    val_mae = {a: mean_absolute_error(yva, Ridge(alpha=a).fit(Ztr, ytr).predict(Zva)) for a in alphas}
    best_alpha = min(val_mae, key=val_mae.get)
    res[f"Ridge (linear, alpha={best_alpha:g})"] = mean_absolute_error(
        yte, Ridge(alpha=best_alpha).fit(Ztr, ytr).predict(Zte))

    # small MLP, config picked on val (fair tuning, so the NN is not a strawman)
    mlp_grid = [
        {"hidden_layer_sizes": (64,), "alpha": 1e-4},
        {"hidden_layer_sizes": (128,), "alpha": 1e-4},
        {"hidden_layer_sizes": (128,), "alpha": 1e-2},
        {"hidden_layer_sizes": (128, 64), "alpha": 1e-4},
    ]
    mlp_val = []
    for cfg in mlp_grid:
        m = MLPRegressor(max_iter=500, early_stopping=True, random_state=R.RANDOM_STATE, **cfg).fit(Ztr, ytr)
        mlp_val.append((mean_absolute_error(yva, m.predict(Zva)), cfg, m))
    mlp_val.sort(key=lambda t: t[0])
    _, best_cfg, best_mlp = mlp_val[0]
    res[f"MLP {best_cfg['hidden_layer_sizes']} a={best_cfg['alpha']:g}"] = mean_absolute_error(
        yte, best_mlp.predict(Zte))

    res["XGBoost (tuned, our model)"] = mean_absolute_error(yte, p.model.predict(Xte))

    tab = pd.Series(res).round(3).sort_values()
    print("\n[B] model-class comparison (test MAE, v2 title+repo features)")
    print(tab)
    print(f"spread (max-min) = {tab.max() - tab.min():.3f}; Ridge val sweep = "
          + ", ".join(f"a={a:g}:{val_mae[a]:.3f}" for a in alphas))

    RESULTS["B_modelclass"] = {
        "test_mae": {k: round(v, 3) for k, v in res.items()},
        "spread": round(tab.max() - tab.min(), 3),
        "ridge_alpha_val_sweep": {f"alpha={a:g}": round(val_mae[a], 3) for a in alphas},
        "best_alpha": best_alpha,
        "mlp_val_sweep": [{"cfg": str(c), "val_mae": round(v, 3)} for v, c, _ in mlp_val],
    }


def _boot_means(rowvals, B=2000, seed=R.RANDOM_STATE):
    """Paired bootstrap over test rows: returns per-policy arrays of B resampled
    means, all using the SAME resamples so differences are paired."""
    n = len(next(iter(rowvals.values())))
    rng = np.random.default_rng(seed)
    means = {k: np.empty(B) for k in rowvals}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for k, v in rowvals.items():
            means[k][b] = v[idx].mean()
    return means


def _ci(arr):
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def exp_c_ci(p):
    """Bootstrap 95% CIs for (1) success-only policy values and (2) the cost-gap
    frontier, to test whether 'no router beats always-Codex' and the cost-gap
    router's ~2-point edge are real or within noise."""
    est = p.est
    n = len(p.test_df)
    Qva = lambda ch: est.Q(est.q_va, est.gm_va, ch)

    # ---- success-only policies: per-row valuation under the val-cell estimator ----
    cell_best = est.q_tr.idxmax(1)
    rng = np.random.default_rng(R.RANDOM_STATE)
    pol_rowvals = {
        "Always-Codex": Qva(["OpenAI_Codex"] * n),
        "Instance router": Qva(p.scores[R.AGENTS].idxmax(1).values),
        "Cell argmax router": Qva([cell_best.get(c, est.gm_tr.idxmax()) for c in est.tc]),
        "Random": Qva(list(rng.choice(R.AGENTS, n))),
        "Always-cheapest": Qva(["Copilot"] * n),
    }
    bm = _boot_means(pol_rowvals)
    pol_ci = {k: _ci(bm[k]) for k in pol_rowvals}
    # paired gaps vs always-Codex (positive => Codex better)
    d_codex_inst = _ci(bm["Always-Codex"] - bm["Instance router"])
    d_codex_cell = _ci(bm["Always-Codex"] - bm["Cell argmax router"])

    print("\n[C] bootstrap 95% CIs (2000 resamples of test rows)")
    print("success-only policy values:")
    for k, (m, lo, hi) in pol_ci.items():
        print(f"  {k:20} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  diff Always-Codex - Instance     {d_codex_inst[0]:+.3f}  [{d_codex_inst[1]:+.3f}, {d_codex_inst[2]:+.3f}]")
    print(f"  diff Always-Codex - Cell-argmax  {d_codex_cell[0]:+.3f}  [{d_codex_cell[1]:+.3f}, {d_codex_cell[2]:+.3f}]")

    # ---- cost-gap frontier: is cell cost-gap significantly > random / instance? ----
    valC, valX = Qva(["Copilot"] * n), Qva(["OpenAI_Codex"] * n)
    qtrA = {a: est.Q(est.q_tr, est.gm_tr, [a] * n) for a in ("Copilot", "OpenAI_Codex")}
    gaps = {
        "random": rng.standard_normal(n),
        "instance": p.scores["Copilot"].to_numpy() - p.scores["OpenAI_Codex"].to_numpy(),
        "cell cost-gap": qtrA["Copilot"] - qtrA["OpenAI_Codex"],
    }

    def sel_rowval(gap, frac):
        k = int(round(frac * n))
        sel = np.zeros(n, bool)
        sel[np.argsort(-gap)[:k]] = True
        return np.where(sel, valC, valX)

    frontier = {}
    for frac, costlbl in [(0.2, "$3.30/14% cheaper"), (0.4, "$2.75/29% cheaper"), (0.6, "$2.20/43% cheaper")]:
        rowvals = {m: sel_rowval(g, frac) for m, g in gaps.items()}
        bmf = _boot_means(rowvals)
        cis = {m: _ci(bmf[m]) for m in rowvals}
        d_cr = _ci(bmf["cell cost-gap"] - bmf["random"])
        d_ci = _ci(bmf["cell cost-gap"] - bmf["instance"])
        frontier[costlbl] = {"ci": cis, "cell_minus_random": d_cr, "cell_minus_instance": d_ci}
        print(f"\nfrontier @ {costlbl}:")
        for m, (mean, lo, hi) in cis.items():
            print(f"  {m:14} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
        print(f"  cell - random   {d_cr[0]:+.3f}  [{d_cr[1]:+.3f}, {d_cr[2]:+.3f}]"
              f"   ({'significant' if d_cr[1] > 0 else 'NOT significant'})")
        print(f"  cell - instance {d_ci[0]:+.3f}  [{d_ci[1]:+.3f}, {d_ci[2]:+.3f}]"
              f"   ({'significant' if d_ci[1] > 0 else 'NOT significant'})")

    # figure: forest plot of success-only policy CIs
    fig, ax = plt.subplots(figsize=(7, 3.5))
    order = sorted(pol_ci, key=lambda k: pol_ci[k][0])
    ys = range(len(order))
    for y, k in zip(ys, order):
        m, lo, hi = pol_ci[k]
        ax.plot([lo, hi], [y, y], color="#555", lw=2)
        ax.plot(m, y, "o", color="#4C72B0")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(order)
    ax.set_xlabel("estimated success (95% bootstrap CI)")
    ax.set_title("Success-only routing: policy values with CIs")
    fig.tight_layout()
    fig.savefig("fig/fig10_policy_ci.png", dpi=150)  # supplementary
    plt.close(fig)

    RESULTS["C_ci"] = {
        "success_only": {k: {"mean": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3)}
                         for k, (m, lo, hi) in pol_ci.items()},
        "diff_codex_minus_instance": [round(x, 4) for x in d_codex_inst],
        "diff_codex_minus_cellargmax": [round(x, 4) for x in d_codex_cell],
        "frontier": {lbl: {
            "ci": {m: [round(x, 3) for x in c] for m, c in d["ci"].items()},
            "cell_minus_random": [round(x, 4) for x in d["cell_minus_random"]],
            "cell_minus_instance": [round(x, 4) for x in d["cell_minus_instance"]],
        } for lbl, d in frontier.items()},
    }


def exp_d_binary():
    """Robustness to the label: rebuild with a binary 'merged' target (still_open
    and closed_unmerged -> 0) and check the two qualitative conclusions hold:
    (1) Codex has the top merge rate, (2) no learned router beats always-Codex."""
    bin_map = {"merged": 1.0, "still_open": 0.0, "closed_unmerged": 0.0}
    p = R.prepare(success_map=bin_map)
    est, n = p.est, len(p.test_df)

    merge_rate = p.train_df.groupby("agent").success.mean().sort_values(ascending=False)
    cell_best = est.q_tr.idxmax(1)
    pols = {
        "Always-Codex": ["OpenAI_Codex"] * n,
        "Instance router": p.scores[R.AGENTS].idxmax(1).values,
        "Cell argmax router": [cell_best.get(c, est.gm_tr.idxmax()) for c in est.tc],
    }
    pvals = {k: est.pval(v) for k, v in pols.items()}

    # one frontier point (29% cheaper) under the binary estimator
    Qva = lambda ch: est.Q(est.q_va, est.gm_va, ch)
    valC, valX = Qva(["Copilot"] * n), Qva(["OpenAI_Codex"] * n)
    qtrA = {a: est.Q(est.q_tr, est.gm_tr, [a] * n) for a in ("Copilot", "OpenAI_Codex")}
    gap_cell = qtrA["Copilot"] - qtrA["OpenAI_Codex"]
    sel = np.zeros(n, bool); sel[np.argsort(-gap_cell)[:int(0.4 * n)]] = True
    cell_succ = float(np.where(sel, valC, valX).mean())
    rng = np.random.default_rng(R.RANDOM_STATE)
    rsel = np.zeros(n, bool); rsel[rng.permutation(n)[:int(0.4 * n)]] = True
    rand_succ = float(np.where(rsel, valC, valX).mean())

    codex_top = merge_rate.index[0] == "OpenAI_Codex"
    no_router_beats = pvals["Always-Codex"] >= max(pvals["Instance router"], pvals["Cell argmax router"])

    print("\n[D] binary 'merged' label robustness")
    print("per-agent merge rate (train):\n", merge_rate.round(3).to_string())
    print("success-only routing (binary val-cell estimator):")
    for k, v in pvals.items():
        print(f"  {k:20} {v:.3f}")
    print(f"frontier @29% cheaper: cell cost-gap {cell_succ:.3f} vs random {rand_succ:.3f}")
    print(f"=> Codex top merge rate: {codex_top}; no router beats always-Codex: {no_router_beats}; "
          f"cell>random under cost: {cell_succ > rand_succ}")

    RESULTS["D_binary"] = {
        "merge_rate_train": merge_rate.round(3).to_dict(),
        "success_only_binary": {k: round(v, 3) for k, v in pvals.items()},
        "frontier_29pct": {"cell_cost_gap": round(cell_succ, 3), "random": round(rand_succ, 3)},
        "codex_top_merge_rate": bool(codex_top),
        "no_router_beats_codex": bool(no_router_beats),
        "cell_beats_random_under_cost": bool(cell_succ > rand_succ),
    }


def _propensity(p, clip=0.02):
    """P(agent | repo+task covariates) from multinomial logistic (no text, no agent).
    On the balanced corpus the marginal is 1/5; deviations are the selection signal."""
    from sklearn.linear_model import LogisticRegression
    lang_oh, task_oh, _ = p.enc

    def Xp(d):
        return np.hstack([d[R.NUM].to_numpy(float),
                          lang_oh.transform(d[["language"]]),
                          task_oh.transform(d[["task_type"]])])

    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xp(p.train_df), p.train_df.agent.values)
    e_raw = clf.predict_proba(Xp(p.test_df))
    e = np.clip(e_raw, clip, 1.0)
    col = {a: i for i, a in enumerate(clf.classes_)}
    acc = float(clf.score(Xp(p.test_df), p.test_df.agent.values))
    # positivity / overlap diagnostics: how often the propensity for the agent that
    # actually acted falls below the clip floor (a near-violation of positivity).
    a_idx = np.array([col[a] for a in p.test_df.agent.values])
    e_actual_raw = e_raw[np.arange(len(a_idx)), a_idx]
    positivity = {
        "clip_floor": clip,
        "min_propensity_overall": round(float(e_raw.min()), 4),
        "min_propensity_actual_agent": round(float(e_actual_raw.min()), 4),
        "frac_actual_below_floor": round(float((e_actual_raw < clip).mean()), 4),
        "frac_any_cell_below_floor": round(float((e_raw < clip).mean()), 4),
    }
    return e, col, acc, positivity


def exp_e_ipw(p):
    """Cross-check the direct-method (DM) policy values with self-normalized IPW
    (SNIPW) and doubly-robust (DR) estimators that use a covariate propensity
    model. Agreement => conclusions are not an artifact of the cell estimator."""
    est, n = p.est, len(p.test_df)
    A = p.test_df.agent.values
    Rv = p.test_df.success.to_numpy()
    e, col, prop_acc, positivity = _propensity(p)
    Q_actual = est.Q(est.q_va, est.gm_va, A)

    def e_of(choice):
        return e[np.arange(n), [col[a] for a in choice]]

    def snipw(choice):
        match = (A == np.asarray(choice))
        w = np.where(match, 1.0 / e_of(choice), 0.0)
        return float((w * Rv).sum() / w.sum())

    def ipw_diag(choice):
        match = (A == np.asarray(choice))
        wm = (1.0 / e_of(choice))[match]
        ess = float(wm.sum() ** 2 / (wm ** 2).sum())  # Kish effective sample size
        return int(match.sum()), round(ess, 0)

    def dr(choice):
        match = (A == np.asarray(choice)).astype(float)
        Qpi = est.Q(est.q_va, est.gm_va, choice)
        return float((Qpi + match / e_of(choice) * (Rv - Q_actual)).mean())

    cell_best = est.q_tr.idxmax(1)
    rng = np.random.default_rng(R.RANDOM_STATE)
    policies = {
        "Always-Codex": ["OpenAI_Codex"] * n,
        "Instance router": list(p.scores[R.AGENTS].idxmax(1).values),
        "Cell argmax router": [cell_best.get(c, est.gm_tr.idxmax()) for c in est.tc],
        "Random": list(rng.choice(R.AGENTS, n)),
        "Always-cheapest": ["Copilot"] * n,
    }
    tab = pd.DataFrame({name: {"direct": est.pval(ch), "snipw": snipw(ch), "dr": dr(ch)}
                        for name, ch in policies.items()}).T
    diag = {name: ipw_diag(ch) for name, ch in policies.items()}
    tab["n_match"] = [diag[k][0] for k in tab.index]
    tab["ipw_ess"] = [diag[k][1] for k in tab.index]
    tab = tab.round({"direct": 3, "snipw": 3, "dr": 3})
    tab["max_gap"] = (tab[["direct", "snipw", "dr"]].max(axis=1) - tab[["direct", "snipw", "dr"]].min(axis=1)).round(3)

    print(f"\n[E] estimator cross-check (propensity model test accuracy {prop_acc:.3f}, base 0.200)")
    print(tab)
    print("IPW positivity/overlap diagnostics:", positivity)
    # do the conclusions survive every estimator?
    best_each = {est_name: tab[est_name].idxmax() for est_name in ["direct", "snipw", "dr"]}
    codex_wins_all = all(v == "Always-Codex" for v in best_each.values())
    print(f"highest-value policy under each estimator: {best_each}")
    print(f"=> always-Codex top under all three estimators: {codex_wins_all}")

    # does the cost-gap positive result survive the deconfounded estimators?
    # value each cost-saving policy (send 40% of tasks to Copilot, 29% cheaper) under DM/SNIPW/DR
    qtrA = {a: est.Q(est.q_tr, est.gm_tr, [a] * n) for a in ("Copilot", "OpenAI_Codex")}
    gaps = {
        "random": rng.standard_normal(n),
        "instance": p.scores["Copilot"].to_numpy() - p.scores["OpenAI_Codex"].to_numpy(),
        "cell cost-gap": qtrA["Copilot"] - qtrA["OpenAI_Codex"],
    }
    frontier = {}
    for m, g in gaps.items():
        sel = np.zeros(n, bool); sel[np.argsort(-g)[:int(0.4 * n)]] = True
        ch = list(np.where(sel, "Copilot", "OpenAI_Codex"))
        frontier[m] = {"direct": round(est.pval(ch), 3), "snipw": round(snipw(ch), 3), "dr": round(dr(ch), 3)}
    ftab = pd.DataFrame(frontier).T
    cell_best_each = {c: ftab[c].idxmax() for c in ["direct", "snipw", "dr"]}
    cost_gap_robust = all(v == "cell cost-gap" for v in cell_best_each.values())
    print("\ncost-saving policies @29% cheaper, valued under each estimator:")
    print(ftab)
    print(f"=> cell cost-gap is best under all three estimators: {cost_gap_robust}")

    # figure: DM vs IPW vs DR for the three routing policies
    show = ["Always-Codex", "Instance router", "Cell argmax router"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(show)); w = 0.26
    for k, off, c in [("direct", -w, "#4C72B0"), ("snipw", 0, "#DD8452"), ("dr", w, "#55A868")]:
        ax.bar(x + off, [tab.loc[s, k] for s in show], w, label=k.upper(), color=c)
    ax.set_xticks(x); ax.set_xticklabels(show, fontsize=9)
    ax.set_ylim(0.83, 0.89); ax.set_ylabel("estimated success")
    ax.set_title("Policy value under three off-policy estimators")
    ax.legend()
    fig.tight_layout(); fig.savefig("fig/fig8_estimator_crosscheck.png", dpi=150); plt.close(fig)

    RESULTS["E_ipw"] = {
        "propensity_test_accuracy": round(prop_acc, 3),
        "positivity": positivity,
        "policy_values": tab.to_dict("index"),
        "best_under_each_estimator": best_each,
        "codex_top_under_all_estimators": bool(codex_wins_all),
        "cost_gap_frontier_29pct": ftab.to_dict("index"),
        "cost_gap_best_under_each_estimator": cell_best_each,
        "cost_gap_robust_all_estimators": bool(cost_gap_robust),
    }


def exp_f_twolevel(p, B=500):
    """Two-level bootstrap: resample the validation set (re-estimating the cell
    values) AND the test set, so the CI includes the estimator's own uncertainty,
    not only test sampling. Tests whether the Codex-vs-instance gap survives."""
    AGENTS = R.AGENTS
    val = p.val_df[["cell", "agent", "success"]]
    test_cells = p.test_df["cell"].to_numpy()
    n = len(test_cells)
    cell_best = p.est.q_tr.idxmax(1)
    choices = {
        "Always-Codex": np.array(["OpenAI_Codex"] * n),
        "Instance router": p.scores[AGENTS].idxmax(1).to_numpy(),
        "Cell argmax router": np.array([cell_best.get(c, p.est.gm_tr.idxmax()) for c in test_cells]),
    }
    # Cost-gap edge under two-level resampling: fixed routing masks (train signal +
    # instance scores), but the EVALUATOR (val cell estimate) is resampled each rep,
    # so the cost-gap-vs-random/instance gaps face estimator+test uncertainty too.
    frac = 0.4  # 29% cheaper
    k = int(round(frac * n))
    rng0 = np.random.default_rng(R.RANDOM_STATE)
    gap_signals = {
        "random": rng0.standard_normal(n),
        "instance": p.scores["Copilot"].to_numpy() - p.scores["OpenAI_Codex"].to_numpy(),
        "cell cost-gap": p.est.Q(p.est.q_tr, p.est.gm_tr, ["Copilot"] * n)
        - p.est.Q(p.est.q_tr, p.est.gm_tr, ["OpenAI_Codex"] * n),
    }
    cg_sel = {}
    for m, g in gap_signals.items():
        s = np.zeros(n, bool); s[np.argsort(-g)[:k]] = True
        cg_sel[m] = s

    rng = np.random.default_rng(R.RANDOM_STATE)
    means = {k2: np.empty(B) for k2 in choices}
    cg_means = {m: np.empty(B) for m in gap_signals}
    for b in range(B):
        vb = val.sample(frac=1.0, replace=True, random_state=int(rng.integers(1 << 31)))
        gm = vb.groupby("agent").success.mean().reindex(AGENTS)
        q = vb.groupby(["cell", "agent"]).success.mean().unstack().reindex(columns=AGENTS)
        q = q.apply(lambda colx: colx.fillna(gm[colx.name]))
        qs = q.stack()
        tb = rng.integers(0, n, n)
        tcb = test_cells[tb]

        def val_route(chb):
            v = qs.reindex(pd.MultiIndex.from_arrays([tcb, chb])).to_numpy()
            miss = np.isnan(v)
            if miss.any():
                v[miss] = gm.reindex(chb[miss]).to_numpy()
            return v
        for name, ch in choices.items():
            means[name][b] = float(np.mean(val_route(ch[tb])))
        # cost-gap evaluation: route selected rows to Copilot else Codex, value on resampled val
        for m, sel in cg_sel.items():
            chb = np.where(sel[tb], "Copilot", "OpenAI_Codex")
            cg_means[m][b] = float(np.mean(val_route(chb)))
    ci = {k2: _ci(means[k2]) for k2 in choices}
    d = _ci(means["Always-Codex"] - means["Instance router"])
    dcg_r = _ci(cg_means["cell cost-gap"] - cg_means["random"])
    dcg_i = _ci(cg_means["cell cost-gap"] - cg_means["instance"])
    print(f"\n[F] two-level bootstrap (resample val-estimator + test, {B} reps), 95% CIs:")
    for k2, (m, lo, hi) in ci.items():
        print(f"  {k2:20} {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    sig = d[1] > 0
    print(f"  diff Codex - instance  {d[0]:+.3f}  [{d[1]:+.3f}, {d[2]:+.3f}]  -> "
          f"{'significant' if sig else 'NOT significant (within estimator+sampling noise)'}")
    print(f"  cost-gap - random   {dcg_r[0]:+.3f} [{dcg_r[1]:+.3f}, {dcg_r[2]:+.3f}]  "
          f"{'significant' if dcg_r[1] > 0 else 'NOT significant'}")
    print(f"  cost-gap - instance {dcg_i[0]:+.3f} [{dcg_i[1]:+.3f}, {dcg_i[2]:+.3f}]  "
          f"{'significant' if dcg_i[1] > 0 else 'NOT significant'}")
    RESULTS["F_twolevel"] = {
        "ci": {k2: [round(x, 3) for x in v] for k2, v in ci.items()},
        "diff_codex_minus_instance": [round(x, 4) for x in d],
        "diff_significant": bool(sig),
        "costgap_minus_random": [round(x, 4) for x in dcg_r],
        "costgap_minus_instance": [round(x, 4) for x in dcg_i],
        "costgap_edge_significant_twolevel": bool(dcg_r[1] > 0 and dcg_i[1] > 0),
    }


def exp_g_cost_router(p):
    """Cost-aware router as its real operating curve: pi_lambda(x) = argmax_a
    [Qcell_train(x,a) - lambda*cost(a)], swept over lambda to trace the full
    cost-success Pareto frontier (evaluated on val cells). Also reports which
    agents are ever selected and the success-per-dollar of each always-agent."""
    est, n = p.est, len(p.test_df)
    AG = R.AGENTS
    cost = np.array([R.COST[a] for a in AG])
    Qtr = np.column_stack([est.Q(est.q_tr, est.gm_tr, [a] * n) for a in AG])  # routing signal
    Qva = np.column_stack([est.Q(est.q_va, est.gm_va, [a] * n) for a in AG])  # evaluator

    base_succ = {a: float(Qva[:, i].mean()) for i, a in enumerate(AG)}
    eff = {a: round(base_succ[a] / R.COST[a], 3) for a in AG}  # success per dollar

    # Pareto-efficient single agents (higher success and/or lower cost than all others)
    def dominated(a):
        return any(b != a and base_succ[b] >= base_succ[a] and R.COST[b] <= R.COST[a]
                   and (base_succ[b] > base_succ[a] or R.COST[b] < R.COST[a]) for b in AG)
    pareto = [a for a in AG if not dominated(a)]

    ci, cc = AG.index("Copilot"), AG.index("OpenAI_Codex")

    # the disciplined cost-gap router: downgrade the smallest Copilot-minus-Codex cells first
    order = np.argsort(-(Qtr[:, ci] - Qtr[:, cc]))

    def costgap(fr):
        k = int(round(fr * n)); sel = np.zeros(n, bool); sel[order[:k]] = True
        return (float(np.where(sel, cost[ci], cost[cc]).mean()),
                float(np.where(sel, Qva[:, ci], Qva[:, cc]).mean()))
    cg = pd.DataFrame([costgap(fr) for fr in np.linspace(0, 1, 11)], columns=["cost", "succ"])

    # random mix (Codex <-> Copilot) for contrast
    rng = np.random.default_rng(R.RANDOM_STATE)
    ro = np.argsort(rng.standard_normal(n))

    def randmix(fr):
        k = int(round(fr * n)); sel = np.zeros(n, bool); sel[ro[:k]] = True
        return (float(np.where(sel, cost[ci], cost[cc]).mean()),
                float(np.where(sel, Qva[:, ci], Qva[:, cc]).mean()))
    rd = pd.DataFrame([randmix(fr) for fr in np.linspace(0, 1, 11)], columns=["cost", "succ"])

    # naive 5-agent lambda router: is it dominated by the targeted cost-gap rule?
    lam_at_275 = None
    for lam in np.linspace(0, 1, 401):
        pick = np.argmax(Qtr - lam * cost, axis=1)
        c = cost[pick].mean()
        if abs(c - 2.75) < 0.05:
            lam_at_275 = round(float(Qva[np.arange(n), pick].mean()), 3); break
    cg_at_275 = round(float(costgap(0.4)[1]), 3)  # 0.4 -> ~$2.75

    print("\n[G] cost-aware router (cost-success Pareto view)")
    print("Pareto-efficient single agents:", pareto, "(others are cost-dominated)")
    print("success-per-dollar (always-agent):", eff)
    print("cost-gap router knees:", {f"${c:.2f}": round(s, 3) for c, s in
                                      [costgap(0.0), costgap(0.2), costgap(0.4), costgap(0.6), costgap(1.0)]})
    print(f"naive 5-agent lambda router @ ~$2.75: {lam_at_275}  vs  cost-gap @ $2.75: {cg_at_275}  "
          f"=> targeted cost-gap {'beats' if cg_at_275 > (lam_at_275 or 0) else 'does not beat'} naive lambda")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(rd.cost, rd.succ, "o--", color="#999", ms=4, label="random mix")
    ax.plot(cg.cost, cg.succ, "-", color="#4C72B0", lw=2.6, label="cost-gap router (Codex/Copilot)")
    for a in AG:
        on = a in pareto
        ax.scatter(R.COST[a], base_succ[a], s=90 if on else 55,
                   color="#C44E52" if on else "#888", zorder=5, edgecolor="k" if on else "none")
        ax.annotate(a.replace("OpenAI_", "").replace("_Code", ""), (R.COST[a], base_succ[a]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("mean cost ($ / task)"); ax.set_ylabel("estimated success (val-cell estimator)")
    ax.set_title("Cost-success frontier: only Codex and Copilot are Pareto-efficient")
    ax.grid(alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("fig/fig9_cost_router_pareto.png", dpi=150); plt.close(fig)

    RESULTS["G_cost_router"] = {
        "pareto_efficient_agents": pareto,
        "success_per_dollar": eff,
        "always_agent_success": {a: round(v, 3) for a, v in base_succ.items()},
        "cost_gap_knees": {f"frac_{fr}": [round(x, 3) for x in costgap(fr)] for fr in [0.0, 0.2, 0.4, 0.6, 1.0]},
        "naive_lambda_at_275": lam_at_275,
        "cost_gap_at_275": cg_at_275,
    }


def exp_h_pricing(p):
    """Pricing sensitivity. (1) The method ranking on the cost axis is price-invariant:
    a policy's success at a given routing fraction does not depend on the dollar values,
    so pricing only rescales the x-axis. (2) Break-even price: how cheap each currently
    dominated agent would have to be before a cost-aware router would ever select it."""
    est, n, AG = p.est, len(p.test_df), R.AGENTS
    succ = {a: float(est.Q(est.q_va, est.gm_va, [a] * n).mean()) for a in AG}
    cost = dict(R.COST)

    def picks(costs):
        """Agents a cost-aware (success - lambda*cost) router would select for some lambda>=0."""
        return {max(AG, key=lambda a: succ[a] - lam * costs[a]) for lam in np.linspace(0, 2, 400)}

    cur = picks(cost)

    def breakeven(a):
        if a in cur:
            return cost[a]
        lo, hi = 0.0, cost[a]
        for _ in range(40):
            mid = (lo + hi) / 2
            c2 = dict(cost); c2[a] = mid
            if a in picks(c2):
                lo = mid
            else:
                hi = mid
        return lo

    be = {a: round(breakeven(a), 2) for a in AG}

    print("\n[H] pricing sensitivity")
    print("agents a cost-router selects at current prices:", sorted(cur))
    print("break-even price (max $/PR to ever be selected) vs current:")
    for a in AG:
        tag = "selected now" if a in cur else f"needs <= ${be[a]:.2f}  (now ${cost[a]:.2f}, {be[a]/cost[a]*100:.0f}% of current)"
        print(f"  {a:13} succ={succ[a]:.3f}  {tag}")
    print("note: the cost-gap router's success at a given routing fraction is independent of the "
          "dollar values, so its advantage over random/instance (Section 4.3) is price-invariant; "
          "pricing only rescales the cost axis.")

    RESULTS["H_pricing"] = {
        "selected_at_current_prices": sorted(cur),
        "always_agent_success": {a: round(succ[a], 3) for a in AG},
        "breakeven_price": be,
        "breakeven_pct_of_current": {a: round(be[a] / cost[a] * 100) for a in AG if a not in cur},
    }


def exp_i_costgap_robust(p, B=2000):
    """CRITICAL robustness for the one positive result. The success-axis ranking is
    only uncertainty-quantified under deconfounded estimators (Exp E/F); the cost-gap
    edge is, in the report, only bootstrapped under the direct method (Exp C). Here we
    paired-bootstrap the cost-gap router's edge over the random and instance mixes at
    29% cheaper under ALL THREE estimators (DM, SNIPW, DR), so the positive claim faces
    the same uncertainty model as the negative one. A CI excluding zero under every
    estimator is what licenses 'robust across estimators' for the cost axis."""
    est, n = p.est, len(p.test_df)
    A = p.test_df.agent.values
    Rv = p.test_df.success.to_numpy()
    e, col, _, _ = _propensity(p)
    Q_actual = est.Q(est.q_va, est.gm_va, A)
    valC = est.Q(est.q_va, est.gm_va, ["Copilot"] * n)      # DM per-row value, route->Copilot
    valX = est.Q(est.q_va, est.gm_va, ["OpenAI_Codex"] * n)  # DM per-row value, route->Codex
    qtrA = {a: est.Q(est.q_tr, est.gm_tr, [a] * n) for a in ("Copilot", "OpenAI_Codex")}
    rng = np.random.default_rng(R.RANDOM_STATE)
    gaps = {
        "random": rng.standard_normal(n),
        "instance": p.scores["Copilot"].to_numpy() - p.scores["OpenAI_Codex"].to_numpy(),
        "cell cost-gap": qtrA["Copilot"] - qtrA["OpenAI_Codex"],
        # non-learned difficulty proxy: rank by Copilot's own cell rate, ignoring Codex.
        # If cost-gap (which subtracts Codex) does not beat this, the "gap" adds nothing.
        "cell Copilot-rate": qtrA["Copilot"],
    }
    frac = 0.4  # ~$2.75, 29% cheaper

    def sel_of(gap):
        k = int(round(frac * n)); s = np.zeros(n, bool); s[np.argsort(-gap)[:k]] = True
        return s

    # per-row contribution arrays per policy, per estimator
    dm_row, dr_row, snipw_num, snipw_den = {}, {}, {}, {}
    for m, g in gaps.items():
        sel = sel_of(g)
        ch = np.where(sel, "Copilot", "OpenAI_Codex")
        eofch = e[np.arange(n), [col[a] for a in ch]]
        match = (A == ch)
        qpi = np.where(sel, valC, valX)          # DM/DR plug-in per row
        dm_row[m] = qpi
        dr_row[m] = qpi + np.where(match, 1.0 / eofch, 0.0) * (Rv - Q_actual)
        snipw_num[m] = np.where(match, Rv / eofch, 0.0)
        snipw_den[m] = np.where(match, 1.0 / eofch, 0.0)

    def diff_ci(est_kind, a, b):
        d = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, n, n)
            if est_kind == "direct":
                d[i] = dm_row[a][idx].mean() - dm_row[b][idx].mean()
            elif est_kind == "dr":
                d[i] = dr_row[a][idx].mean() - dr_row[b][idx].mean()
            else:  # snipw (ratio estimator)
                va = snipw_num[a][idx].sum() / snipw_den[a][idx].sum()
                vb = snipw_num[b][idx].sum() / snipw_den[b][idx].sum()
                d[i] = va - vb
        return _ci(d)

    out = {}
    print("\n[I] cost-gap router edge under all three estimators (29% cheaper, paired bootstrap)")
    for est_kind in ("direct", "snipw", "dr"):
        cr = diff_ci(est_kind, "cell cost-gap", "random")
        cii = diff_ci(est_kind, "cell cost-gap", "instance")
        out[est_kind] = {
            "cell_minus_random": [round(x, 4) for x in cr],
            "cell_minus_instance": [round(x, 4) for x in cii],
            "random_excl_zero": bool(cr[1] > 0),
            "instance_excl_zero": bool(cii[1] > 0),
        }
        print(f"  {est_kind.upper():7} cg-random {cr[0]:+.3f} [{cr[1]:+.3f},{cr[2]:+.3f}]"
              f" {'sig' if cr[1] > 0 else 'NS'};  "
              f"cg-instance {cii[0]:+.3f} [{cii[1]:+.3f},{cii[2]:+.3f}] {'sig' if cii[1] > 0 else 'NS'}")
    robust = all(v["random_excl_zero"] and v["instance_excl_zero"] for v in out.values())
    print(f"=> cost-gap edge CI excludes zero under ALL three estimators: {robust}")

    # Does the gap signal (Copilot - Codex) beat a non-learned difficulty proxy that
    # ranks by Copilot's own cell rate? If not, "subtracting Codex" is doing nothing.
    cg_pt = float(dm_row["cell cost-gap"].mean())
    cr_pt = float(dm_row["cell Copilot-rate"].mean())
    d_cg_cr = diff_ci("direct", "cell cost-gap", "cell Copilot-rate")
    print(f"[I'] non-learned baseline: cost-gap {cg_pt:.3f} vs Copilot-rate {cr_pt:.3f} "
          f"(direct, 29% cheaper); diff {d_cg_cr[0]:+.3f} [{d_cg_cr[1]:+.3f}, {d_cg_cr[2]:+.3f}] "
          f"{'sig' if d_cg_cr[1] > 0 else 'NS'}")

    RESULTS["I_costgap_robust"] = {
        "by_estimator": out,
        "edge_significant_all_estimators": bool(robust),
        "vs_nonlearned_copilot_rate": {
            "cost_gap_direct": round(cg_pt, 3),
            "copilot_rate_direct": round(cr_pt, 3),
            "diff_ci": [round(x, 4) for x in d_cg_cr],
            "gap_signal_adds_value": bool(d_cg_cr[1] > 0),
        },
    }


def exp_j_label_extreme():
    """Bracket the still_open=0.3 heuristic from the other side. exp_d already tested
    the lower extreme (still_open -> 0); here we test the upper extreme (still_open ->
    1, i.e. treat an open PR as an eventual merge). If both extremes preserve the two
    qualitative conclusions, the 0.3 choice is not driving them."""
    up_map = {"merged": 1.0, "still_open": 1.0, "closed_unmerged": 0.0}
    p = R.prepare(success_map=up_map)
    est, n = p.est, len(p.test_df)
    merge_rate = p.train_df.groupby("agent").success.mean().sort_values(ascending=False)
    cell_best = est.q_tr.idxmax(1)
    pols = {
        "Always-Codex": ["OpenAI_Codex"] * n,
        "Instance router": p.scores[R.AGENTS].idxmax(1).values,
        "Cell argmax router": [cell_best.get(c, est.gm_tr.idxmax()) for c in est.tc],
    }
    pvals = {k: est.pval(v) for k, v in pols.items()}

    Qva = lambda ch: est.Q(est.q_va, est.gm_va, ch)
    valC, valX = Qva(["Copilot"] * n), Qva(["OpenAI_Codex"] * n)
    qtrA = {a: est.Q(est.q_tr, est.gm_tr, [a] * n) for a in ("Copilot", "OpenAI_Codex")}
    gap_cell = qtrA["Copilot"] - qtrA["OpenAI_Codex"]
    sel = np.zeros(n, bool); sel[np.argsort(-gap_cell)[:int(0.4 * n)]] = True
    cell_succ = float(np.where(sel, valC, valX).mean())
    rng = np.random.default_rng(R.RANDOM_STATE)
    rsel = np.zeros(n, bool); rsel[rng.permutation(n)[:int(0.4 * n)]] = True
    rand_succ = float(np.where(rsel, valC, valX).mean())

    codex_top = merge_rate.index[0] == "OpenAI_Codex"
    no_router_beats = pvals["Always-Codex"] >= max(pvals["Instance router"], pvals["Cell argmax router"])
    print("\n[J] upper-extreme label robustness (still_open -> 1.0)")
    print("success-only routing:", {k: round(v, 3) for k, v in pvals.items()})
    print(f"frontier @29% cheaper: cell cost-gap {cell_succ:.3f} vs random {rand_succ:.3f}")
    print(f"=> Codex top: {codex_top}; no router beats Codex: {no_router_beats}; "
          f"cell>random under cost: {cell_succ > rand_succ}")
    RESULTS["J_label_extreme_open1"] = {
        "success_only": {k: round(v, 3) for k, v in pvals.items()},
        "frontier_29pct": {"cell_cost_gap": round(cell_succ, 3), "random": round(rand_succ, 3)},
        "codex_top_merge_rate": bool(codex_top),
        "no_router_beats_codex": bool(no_router_beats),
        "cell_beats_random_under_cost": bool(cell_succ > rand_succ),
    }


def main():
    p = R.prepare()
    exp_a_featimp(p)
    exp_b_modelclass(p)
    exp_c_ci(p)
    exp_d_binary()
    exp_e_ipw(p)
    exp_f_twolevel(p)
    exp_g_cost_router(p)
    exp_h_pricing(p)
    exp_i_costgap_robust(p)
    exp_j_label_extreme()
    with open("experiments_results.md", "w") as f:
        f.write("# Experiment results\n\n```\n")
        f.write(json.dumps(RESULTS, indent=2))
        f.write("\n```\n")
    print("\nwrote experiments_results.md")


if __name__ == "__main__":
    main()
