"""Unit tests for the v2 router pipeline and the dataset build.

Fast, deterministic checks (no model training): run with `pytest test_router.py`.
The heavy end-to-end reproduction gate (trains XGBoost, reprints the headline
MAE/routing numbers) lives in `python router_lib.py` and is exercised separately.

These tests guard the parts most likely to break silently:
  - the leakage-stripping title cleaner (agent tokens / footers must go),
  - the label mapping and feature derivation,
  - the balanced corpus invariants (25,580 rows, 5,116 per agent),
  - the repo-grouped split (no repo leaks across train/val/test),
  - the direct-method cell estimator (cell lookup + base-rate fallback, [0,1] range).
"""
import os

import numpy as np
import pandas as pd
import pytest

import router_lib as R

DATA_AVAILABLE = os.path.exists(R.DATA_PATH)
needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="router_dataset.jsonl not present")


# ---------------------------------------------------------------- title cleaner
def test_clean_title_strips_agent_tokens():
    assert "copilot" not in R.clean_title("Copilot: fix the bug").lower()
    assert "codex" not in R.clean_title("OpenAI Codex added a test").lower()
    for tok in ("devin", "cursor", "claude", "anthropic"):
        assert tok not in R.clean_title(f"{tok} refactor module").lower()


def test_clean_title_strips_footers_and_whitespace():
    out = R.clean_title("Fix bug\n\nCo-authored-by: bot\nGenerated with tool")
    assert "co-authored-by" not in out.lower()
    assert "generated with" not in out.lower()
    assert "  " not in out  # whitespace collapsed
    assert out == out.strip()


def test_clean_title_keeps_real_content():
    assert "fix" in R.clean_title("fix null pointer in parser").lower()


# ---------------------------------------------------------------- config sanity
def test_cost_covers_all_agents():
    assert set(R.COST) == set(R.AGENTS)
    assert all(v > 0 for v in R.COST.values())


def test_success_map_values():
    assert R.SUCCESS_MAP == {"merged": 1.0, "still_open": 0.3, "closed_unmerged": 0.0}
    assert set(R.SUCCESS_MAP.values()) <= {0.0, 0.3, 1.0}


# ------------------------------------------------------------ dataset invariants
@needs_data
def test_dataset_balanced_and_labeled():
    df = R.load_df()
    assert len(df) == 25580
    counts = df.agent.value_counts()
    assert set(counts.index) == set(R.AGENTS)
    assert counts.nunique() == 1 and counts.iloc[0] == 5116  # 5 x 5,116 = 25,580
    assert df.success.isin([0.0, 0.3, 1.0]).all()
    assert not df.success.isna().any()
    for col in ("repo_id", "language", "log_stars", "log_forks", "has_issue_i", "task_type"):
        assert col in df.columns


@needs_data
def test_no_post_treatment_body_feature():
    # the v2 numeric feature set must not contain any body-derived column
    assert "body" not in R.NUM
    assert all("body" not in c for c in R.NUM)


@needs_data
def test_split_has_no_repo_leak():
    df = R.load_df()
    tr, va, te = R.split_df(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not (set(tr.repo_id) & set(va.repo_id))
    assert not (set(tr.repo_id) & set(te.repo_id))
    assert not (set(va.repo_id) & set(te.repo_id))


# ---------------------------------------------------------------- cell estimator
def _toy_frames():
    """Two cells x five agents, deterministic success, for estimator unit tests."""
    rng = np.random.default_rng(0)
    rows = []
    for lang in ("Python", "JavaScript"):
        for tt in ("fix", "feat"):
            for ag in R.AGENTS:
                base = 0.9 if ag == "OpenAI_Codex" else 0.5
                for _ in range(8):
                    rows.append(dict(language=lang, task_type=tt, agent=ag,
                                     success=float(rng.random() < base)))
    return pd.DataFrame(rows)


def test_cell_estimator_range_and_fallback():
    df = _toy_frames()
    tr, va, te = df.sample(frac=0.6, random_state=1), df.sample(frac=0.2, random_state=2), df.sample(frac=0.2, random_state=3)
    tr, va, te = tr.copy(), va.copy(), te.copy()
    est = R.make_cell_estimator(tr, va, te)
    n = len(te)
    for ag in R.AGENTS:
        v = est.Q(est.q_va, est.gm_va, [ag] * n)
        assert v.shape == (n,)
        assert np.all((v >= 0.0) & (v <= 1.0))
        assert not np.isnan(v).any()
    # pval is a mean of in-range per-row values
    assert 0.0 <= est.pval(["OpenAI_Codex"] * n) <= 1.0


def test_cell_estimator_unseen_cell_uses_base_rate():
    df = _toy_frames()
    tr = df.copy()
    va = df.copy()
    te = pd.DataFrame([dict(language="ZZUnknownLang", task_type="zzunknown", agent="Devin", success=0.0)])
    est = R.make_cell_estimator(tr, va, te)
    val = est.Q(est.q_va, est.gm_va, ["Devin"])
    # unseen cell -> falls back to the per-agent base rate, not NaN
    assert not np.isnan(val).any()
    assert abs(val[0] - est.gm_va["Devin"]) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
