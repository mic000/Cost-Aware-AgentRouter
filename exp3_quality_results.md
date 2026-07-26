# Experiment #3: quality-metric Pareto (exploratory, not in report)

Per-agent (full pop set, reviewed where applicable):

```
                    n  cost  merge  reviewed_frac  clean_among_reviewed  reviewed_clean_all
Claude_Code     459.0  4.80  0.590          0.416                 0.618               0.257
Copilot        4970.0  1.10  0.430          0.540                 0.555               0.300
Cursor         1541.0  3.85  0.652          0.501                 0.690               0.346
Devin          4827.0  4.50  0.538          0.445                 0.688               0.306
OpenAI_Codex  21799.0  3.85  0.826          0.108                 0.749               0.081
```

```
{
  "pareto_by_metric": {
    "merge": [
      "Copilot",
      "OpenAI_Codex"
    ],
    "clean_among_reviewed": [
      "Copilot",
      "OpenAI_Codex"
    ],
    "reviewed_clean_all": [
      "Copilot",
      "Cursor"
    ]
  },
  "success_per_dollar_by_metric": {
    "merge": {
      "Copilot": 0.391,
      "OpenAI_Codex": 0.215,
      "Cursor": 0.169,
      "Claude_Code": 0.123,
      "Devin": 0.119
    },
    "clean_among_reviewed": {
      "Copilot": 0.505,
      "OpenAI_Codex": 0.195,
      "Cursor": 0.179,
      "Claude_Code": 0.129,
      "Devin": 0.153
    },
    "reviewed_clean_all": {
      "Copilot": 0.272,
      "OpenAI_Codex": 0.021,
      "Cursor": 0.09,
      "Claude_Code": 0.054,
      "Devin": 0.068
    }
  },
  "cursor_reenters_under_quality": true,
  "coverage_note": "reviewed_frac is small and agent-dependent; quality metrics are descriptive, not a router"
}
```
