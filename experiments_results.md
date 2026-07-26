# Experiment results

```
{
  "A_featimp": {
    "grouped_gain": [
      {
        "group": "title (MiniLM, 384d)",
        "gain_%": 84.32,
        "n_feat": 384,
        "gain_per_feat_%": 0.22
      },
      {
        "group": "agent",
        "gain_%": 6.26,
        "n_feat": 5,
        "gain_per_feat_%": 1.25
      },
      {
        "group": "language",
        "gain_%": 5.2,
        "n_feat": 25,
        "gain_per_feat_%": 0.21
      },
      {
        "group": "task_type",
        "gain_%": 1.86,
        "n_feat": 12,
        "gain_per_feat_%": 0.16
      },
      {
        "group": "log_stars",
        "gain_%": 1.1,
        "n_feat": 1,
        "gain_per_feat_%": 1.1
      },
      {
        "group": "log_forks",
        "gain_%": 0.93,
        "n_feat": 1,
        "gain_per_feat_%": 0.93
      },
      {
        "group": "has_issue_i",
        "gain_%": 0.33,
        "n_feat": 1,
        "gain_per_feat_%": 0.33
      }
    ],
    "top15_individual_gain_pct": {
      "agent_OpenAI_Codex": 2.38,
      "language_Unknown": 1.55,
      "agent_Cursor": 1.23,
      "log_stars": 1.1,
      "agent_Copilot": 0.99,
      "agent_Claude_Code": 0.97,
      "log_forks": 0.93,
      "task_type_fix": 0.88,
      "language_HTML": 0.77,
      "agent_Devin": 0.7,
      "emb_055": 0.54,
      "task_type_test": 0.52,
      "language_C++": 0.5,
      "language_OCaml": 0.49,
      "task_type_other": 0.46
    },
    "title_block_gain_pct": 84.32,
    "base_test_mae": 0.3058,
    "grouped_permutation_mae_increase": {
      "agent": 0.0161,
      "language": 0.0129,
      "title (MiniLM, 384d)": 0.0079,
      "log_forks": 0.0016,
      "log_stars": 0.0011,
      "task_type": 0.0002,
      "has_issue_i": 0.0001
    }
  },
  "B_modelclass": {
    "test_mae": {
      "per-agent-mean baseline": 0.331,
      "Ridge (linear, alpha=10)": 0.306,
      "MLP (128, 64) a=0.0001": 0.35,
      "XGBoost (tuned, our model)": 0.306
    },
    "spread": 0.044,
    "ridge_alpha_val_sweep": {
      "alpha=0.1": 0.304,
      "alpha=1": 0.304,
      "alpha=10": 0.304,
      "alpha=100": 0.304
    },
    "best_alpha": 10.0,
    "mlp_val_sweep": [
      {
        "cfg": "{'hidden_layer_sizes': (128, 64), 'alpha': 0.0001}",
        "val_mae": 0.348
      },
      {
        "cfg": "{'hidden_layer_sizes': (64,), 'alpha': 0.0001}",
        "val_mae": 0.353
      },
      {
        "cfg": "{'hidden_layer_sizes': (128,), 'alpha': 0.01}",
        "val_mae": 0.363
      },
      {
        "cfg": "{'hidden_layer_sizes': (128,), 'alpha': 0.0001}",
        "val_mae": 0.364
      }
    ]
  },
  "C_ci": {
    "success_only": {
      "Always-Codex": {
        "mean": 0.88,
        "lo": 0.876,
        "hi": 0.883
      },
      "Instance router": {
        "mean": 0.876,
        "lo": 0.872,
        "hi": 0.879
      },
      "Cell argmax router": {
        "mean": 0.862,
        "lo": 0.857,
        "hi": 0.866
      },
      "Random": {
        "mean": 0.759,
        "lo": 0.753,
        "hi": 0.764
      },
      "Always-cheapest": {
        "mean": 0.682,
        "lo": 0.677,
        "hi": 0.687
      }
    },
    "diff_codex_minus_instance": [
      0.0037,
      0.0021,
      0.0053
    ],
    "diff_codex_minus_cellargmax": [
      0.0178,
      0.0144,
      0.0213
    ],
    "frontier": {
      "$3.30/14% cheaper": {
        "ci": {
          "random": [
            0.841,
            0.836,
            0.846
          ],
          "instance": [
            0.839,
            0.835,
            0.844
          ],
          "cell cost-gap": [
            0.855,
            0.851,
            0.859
          ]
        },
        "cell_minus_random": [
          0.014,
          0.009,
          0.0187
        ],
        "cell_minus_instance": [
          0.0156,
          0.011,
          0.0204
        ]
      },
      "$2.75/29% cheaper": {
        "ci": {
          "random": [
            0.801,
            0.795,
            0.806
          ],
          "instance": [
            0.803,
            0.797,
            0.808
          ],
          "cell cost-gap": [
            0.821,
            0.817,
            0.826
          ]
        },
        "cell_minus_random": [
          0.0207,
          0.0145,
          0.0267
        ],
        "cell_minus_instance": [
          0.0188,
          0.0133,
          0.0246
        ]
      },
      "$2.20/43% cheaper": {
        "ci": {
          "random": [
            0.761,
            0.756,
            0.767
          ],
          "instance": [
            0.763,
            0.757,
            0.768
          ],
          "cell cost-gap": [
            0.787,
            0.782,
            0.792
          ]
        },
        "cell_minus_random": [
          0.0261,
          0.0192,
          0.0323
        ],
        "cell_minus_instance": [
          0.0245,
          0.0188,
          0.0306
        ]
      }
    }
  },
  "D_binary": {
    "merge_rate_train": {
      "OpenAI_Codex": 0.876,
      "Claude_Code": 0.756,
      "Cursor": 0.73,
      "Devin": 0.64,
      "Copilot": 0.579
    },
    "success_only_binary": {
      "Always-Codex": 0.852,
      "Instance router": 0.845,
      "Cell argmax router": 0.825
    },
    "frontier_29pct": {
      "cell_cost_gap": 0.777,
      "random": 0.757
    },
    "codex_top_merge_rate": true,
    "no_router_beats_codex": true,
    "cell_beats_random_under_cost": true
  },
  "E_ipw": {
    "propensity_test_accuracy": 0.36,
    "positivity": {
      "clip_floor": 0.02,
      "min_propensity_overall": 0.0009,
      "min_propensity_actual_agent": 0.0038,
      "frac_actual_below_floor": 0.001,
      "frac_any_cell_below_floor": 0.017
    },
    "policy_values": {
      "Always-Codex": {
        "direct": 0.88,
        "snipw": 0.841,
        "dr": 0.847,
        "n_match": 757,
        "ipw_ess": 484.0,
        "max_gap": 0.039
      },
      "Instance router": {
        "direct": 0.876,
        "snipw": 0.857,
        "dr": 0.862,
        "n_match": 780,
        "ipw_ess": 525.0,
        "max_gap": 0.019
      },
      "Cell argmax router": {
        "direct": 0.862,
        "snipw": 0.824,
        "dr": 0.831,
        "n_match": 686,
        "ipw_ess": 375.0,
        "max_gap": 0.038
      },
      "Random": {
        "direct": 0.759,
        "snipw": 0.781,
        "dr": 0.776,
        "n_match": 728,
        "ipw_ess": 444.0,
        "max_gap": 0.022
      },
      "Always-cheapest": {
        "direct": 0.682,
        "snipw": 0.678,
        "dr": 0.682,
        "n_match": 847,
        "ipw_ess": 581.0,
        "max_gap": 0.004
      }
    },
    "best_under_each_estimator": {
      "direct": "Always-Codex",
      "snipw": "Instance router",
      "dr": "Instance router"
    },
    "codex_top_under_all_estimators": false,
    "cost_gap_frontier_29pct": {
      "random": {
        "direct": 0.801,
        "snipw": 0.772,
        "dr": 0.786
      },
      "instance": {
        "direct": 0.803,
        "snipw": 0.773,
        "dr": 0.792
      },
      "cell cost-gap": {
        "direct": 0.821,
        "snipw": 0.789,
        "dr": 0.797
      }
    },
    "cost_gap_best_under_each_estimator": {
      "direct": "cell cost-gap",
      "snipw": "cell cost-gap",
      "dr": "cell cost-gap"
    },
    "cost_gap_robust_all_estimators": true
  },
  "F_twolevel": {
    "ci": {
      "Always-Codex": [
        0.879,
        0.854,
        0.903
      ],
      "Instance router": [
        0.876,
        0.851,
        0.898
      ],
      "Cell argmax router": [
        0.859,
        0.833,
        0.886
      ]
    },
    "diff_codex_minus_instance": [
      0.0036,
      -0.0002,
      0.0075
    ],
    "diff_significant": false,
    "costgap_minus_random": [
      0.023,
      0.0028,
      0.0439
    ],
    "costgap_minus_instance": [
      0.0203,
      0.0006,
      0.04
    ],
    "costgap_edge_significant_twolevel": true
  },
  "G_cost_router": {
    "pareto_efficient_agents": [
      "OpenAI_Codex",
      "Copilot"
    ],
    "success_per_dollar": {
      "OpenAI_Codex": 0.228,
      "Copilot": 0.62,
      "Devin": 0.143,
      "Cursor": 0.196,
      "Claude_Code": 0.174
    },
    "always_agent_success": {
      "OpenAI_Codex": 0.88,
      "Copilot": 0.682,
      "Devin": 0.643,
      "Cursor": 0.756,
      "Claude_Code": 0.835
    },
    "cost_gap_knees": {
      "frac_0.0": [
        3.85,
        0.88
      ],
      "frac_0.2": [
        3.3,
        0.855
      ],
      "frac_0.4": [
        2.75,
        0.821
      ],
      "frac_0.6": [
        2.2,
        0.787
      ],
      "frac_1.0": [
        1.1,
        0.682
      ]
    },
    "naive_lambda_at_275": 0.809,
    "cost_gap_at_275": 0.821
  },
  "H_pricing": {
    "selected_at_current_prices": [
      "Copilot",
      "OpenAI_Codex"
    ],
    "always_agent_success": {
      "OpenAI_Codex": 0.88,
      "Copilot": 0.682,
      "Devin": 0.643,
      "Cursor": 0.756,
      "Claude_Code": 0.835
    },
    "breakeven_price": {
      "OpenAI_Codex": 3.85,
      "Copilot": 1.1,
      "Devin": 1.08,
      "Cursor": 2.1,
      "Claude_Code": 3.21
    },
    "breakeven_pct_of_current": {
      "Devin": 24,
      "Cursor": 55,
      "Claude_Code": 67
    }
  },
  "I_costgap_robust": {
    "by_estimator": {
      "direct": {
        "cell_minus_random": [
          0.0217,
          0.0161,
          0.0274
        ],
        "cell_minus_instance": [
          0.0188,
          0.0131,
          0.0245
        ],
        "random_excl_zero": true,
        "instance_excl_zero": true
      },
      "snipw": {
        "cell_minus_random": [
          0.032,
          -0.0116,
          0.0752
        ],
        "cell_minus_instance": [
          0.0149,
          -0.026,
          0.0547
        ],
        "random_excl_zero": false,
        "instance_excl_zero": false
      },
      "dr": {
        "cell_minus_random": [
          0.028,
          -0.0192,
          0.0758
        ],
        "cell_minus_instance": [
          0.0043,
          -0.038,
          0.0456
        ],
        "random_excl_zero": false,
        "instance_excl_zero": false
      }
    },
    "edge_significant_all_estimators": false,
    "vs_nonlearned_copilot_rate": {
      "cost_gap_direct": 0.821,
      "copilot_rate_direct": 0.811,
      "diff_ci": [
        0.0106,
        0.0069,
        0.0146
      ],
      "gap_signal_adds_value": true
    }
  },
  "J_label_extreme_open1": {
    "success_only": {
      "Always-Codex": 0.944,
      "Instance router": 0.932,
      "Cell argmax router": 0.934
    },
    "frontier_29pct": {
      "cell_cost_gap": 0.914,
      "random": 0.899
    },
    "codex_top_merge_rate": true,
    "no_router_beats_codex": true,
    "cell_beats_random_under_cost": true
  }
}
```
