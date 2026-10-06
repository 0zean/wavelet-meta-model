# 5Min pilot rule (experiments/specs/u11_a1.yaml)

Verdict: **run** — gate: every one-sided 95 % session-bootstrap upper bound on OOS meta AUC < 0.52.

| label | n_events | n_sessions | auc | auc_ub95 | status | below_gate |
|---|---|---|---|---|---|---|
| SPY_5Min_wavelet_trend_xgb_oof_fixed_none | 37451 | 2380 | 0.530 | 0.536 | ok | False |
| SPY_5Min_wavelet_trend_rf_ldp_oof_fixed_none | 37451 | 2380 | 0.536 | 0.542 | ok | False |
| SPY_5Min_sma_cross_xgb_oof_fixed_none | 37451 | 2380 | 0.518 | 0.524 | ok | False |
| SPY_5Min_sma_cross_rf_ldp_oof_fixed_none | 37451 | 2380 | 0.523 | 0.530 | ok | False |
| SPY_5Min_bollinger_mr_xgb_oof_fixed_none | 37451 | 2380 | 0.526 | 0.532 | ok | False |
| SPY_5Min_bollinger_mr_rf_ldp_oof_fixed_none | 37451 | 2380 | 0.533 | 0.539 | ok | False |
| SPY_5Min_donchian_breakout_xgb_oof_fixed_none | 37451 | 2380 | 0.531 | 0.537 | ok | False |
| SPY_5Min_donchian_breakout_rf_ldp_oof_fixed_none | 37451 | 2380 | 0.536 | 0.542 | ok | False |
| SPY_5Min_ml_xgb_xgb_oof_fixed_none | 37451 | 2380 | 0.520 | 0.525 | ok | False |
| SPY_5Min_ml_xgb_rf_ldp_oof_fixed_none | 37451 | 2380 | 0.526 | 0.531 | ok | False |
| AAPL_5Min_wavelet_trend_xgb_oof_fixed_none | 37086 | 2380 | 0.513 | 0.519 | ok | True |
| AAPL_5Min_wavelet_trend_rf_ldp_oof_fixed_none | 37086 | 2380 | 0.519 | 0.525 | ok | False |
| AAPL_5Min_sma_cross_xgb_oof_fixed_none | 37086 | 2380 | 0.509 | 0.515 | ok | True |
| AAPL_5Min_sma_cross_rf_ldp_oof_fixed_none | 37086 | 2380 | 0.515 | 0.521 | ok | False |
| AAPL_5Min_bollinger_mr_xgb_oof_fixed_none | 37086 | 2380 | 0.514 | 0.520 | ok | True |
| AAPL_5Min_bollinger_mr_rf_ldp_oof_fixed_none | 37086 | 2380 | 0.515 | 0.521 | ok | False |
| AAPL_5Min_donchian_breakout_xgb_oof_fixed_none | 37086 | 2380 | 0.520 | 0.526 | ok | False |
| AAPL_5Min_donchian_breakout_rf_ldp_oof_fixed_none | 37086 | 2380 | 0.524 | 0.529 | ok | False |
| AAPL_5Min_ml_xgb_xgb_oof_fixed_none | 37086 | 2380 | 0.504 | 0.510 | ok | True |
| AAPL_5Min_ml_xgb_rf_ldp_oof_fixed_none | 37086 | 2380 | 0.509 | 0.515 | ok | True |
| TLT_5Min_wavelet_trend_xgb_oof_fixed_none | 31901 | 2380 | 0.513 | 0.519 | ok | True |
| TLT_5Min_wavelet_trend_rf_ldp_oof_fixed_none | 31901 | 2380 | 0.518 | 0.524 | ok | False |
| TLT_5Min_sma_cross_xgb_oof_fixed_none | 31901 | 2380 | 0.511 | 0.518 | ok | True |
| TLT_5Min_sma_cross_rf_ldp_oof_fixed_none | 31901 | 2380 | 0.514 | 0.521 | ok | False |
| TLT_5Min_bollinger_mr_xgb_oof_fixed_none | 31901 | 2380 | 0.510 | 0.516 | ok | True |
| TLT_5Min_bollinger_mr_rf_ldp_oof_fixed_none | 31901 | 2380 | 0.515 | 0.521 | ok | False |
| TLT_5Min_donchian_breakout_xgb_oof_fixed_none | 31901 | 2380 | 0.514 | 0.520 | ok | False |
| TLT_5Min_donchian_breakout_rf_ldp_oof_fixed_none | 31901 | 2380 | 0.518 | 0.525 | ok | False |
| TLT_5Min_ml_xgb_xgb_oof_fixed_none | 31901 | 2380 | 0.510 | 0.516 | ok | True |
| TLT_5Min_ml_xgb_rf_ldp_oof_fixed_none | 31901 | 2380 | 0.517 | 0.523 | ok | False |
