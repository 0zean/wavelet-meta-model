# Stage A survivors

Rule: meta AUC > 0.515, PSR > 0.5; top 20 by DSR, at most 2 per (symbol, timeframe); xgb rows only. 475 cells, 54 passed the gates.

| symbol | timeframe | primary | meta_auc | psr | sharpe | dsr | n_trials | n_trades | label | cell_hash | passed | survivor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | 1Day | ml_xgb | 0.588 | 0.968 | 0.933 | 0.041 | 737 | 62 | NVDA_1Day_ml_xgb_xgb_oof_fixed_none | 8f453ed603ad09d6 | True | True |
| XLE | 1Day | ml_xgb | 0.540 | 0.832 | 0.501 | 0.006 | 737 | 82 | XLE_1Day_ml_xgb_xgb_oof_fixed_none | 91f686cb6bde2b92 | True | True |
| QQQ | 15Min | bollinger_mr | 0.532 | 1.000 | 1.096 | 0.004 | 737 | 739 | QQQ_15Min_bollinger_mr_xgb_oof_fixed_none | 7b1ec750bdf686e8 | True | True |
| GLD | 1Day | bollinger_mr | 0.547 | 0.796 | 0.428 | 0.004 | 737 | 66 | GLD_1Day_bollinger_mr_xgb_oof_fixed_none | cc932fae4bd3f3f8 | True | True |
| QQQ | 1Day | ml_xgb | 0.557 | 0.772 | 0.387 | 0.003 | 737 | 80 | QQQ_1Day_ml_xgb_xgb_oof_fixed_none | 26dd16ef4275f74f | True | True |
| SPY | 1Day | wavelet_trend | 0.531 | 0.769 | 0.381 | 0.003 | 737 | 92 | SPY_1Day_wavelet_trend_xgb_oof_fixed_none | 27a465facf38fabf | True | True |
| JPM | 1Day | ml_xgb | 0.523 | 0.735 | 0.324 | 0.002 | 737 | 61 | JPM_1Day_ml_xgb_xgb_oof_fixed_none | 0e753fbf81604a38 | True | True |
| MSFT | 1Day | bollinger_mr | 0.519 | 0.704 | 0.278 | 0.002 | 737 | 66 | MSFT_1Day_bollinger_mr_xgb_oof_fixed_none | ab1e90b25b804d7f | True | True |
| XLV | 1Day | wavelet_trend | 0.597 | 0.702 | 0.275 | 0.002 | 737 | 76 | XLV_1Day_wavelet_trend_xgb_oof_fixed_none | a07dc16efe87aa1c | True | True |
| XLK | 1Day | bollinger_mr | 0.540 | 0.662 | 0.218 | 0.001 | 737 | 24 | XLK_1Day_bollinger_mr_xgb_oof_fixed_none | 10d935799d1f6308 | True | True |
| SPY | 1Day | ml_xgb | 0.525 | 0.647 | 0.197 | 0.001 | 737 | 70 | SPY_1Day_ml_xgb_xgb_oof_fixed_none | 2a882f4a2c70c459 | True | True |
| AAPL | 1Day | bollinger_mr | 0.574 | 0.603 | 0.136 | 0.001 | 737 | 19 | AAPL_1Day_bollinger_mr_xgb_oof_fixed_none | 692d01bf3f30f26a | True | True |
| GOOGL | 1Day | bollinger_mr | 0.525 | 0.560 | 0.079 | 0.000 | 737 | 82 | GOOGL_1Day_bollinger_mr_xgb_oof_fixed_none | 1c6d95b86451b62c | True | True |
| GOOGL | 1Day | wavelet_trend | 0.517 | 0.548 | 0.063 | 0.000 | 737 | 53 | GOOGL_1Day_wavelet_trend_xgb_oof_fixed_none | 5d3e6ec2625e760c | True | True |
| QQQ | 1Day | bollinger_mr | 0.526 | 0.536 | 0.046 | 0.000 | 737 | 32 | QQQ_1Day_bollinger_mr_xgb_oof_fixed_none | 62103166d5de605f | True | True |
| AMZN | 1Day | ml_xgb | 0.530 | 0.528 | 0.037 | 0.000 | 737 | 94 | AMZN_1Day_ml_xgb_xgb_oof_fixed_none | 80091af7b0dc1351 | True | True |
| AAPL | 1Day | sma_cross | 0.552 | 0.513 | 0.017 | 0.000 | 737 | 72 | AAPL_1Day_sma_cross_xgb_oof_fixed_none | fdc6e3f0a9739c97 | True | True |
| XLK | 30Min | wavelet_trend | 0.521 | 0.954 | 0.576 | 0.000 | 737 | 2187 | XLK_30Min_wavelet_trend_xgb_oof_fixed_none | 49fd7b1912c53242 | True | True |
| AMZN | 15Min | ml_xgb | 0.516 | 0.965 | 0.598 | 0.000 | 737 | 4171 | AMZN_15Min_ml_xgb_xgb_oof_fixed_none | 10daa31f95ebd4af | True | True |
| XLE | 1Hour | donchian_breakout | 0.541 | 0.940 | 0.533 | 0.000 | 737 | 908 | XLE_1Hour_donchian_breakout_xgb_oof_fixed_none | f9de018835d8840a | True | True |
