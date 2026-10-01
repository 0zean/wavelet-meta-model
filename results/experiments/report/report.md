# Experiment ledger report

- Ledger: /Users/nick/Documents/Python Projects/wavelet/results/ledger.jsonl — 181 trial rows, 0 cell(s) in error (errors are not counted in N).
- N = distinct counted (ok / no_fit) cell hashes through the stage, a PWFO cell counting its grid size.
- Holdout accessed: never.

## Stage funnel

| stage | cells | ok | no_fit | error | cache_hits | trials_in_stage | n_trials_cumulative |
|---|---|---|---|---|---|---|---|
| U6 | 35 | 35 | 0 | 0 | 0 | 35 | 35 |
| U7 | 100 | 100 | 0 | 0 | 0 | 100 | 135 |
| U8 | 4 | 4 | 0 | 0 | 0 | 4 | 139 |
| U9 | 26 | 18 | 8 | 0 | 0 | 24 | 163 |
| U10 | 16 | 16 | 0 | 0 | 0 | 16 | 179 |

## PBO per stage

| stage | n_cells | n_distinct | n_common_days | pbo | prob_oos_loss | degradation_slope |
|---|---|---|---|---|---|---|
| U10 | 16 | 8 | 2070 | 0.466 | 0.898 | -0.687 |

## Leaderboard (top 50 per stage by DSR)

| stage | label | kind | sharpe | psr | dsr | n_trials | dsr_all | n_trials_all | ret_ann | max_dd | meta_auc | n_trades | n_obs | wfe | pbo | cache_hit | cell_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U6 | SPY_1Day_wavelet_trend_logit_l2 | legacy | 1.073 |  |  | 35 |  | 179 |  |  | 0.576 | 69.000 |  |  |  | False | legacy-73a64c5c44185f1c |
| U6 | SPY_1Day_wavelet_trend_legacy | legacy | 0.937 |  |  | 35 |  | 179 |  |  | 0.545 | 67.000 |  |  |  | False | legacy-0fd42cb9392ce6f5 |
| U6 | SPY_1Day_wavelet_trend_logit_l1 | legacy | 0.876 |  |  | 35 |  | 179 |  |  | 0.575 | 67.000 |  |  |  | False | legacy-4c2391e0b0c53100 |
| U6 | SPY_1Day_wavelet_trend_legacy | legacy | 0.603 |  |  | 35 |  | 179 |  |  | 0.531 | 69.000 |  |  |  | False | legacy-f3a5bb7bc89e672a |
| U6 | SPY_1Day_wavelet_trend_rf_ldp | legacy | 0.599 |  |  | 35 |  | 179 |  |  | 0.579 | 64.000 |  |  |  | False | legacy-114f14c27312b28d |
| U6 | SPY_1Day_wavelet_trend_logit_l1 | legacy | 0.586 |  |  | 35 |  | 179 |  |  | 0.433 | 88.000 |  |  |  | False | legacy-de512241bad1defa |
| U6 | SPY_1Day_wavelet_trend_logit_l2 | legacy | 0.462 |  |  | 35 |  | 179 |  |  | 0.529 | 72.000 |  |  |  | False | legacy-99015a017a6dda68 |
| U6 | SPY_1Day_wavelet_trend_lightgbm | legacy | 0.449 |  |  | 35 |  | 179 |  |  | 0.522 | 90.000 |  |  |  | False | legacy-fa9802609fc9bc42 |
| U6 | SPY_1Day_ml_xgb_extra_trees | legacy | 0.425 |  |  | 35 |  | 179 |  |  | 0.433 | 72.000 |  |  |  | False | legacy-0088141832f67d20 |
| U6 | SPY_1Day_ml_xgb_lightgbm | legacy | 0.363 |  |  | 35 |  | 179 |  |  | 0.430 | 75.000 |  |  |  | False | legacy-ffdc281d48ee865b |
| U6 | SPY_1Day_ml_xgb_logit_l1 | legacy | 0.361 |  |  | 35 |  | 179 |  |  | 0.430 | 73.000 |  |  |  | False | legacy-b961fd4e5691d37d |
| U6 | SPY_1Day_ml_xgb_rf_ldp | legacy | 0.345 |  |  | 35 |  | 179 |  |  | 0.427 | 73.000 |  |  |  | False | legacy-0e297c155882740a |
| U6 | SPY_1Day_ml_xgb_logit_l2 | legacy | 0.342 |  |  | 35 |  | 179 |  |  | 0.443 | 73.000 |  |  |  | False | legacy-9ebb9a320e30e76a |
| U6 | SPY_1Day_ml_xgb_xgb | legacy | 0.336 |  |  | 35 |  | 179 |  |  | 0.450 | 73.000 |  |  |  | False | legacy-ca52959c12faea52 |
| U6 | SPY_1Day_wavelet_trend_extra_trees | legacy | 0.316 |  |  | 35 |  | 179 |  |  | 0.568 | 59.000 |  |  |  | False | legacy-688c77bcfc2fdc3d |
| U6 | SPY_1Day_ml_xgb_legacy | legacy | 0.119 |  |  | 35 |  | 179 |  |  | 0.507 | 68.000 |  |  |  | False | legacy-5a9f1486ada3fbc5 |
| U6 | SPY_1Day_wavelet_trend_xgb | legacy | -0.020 |  |  | 35 |  | 179 |  |  | 0.475 | 77.000 |  |  |  | False | legacy-11165f491860a355 |
| U6 | SPY_1Day_wavelet_trend_xgb | legacy | -0.074 |  |  | 35 |  | 179 |  |  | 0.540 | 83.000 |  |  |  | False | legacy-79926efe6123460f |
| U6 | SPY_1Day_wavelet_trend_lightgbm | legacy | -0.078 |  |  | 35 |  | 179 |  |  | 0.470 | 79.000 |  |  |  | False | legacy-d534c76f7b4f5fc1 |
| U6 | SPY_1Day_wavelet_trend_rf_ldp | legacy | -0.145 |  |  | 35 |  | 179 |  |  | 0.542 | 75.000 |  |  |  | False | legacy-0dc05dbab5ffde26 |
| U6 | SPY_1Day_wavelet_trend_extra_trees | legacy | -0.226 |  |  | 35 |  | 179 |  |  | 0.529 | 78.000 |  |  |  | False | legacy-df24f3cf61f73b52 |
| U6 | SPY_5Min_wavelet_trend_logit_l1 | legacy | -0.764 |  |  | 35 |  | 179 |  |  | 0.509 | 1372.000 |  |  |  | False | legacy-2dccedd0c395dca2 |
| U6 | SPY_5Min_wavelet_trend_logit_l2 | legacy | -0.984 |  |  | 35 |  | 179 |  |  | 0.495 | 281.000 |  |  |  | False | legacy-89ff7d025ca7a97a |
| U6 | SPY_5Min_wavelet_trend_rf_ldp | legacy | -1.165 |  |  | 35 |  | 179 |  |  | 0.500 | 1466.000 |  |  |  | False | legacy-e992a34ef53df28a |
| U6 | SPY_5Min_wavelet_trend_logit_l2 | legacy | -1.214 |  |  | 35 |  | 179 |  |  | 0.498 | 1474.000 |  |  |  | False | legacy-43a84fb160f93d36 |
| U6 | SPY_5Min_wavelet_trend_lightgbm | legacy | -1.494 |  |  | 35 |  | 179 |  |  | 0.499 | 1335.000 |  |  |  | False | legacy-effdb248fac3b1ee |
| U6 | SPY_5Min_wavelet_trend_extra_trees | legacy | -1.540 |  |  | 35 |  | 179 |  |  | 0.508 | 1478.000 |  |  |  | False | legacy-e80acd642b55909d |
| U6 | SPY_5Min_wavelet_trend_xgb | legacy | -2.066 |  |  | 35 |  | 179 |  |  | 0.500 | 1413.000 |  |  |  | False | legacy-c65758594504a9af |
| U6 | SPY_5Min_wavelet_trend_logit_l1 | legacy | -2.111 |  |  | 35 |  | 179 |  |  | 0.494 | 349.000 |  |  |  | False | legacy-25dda4a7cd746388 |
| U6 | SPY_5Min_wavelet_trend_xgb | legacy | -2.584 |  |  | 35 |  | 179 |  |  | 0.512 | 353.000 |  |  |  | False | legacy-d714a1d07138b10a |
| U6 | SPY_5Min_wavelet_trend_legacy | legacy | -2.671 |  |  | 35 |  | 179 |  |  | 0.502 | 2288.000 |  |  |  | False | legacy-a663899b7edd6c50 |
| U6 | SPY_5Min_wavelet_trend_legacy | legacy | -2.699 |  |  | 35 |  | 179 |  |  | 0.509 | 455.000 |  |  |  | False | legacy-78849b57ae903c9b |
| U6 | SPY_5Min_wavelet_trend_extra_trees | legacy | -3.117 |  |  | 35 |  | 179 |  |  | 0.496 | 302.000 |  |  |  | False | legacy-d7f8cc1fe6b01c5a |
| U6 | SPY_5Min_wavelet_trend_rf_ldp | legacy | -3.523 |  |  | 35 |  | 179 |  |  | 0.500 | 304.000 |  |  |  | False | legacy-942ecac8b356808b |
| U6 | SPY_5Min_wavelet_trend_lightgbm | legacy | -4.036 |  |  | 35 |  | 179 |  |  | 0.505 | 333.000 |  |  |  | False | legacy-a708597782ec5f64 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_fixed_single | legacy | 1.073 |  |  | 135 |  | 179 | 0.121 | -0.079 | 0.576 | 69.000 |  |  |  | False | legacy-2ee9fd1c20eddb17 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_fixed_single | legacy | 1.073 |  |  | 135 |  | 179 | 0.121 | -0.079 | 0.576 | 69.000 |  |  |  | False | legacy-cc55ee478fdfaa12 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_fixed_average | legacy | 0.876 |  |  | 135 |  | 179 | 0.102 | -0.124 | 0.576 | 159.000 |  |  |  | False | legacy-98fab9fab28b47ea |
| U7 | SPY_1Day_wavelet_trend_logit_l2_fixed_average | legacy | 0.872 |  |  | 135 |  | 179 | 0.102 | -0.124 | 0.576 | 159.000 |  |  |  | False | legacy-ed11250ff3b1aa2e |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ecdf_single | legacy | 0.767 |  |  | 135 |  | 179 | 0.051 | -0.058 | 0.576 | 66.000 |  |  |  | False | legacy-e7336ff3b30aa723 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ecdf_single | legacy | 0.767 |  |  | 135 |  | 179 | 0.051 | -0.058 | 0.576 | 66.000 |  |  |  | False | legacy-6e8f2c8704009950 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_fixed_average | legacy | 0.751 |  |  | 135 |  | 179 | 0.087 | -0.184 | 0.579 | 142.000 |  |  |  | False | legacy-3e0decdc4f01efec |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_fixed_average | legacy | 0.751 |  |  | 135 |  | 179 | 0.087 | -0.184 | 0.579 | 142.000 |  |  |  | False | legacy-dcec23d875bfa3c5 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_linear_single | legacy | 0.746 |  |  | 135 |  | 179 | 0.020 | -0.053 | 0.579 | 59.000 |  |  |  | False | legacy-f27e0f1e12cc16e5 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_linear_single | legacy | 0.746 |  |  | 135 |  | 179 | 0.020 | -0.053 | 0.579 | 59.000 |  |  |  | False | legacy-5415449afcd27bd8 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_ldp_sigmoid_average | legacy | 0.744 |  |  | 135 |  | 179 | 0.024 | -0.044 | 0.541 | 158.000 |  |  |  | False | legacy-919988aed7d8746a |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_ldp_sigmoid_average | legacy | 0.734 |  |  | 135 |  | 179 | 0.023 | -0.044 | 0.541 | 158.000 |  |  |  | False | legacy-0a92c3596ec8cc60 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_ecdf_average | legacy | 0.729 |  |  | 135 |  | 179 | 0.077 | -0.124 | 0.541 | 158.000 |  |  |  | False | legacy-eee8cb16150cedaf |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_ecdf_average | legacy | 0.725 |  |  | 135 |  | 179 | 0.076 | -0.124 | 0.541 | 158.000 |  |  |  | False | legacy-65bd2c7444646239 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ecdf_single | legacy | 0.701 |  |  | 135 |  | 179 | 0.044 | -0.124 | 0.579 | 60.000 |  |  |  | False | legacy-cce218203a16c721 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ecdf_single | legacy | 0.701 |  |  | 135 |  | 179 | 0.044 | -0.124 | 0.579 | 60.000 |  |  |  | False | legacy-06ee4beab9fb790a |
| U7 | SPY_1Day_wavelet_trend_logit_l2_linear_single | legacy | 0.685 |  |  | 135 |  | 179 | 0.017 | -0.026 | 0.576 | 64.000 |  |  |  | False | legacy-9e04befab2eaf8a8 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_linear_single | legacy | 0.685 |  |  | 135 |  | 179 | 0.017 | -0.026 | 0.576 | 64.000 |  |  |  | False | legacy-8f2e4c2d2d22776a |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ecdf_average | legacy | 0.672 |  |  | 135 |  | 179 | 0.047 | -0.110 | 0.576 | 158.000 |  |  |  | False | legacy-fc4f1b487dbce1e0 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ecdf_average | legacy | 0.671 |  |  | 135 |  | 179 | 0.047 | -0.110 | 0.576 | 158.000 |  |  |  | False | legacy-cf65c7c9a237c188 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ldp_sigmoid_single | legacy | 0.608 |  |  | 135 |  | 179 | 0.014 | -0.049 | 0.579 | 59.000 |  |  |  | False | legacy-455e2c0213cc50b4 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ldp_sigmoid_single | legacy | 0.608 |  |  | 135 |  | 179 | 0.014 | -0.049 | 0.579 | 59.000 |  |  |  | False | legacy-38f94a2367a05638 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_linear_average | legacy | 0.602 |  |  | 135 |  | 179 | 0.017 | -0.056 | 0.579 | 142.000 |  |  |  | False | legacy-1cbc014a7ba2a15e |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_linear_average | legacy | 0.602 |  |  | 135 |  | 179 | 0.023 | -0.053 | 0.541 | 158.000 |  |  |  | False | legacy-0eb07015986a0a15 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_linear_average | legacy | 0.599 |  |  | 135 |  | 179 | 0.017 | -0.056 | 0.579 | 142.000 |  |  |  | False | legacy-fdf1aec2f2f87892 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_fixed_single | legacy | 0.599 |  |  | 135 |  | 179 | 0.062 | -0.155 | 0.579 | 64.000 |  |  |  | False | legacy-f8824f5dba07633b |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_fixed_single | legacy | 0.599 |  |  | 135 |  | 179 | 0.062 | -0.155 | 0.579 | 64.000 |  |  |  | False | legacy-2e01350a70a12a34 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_linear_average | legacy | 0.598 |  |  | 135 |  | 179 | 0.022 | -0.053 | 0.541 | 158.000 |  |  |  | False | legacy-0dbd707581c8ee38 |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ecdf_average | legacy | 0.568 |  |  | 135 |  | 179 | 0.036 | -0.136 | 0.579 | 142.000 |  |  |  | False | legacy-5e248eb6445ccdef |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ecdf_average | legacy | 0.567 |  |  | 135 |  | 179 | 0.036 | -0.136 | 0.579 | 142.000 |  |  |  | False | legacy-43f0abac48a6332b |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ldp_sigmoid_average | legacy | 0.545 |  |  | 135 |  | 179 | 0.013 | -0.056 | 0.579 | 142.000 |  |  |  | False | legacy-9899d6b234db573a |
| U7 | SPY_1Day_wavelet_trend_rf_ldp_ldp_sigmoid_average | legacy | 0.545 |  |  | 135 |  | 179 | 0.013 | -0.056 | 0.579 | 142.000 |  |  |  | False | legacy-f30a6f58026320e0 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_kelly_capped_single | legacy | 0.534 |  |  | 135 |  | 179 | 0.006 | -0.020 | 0.541 | 44.000 |  |  |  | False | legacy-f7985cafb98704a6 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_kelly_capped_single | legacy | 0.534 |  |  | 135 |  | 179 | 0.006 | -0.020 | 0.541 | 44.000 |  |  |  | False | legacy-69b40577dd0748ad |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ldp_sigmoid_average | legacy | 0.507 |  |  | 135 |  | 179 | 0.011 | -0.043 | 0.576 | 159.000 |  |  |  | False | legacy-27190fd7c79b1aa0 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ldp_sigmoid_average | legacy | 0.507 |  |  | 135 |  | 179 | 0.011 | -0.043 | 0.576 | 159.000 |  |  |  | False | legacy-c1caac773d24bc5b |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ldp_sigmoid_single | legacy | 0.506 |  |  | 135 |  | 179 | 0.010 | -0.027 | 0.576 | 62.000 |  |  |  | False | legacy-8641eda1221945bc |
| U7 | SPY_1Day_wavelet_trend_logit_l2_ldp_sigmoid_single | legacy | 0.506 |  |  | 135 |  | 179 | 0.010 | -0.027 | 0.576 | 62.000 |  |  |  | False | legacy-1ac0fbf5d2a731e9 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_kelly_capped_single | legacy | 0.489 |  |  | 135 |  | 179 | 0.002 | -0.005 | 0.576 | 13.000 |  |  |  | False | legacy-c6dda85f71fc6621 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_kelly_capped_single | legacy | 0.489 |  |  | 135 |  | 179 | 0.002 | -0.005 | 0.576 | 13.000 |  |  |  | False | legacy-f317645508d4bc33 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_linear_average | legacy | 0.461 |  |  | 135 |  | 179 | 0.012 | -0.049 | 0.576 | 159.000 |  |  |  | False | legacy-356ce83a1b5b28e6 |
| U7 | SPY_1Day_wavelet_trend_logit_l2_linear_average | legacy | 0.461 |  |  | 135 |  | 179 | 0.012 | -0.049 | 0.576 | 159.000 |  |  |  | False | legacy-95e3b62bc0970c9c |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_kelly_capped_average | legacy | 0.453 |  |  | 135 |  | 179 | 0.005 | -0.015 | 0.541 | 145.000 |  |  |  | False | legacy-e85d2008b51c2060 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_kelly_capped_average | legacy | 0.453 |  |  | 135 |  | 179 | 0.005 | -0.015 | 0.541 | 145.000 |  |  |  | False | legacy-0fdcc58d76a8b87e |
| U7 | IWM_1Day_wavelet_trend_logit_l2_linear_single | legacy | 0.429 |  |  | 135 |  | 179 | 0.002 | -0.007 | 0.479 | 3.000 |  |  |  | False | legacy-d8e2464401925a42 |
| U7 | IWM_1Day_wavelet_trend_logit_l2_ldp_sigmoid_single | legacy | 0.429 |  |  | 135 |  | 179 | 0.002 | -0.007 | 0.479 | 3.000 |  |  |  | False | legacy-76091e28f8a694fd |
| U7 | IWM_1Day_wavelet_trend_logit_l2_linear_single | legacy | 0.429 |  |  | 135 |  | 179 | 0.002 | -0.007 | 0.479 | 3.000 |  |  |  | False | legacy-d6e8c871e94ca4d2 |
| U7 | IWM_1Day_wavelet_trend_logit_l2_ldp_sigmoid_single | legacy | 0.429 |  |  | 135 |  | 179 | 0.002 | -0.007 | 0.479 | 3.000 |  |  |  | False | legacy-6dccb6b67d08dd41 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_fixed_average | legacy | 0.393 |  |  | 135 |  | 179 | 0.050 | -0.181 | 0.541 | 158.000 |  |  |  | False | legacy-d14787373f0546c6 |
| U7 | QQQ_1Day_wavelet_trend_logit_l2_fixed_average | legacy | 0.388 |  |  | 135 |  | 179 | 0.049 | -0.181 | 0.541 | 158.000 |  |  |  | False | legacy-e93b00e038a93b6a |
| U8 | DIA-GLD-IWM-QQQ-SPY-TLT_1Hour_wavelet_trend_logit_l2_fixed_none | legacy | -0.119 |  |  | 139 |  | 179 | -0.028 | -0.296 | 0.502 | 2583.000 |  |  |  | False | legacy-f51c4883c807df6d |
| U8 | DIA-GLD-IWM-QQQ-SPY-TLT_1Hour_wavelet_trend_logit_l2_fixed_none | legacy | -0.119 |  |  | 139 |  | 179 | -0.028 | -0.296 | 0.502 | 2583.000 |  |  |  | False | legacy-ad9765ec0d3cc082 |
| U8 | DIA-GLD-IWM-QQQ-SPY-TLT_1Hour_wavelet_trend_logit_l2_fixed_standard | legacy | -1.008 |  |  | 139 |  | 179 | -0.021 | -0.162 | 0.502 | 2583.000 |  |  |  | False | legacy-87e26a46c51e5420 |
| U8 | DIA-GLD-IWM-QQQ-SPY-TLT_1Hour_wavelet_trend_logit_l2_fixed_standard | legacy | -1.008 |  |  | 139 |  | 179 | -0.021 | -0.162 | 0.502 | 2583.000 |  |  |  | False | legacy-4e8b96a44de46a8d |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS21 | legacy | 0.094 |  |  | 163 |  | 179 | 0.004 | -0.107 |  | 771.000 |  | 2.424 |  | False | legacy-19f4e000d6163840 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS21 | legacy | 0.094 |  |  | 163 |  | 179 | 0.004 | -0.107 |  | 771.000 |  |  |  | False | legacy-f2fb81ed3418f8f0 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS63 | legacy | 0.050 |  |  | 163 |  | 179 | 0.001 | -0.154 |  | 715.000 |  | 0.428 |  | False | legacy-a336a1366c012757 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS63 | legacy | 0.050 |  |  | 163 |  | 179 | 0.001 | -0.154 |  | 715.000 |  |  |  | False | legacy-e5d25025305278c8 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_nested | legacy | 0.050 |  |  | 163 |  | 179 | 0.001 | -0.125 |  |  |  |  | 0.768 | False | legacy-1a9c052c9573485c |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_nested | legacy | 0.050 |  |  | 163 |  | 179 | 0.001 | -0.125 |  |  |  |  | 0.768 | False | legacy-41e4a456bd54e65a |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS5 | legacy | 0.005 |  |  | 163 |  | 179 | -0.002 | -0.157 |  | 930.000 |  |  |  | False | legacy-03783c614369be43 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS5 | legacy | 0.005 |  |  | 163 |  | 179 | -0.002 | -0.157 |  | 930.000 |  |  |  | False | legacy-7ada349fd5597387 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS10 | legacy | 0.004 |  |  | 163 |  | 179 | -0.002 | -0.134 |  | 922.000 |  |  |  | False | legacy-65fd3688e33e5810 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS10 | legacy | 0.004 |  |  | 163 |  | 179 | -0.002 | -0.134 |  | 922.000 |  |  |  | False | legacy-6cbfcde79e909c0c |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS63 | legacy | -0.058 |  |  | 163 |  | 179 | -0.006 | -0.124 |  | 933.000 |  |  |  | False | legacy-6ceed6fde27650ed |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS63 | legacy | -0.058 |  |  | 163 |  | 179 | -0.006 | -0.124 |  | 933.000 |  |  |  | False | legacy-b302132e808bb56e |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS10 | legacy | -0.108 |  |  | 163 |  | 179 | -0.009 | -0.091 |  | 796.000 |  | -2.510 |  | False | legacy-b63af51d0dc885e3 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS10 | legacy | -0.108 |  |  | 163 |  | 179 | -0.009 | -0.091 |  | 796.000 |  |  |  | False | legacy-ba679a016d464675 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS21 | legacy | -0.217 |  |  | 163 |  | 179 | -0.017 | -0.166 |  | 918.000 |  |  |  | False | legacy-7a5d948cd2bd9b98 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS252_OOS21 | legacy | -0.217 |  |  | 163 |  | 179 | -0.017 | -0.166 |  | 918.000 |  |  |  | False | legacy-89208977b37e7736 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS5 | legacy | -0.271 |  |  | 163 |  | 179 | -0.019 | -0.147 |  | 784.000 |  | -6.476 |  | False | legacy-911dc98e301f9328 |
| U9 | SPY_1Hour_wavelet_trend_logit_l2_fixed_none_IS504_OOS5 | legacy | -0.271 |  |  | 163 |  | 179 | -0.019 | -0.147 |  | 784.000 |  |  |  | False | legacy-2deeb047d02641b1 |
| U10 | QQQ_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | 0.024 | 0.527 | 0.000 | 179 | 0.000 | 179 | -0.002 | -0.183 | 0.506 | 1070.000 | 2070.000 |  |  | False | d6f03debb7cd5761 |
| U10 | QQQ_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | 0.024 | 0.527 | 0.000 | 179 | 0.000 | 179 | -0.002 | -0.183 | 0.506 | 1070.000 | 2070.000 |  |  | False | ae96e8a6a0192be9 |
| U10 | IWM-QQQ-SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_none | portfolio | -0.059 | 0.433 | 0.000 | 179 | 0.000 | 179 | -0.015 | -0.190 | 0.505 | 1577.000 | 2072.000 |  |  | False | a2e06842c562f254 |
| U10 | IWM-QQQ-SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_none | portfolio | -0.059 | 0.433 | 0.000 | 179 | 0.000 | 179 | -0.015 | -0.190 | 0.505 | 1577.000 | 2072.000 |  |  | False | 218bb500c5a757a1 |
| U10 | IWM_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | -0.120 | 0.366 | 0.000 | 179 | 0.000 | 179 | -0.005 | -0.120 | 0.503 | 199.000 | 2072.000 |  |  | False | 461fe6f3fc30819e |
| U10 | IWM_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | -0.120 | 0.366 | 0.000 | 179 | 0.000 | 179 | -0.005 | -0.120 | 0.503 | 199.000 | 2072.000 |  |  | False | 2610ebe14a1f2140 |
| U10 | SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | -0.124 | 0.361 | 0.000 | 179 | 0.000 | 179 | -0.006 | -0.120 | 0.505 | 308.000 | 2072.000 |  |  | False | fb4aad0069fd2e95 |
| U10 | SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_none | wfo | -0.124 | 0.361 | 0.000 | 179 | 0.000 | 179 | -0.006 | -0.120 | 0.505 | 308.000 | 2072.000 |  |  | False | 7e9fcb36817f5b65 |
| U10 | IWM_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.427 | 0.105 | 0.000 | 179 | 0.000 | 179 | -0.003 | -0.036 | 0.503 | 199.000 | 2072.000 |  |  | False | cddf130651da68b3 |
| U10 | IWM_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.427 | 0.105 | 0.000 | 179 | 0.000 | 179 | -0.003 | -0.036 | 0.503 | 199.000 | 2072.000 |  |  | False | 4b927ca7b4bbefbd |
| U10 | QQQ_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.643 | 0.032 | 0.000 | 179 | 0.000 | 179 | -0.011 | -0.103 | 0.506 | 1070.000 | 2070.000 |  |  | False | 82bd1a7429b878be |
| U10 | QQQ_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.643 | 0.032 | 0.000 | 179 | 0.000 | 179 | -0.011 | -0.103 | 0.506 | 1070.000 | 2070.000 |  |  | False | 59b41e3157a26535 |
| U10 | SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.568 | 0.046 | 0.000 | 179 | 0.000 | 179 | -0.004 | -0.042 | 0.505 | 308.000 | 2072.000 |  |  | False | b6a5653172633441 |
| U10 | SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | wfo | -0.568 | 0.046 | 0.000 | 179 | 0.000 | 179 | -0.004 | -0.042 | 0.505 | 308.000 | 2072.000 |  |  | False | 90356c5bbfa7804d |
| U10 | IWM-QQQ-SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | portfolio | -0.790 | 0.010 | 0.000 | 179 | 0.000 | 179 | -0.016 | -0.127 | 0.505 | 1577.000 | 2072.000 |  |  | False | 47239b605a6fecc0 |
| U10 | IWM-QQQ-SPY_1Hour_wavelet_trend_logit_l2_oof_fixed_standard | portfolio | -0.790 | 0.010 | 0.000 | 179 | 0.000 | 179 | -0.016 | -0.127 | 0.505 | 1577.000 | 2072.000 |  |  | False | 927314e28f92bcb0 |

