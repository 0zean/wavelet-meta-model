# Hypothesis-family program: summary

7 families, 7 tested. A family passes iff its Holm-adjusted headline p < 0.05, its headline clears every floor, it is coherent and its difference is positive (SPEC §17.3).

Trial accounting: 40 / 112 program trials, 7 / 8 families.

| family | sharpe | bench | Δ | CI | p | p_holm | floors | coherence | DSR | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.92 | 0.83 | 0.09 | -0.22 … 0.40 | 0.5462 | 1.0000 | no | 0.80 (yes) | 0.218 | fail |
| F2 | 0.23 | 0.86 | -0.63 | -1.65 … 0.38 | 0.2079 | 0.8316 | no | 1.00 (no) | 0.003 | fail |
| F3 | 0.72 | 0.90 | -0.18 | -0.72 … 0.36 | 0.5027 | 1.0000 | no | 1.00 (no) | 0.090 | fail |
| F4 | 0.57 | 0.67 | -0.10 | -0.91 … 0.71 | 0.8121 | 1.0000 | yes | 0.67 (no) | 0.032 | fail |
| F5 | -0.12 | 0.83 | -0.95 | -2.00 … 0.10 | 0.0710 | 0.4258 | no | 1.00 (no) | 0.000 | fail |
| F7 | -0.69 | 0.87 | -1.55 | -2.55 … -0.56 | 0.0055 | 0.0385 | no | 1.00 (no) | 0.000 | fail |
| F8 | -0.29 | 0.71 | -0.99 | -2.06 … 0.07 | 0.0715 | 0.4258 | no | 1.00 (no) | 0.000 | fail |

Coherence: share of variants with the headline's sign (coherent or not: the median variant must also clear the floors). DSR: N = 40 program trials now (SPEC §17.3).

Per-family reports: [F1](F1/report.md), [F2](F2/report.md), [F3](F3/report.md), [F4](F4/report.md), [F5](F5/report.md), [F7](F7/report.md), [F8](F8/report.md)
