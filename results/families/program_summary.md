# Hypothesis-family program: summary

3 families, 3 tested (program PLAN3: G1, G2, G3, G5, G6). A family passes iff its Holm-adjusted headline p < 0.05, its headline clears every floor, it is coherent and its statistic is positive (SPEC §17.3, §22). The DSR column is a ledger diagnostic.

Trial accounting: 21 / 42 program trials, 3 / 5 families.

| family | kind | sharpe | bench | statistic | CI | p | p_holm | floors | coherence | DSR | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | overlay_alpha | 0.80 | 0.83 | 0.053 | 0.02 … — | 0.0034 | 0.0102 | yes | 0.97 (yes) | 0.457 | PASS |
| G2 | marginal | 0.80 | 0.83 | 0.264 | 0.02 … — | 0.0398 | 0.0796 | no | 1.00 (no) | 0.457 | fail |
| G3 | paired | 0.83 | 0.83 | 0.032 | -0.08 … — | 0.3099 | 0.3099 | yes | 0.67 (no) | 0.512 | fail |

Coherence: share of variants with the headline's sign (coherent or not: the median variant must also clear the floors). DSR: N = 65 program trials now (SPEC §17.3).

Per-family reports: [G1](G1/report.md), [G2](G2/report.md), [G3](G3/report.md)
