# Paper 2 — AUC confidence intervals (deep-revision, computed from existing generators/data)

All bootstrap 95% CIs. Synthetic-spectrum AUCs use a CLUSTERED bootstrap (resample whole trials,
each trial = one sigma across 6 correlated zones) to respect within-trial correlation.

| AUC | point | 95% CI | n / notes |
|---|---|---|---|
| decoupled (outcome label) sign-agreement | 0.897 | [0.893, 0.901] | 36000 comp; circularity-free → PRIMARY |
| decoupled (outcome label) variance/SNR gate | 0.932 | [0.928, 0.935] | 36000 comp |
| controlled-spectrum calib M=10 sign-agree | 0.966 | [0.964, 0.969] | 24000 comp; IN-MODEL (label≈predictor) |
| controlled-spectrum calib M=10 SNR | 0.986 | [0.984, 0.987] | 24000 comp; in-model |
| external thin-film TMM (sign-agreement gate) | 0.906 | [0.833, 0.965] | 880 pooled comp; 852 correct / 28 wrong |
| device n=18 sign-agreement | 1.000 | [1.000, 1.000] | degenerate (6 reliable / 12 unreliable) |
| device n=18 SNR | 1.000 | [1.000, 1.000] | degenerate |
| device n=18 naive |gradient| | 0.875 | [0.650, 1.000] | |

Scripts: _ci_all.py (n=18 + external), _ci_synth.py (calib M=10), _ci_decouple.py (decoupled).
Reviewer note refuted: external benchmark has 28 sign-wrong negatives (not <1), so its AUC CI is well-defined.
