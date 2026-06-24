# -*- coding: utf-8 -*-
# Clustered bootstrap 95% CI for the controlled-spectrum reliability AUC at M=10.
# Resamples whole trials (each trial = one sigma, 6 correlated zones) with replacement, so the CI
# respects the within-trial correlation. Mirrors reliability_calib.py.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260622)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])
K = a_true.size
N_TRIALS = 4000
M = 10

def ensemble(M, sigma):
    J = a_true[None, :] + RNG.normal(0.0, sigma, size=(M, K))
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))
    snr = np.abs(J.mean(0)) / (J.std(0) + 1e-12)
    return sign_agree, snr

# build per-trial blocks
sa_blocks, snr_blocks, lab_blocks = [], [], []
for _ in range(N_TRIALS):
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0)))
    sa, sn = ensemble(M, sigma)
    lab = (np.abs(a_true) / sigma >= 1.0).astype(int)
    sa_blocks.append(sa); snr_blocks.append(sn); lab_blocks.append(lab)
sa_blocks = np.array(sa_blocks); snr_blocks = np.array(snr_blocks); lab_blocks = np.array(lab_blocks)  # (T,K)

y = lab_blocks.ravel()
print(f"M={M}: {N_TRIALS} trials x {K} zones = {y.size} components; reliable frac {y.mean():.2f}")
for name, X in [('sign-agreement', sa_blocks), ('SNR', snr_blocks)]:
    pt = roc_auc_score(y, X.ravel())
    boot = []
    BRNG = np.random.default_rng(7)
    for _ in range(2000):
        sel = BRNG.integers(0, N_TRIALS, N_TRIALS)         # resample trials (clusters)
        yy = lab_blocks[sel].ravel(); xx = X[sel].ravel()
        if len(np.unique(yy)) < 2:
            continue
        boot.append(roc_auc_score(yy, xx))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:>16}: AUC={pt:.3f}  clustered 95% CI [{lo:.3f}, {hi:.3f}]")
