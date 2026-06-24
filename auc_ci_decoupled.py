# -*- coding: utf-8 -*-
# Clustered bootstrap 95% CI for the LABEL-DECOUPLED reliability AUC (outcome label = ensemble-mean sign
# correct vs full-wave truth). Mirrors hybrid_decouple.py; resamples whole trials. This is the
# circularity-free statistic the reviewers asked to be promoted to primary.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260623)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9]); K = a_true.size; M = 10; N_TRIALS = 6000

SA, SNR, Lout = [], [], []
for _ in range(N_TRIALS):
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + RNG.normal(0.0, sigma[None, :], size=(M, K))
    ad = J.mean(0)
    sa = np.maximum((J > 0).mean(0), (J < 0).mean(0))
    snr = np.abs(ad) / (J.std(0) + 1e-12)
    lo = (np.sign(ad) == np.sign(a_true)).astype(int)
    SA.append(sa); SNR.append(snr); Lout.append(lo)
SA = np.array(SA); SNR = np.array(SNR); Lout = np.array(Lout)   # (T,K)

y = Lout.ravel()
print(f"decoupled (outcome) label, M={M}: {y.size} components; sign-correct frac {y.mean():.2f}")
B = np.random.default_rng(7)
for name, X in [('sign-agreement', SA), ('variance/SNR gate', SNR)]:
    pt = roc_auc_score(y, X.ravel())
    boot = []
    for _ in range(2000):
        sel = B.integers(0, N_TRIALS, N_TRIALS)
        yy = Lout[sel].ravel(); xx = X[sel].ravel()
        if len(np.unique(yy)) < 2:
            continue
        boot.append(roc_auc_score(yy, xx))
    lo_, hi_ = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:>18} -> outcome label: AUC={pt:.3f}  clustered 95% CI [{lo_:.3f}, {hi_:.3f}]")
