# -*- coding: utf-8 -*-
# Monte-Carlo over a controlled reliability spectrum: a known per-zone gradient a_true observed
# through M independently-retrained seeds with epistemic noise sigma. Ground-truth per-zone
# reliability is true SNR |a_k|/sigma >= 1. Measures how well the ensemble indicator from M seeds
# predicts that, and how AUC varies with M. Writes reliability_calib.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260622)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])      # same mixed-sign pattern as the controlled study
K = a_true.size
N_TRIALS = 4000
M_LIST = [3, 5, 10, 20, 40]

def ensemble(M, sigma):
    # M seeds of the per-zone gradient: true signal + epistemic noise
    J = a_true[None, :] + RNG.normal(0.0, sigma, size=(M, K))     # (M,K)
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))     # estimated from M seeds
    snr = np.abs(J.mean(0)) / (J.std(0) + 1e-12)
    return sign_agree, snr

print("ensemble-size study: AUC of the cheap indicator for predicting true per-zone reliability")
print(f"{'M seeds':>8} {'AUC(sign-agree)':>16} {'AUC(SNR)':>10}")
auc_sign_by_M = {}; auc_snr_by_M = {}
for M in M_LIST:
    ind_sign, ind_snr, labels = [], [], []
    for _ in range(N_TRIALS):
        sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0)))    # log-uniform reliability level
        sa, sn = ensemble(M, sigma)
        true_snr = np.abs(a_true) / sigma                         # population per-zone SNR
        lab = (true_snr >= 1.0).astype(int)                       # reliable if signal >= noise
        ind_sign.append(sa); ind_snr.append(sn); labels.append(lab)
    ind_sign = np.concatenate(ind_sign); ind_snr = np.concatenate(ind_snr); labels = np.concatenate(labels)
    a_sign = roc_auc_score(labels, ind_sign); a_snr = roc_auc_score(labels, ind_snr)
    auc_sign_by_M[M] = a_sign; auc_snr_by_M[M] = a_snr
    print(f"{M:>8} {a_sign:>16.3f} {a_snr:>10.3f}   (reliable frac {labels.mean():.2f})")

# calibration curve at the paper's M=10: estimated sign-agreement -> empirical fraction truly reliable
M = 10
ind_sign, labels = [], []
for _ in range(N_TRIALS):
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0)))
    sa, _ = ensemble(M, sigma)
    ind_sign.append(sa); labels.append((np.abs(a_true) / sigma >= 1.0).astype(int))
ind_sign = np.concatenate(ind_sign); labels = np.concatenate(labels)
bins = np.linspace(0.5, 1.0, 6)                                   # sign-agreement in [0.5,1]
centers, fracs = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (ind_sign >= lo) & (ind_sign < hi if hi < 1.0 else ind_sign <= hi)
    if m.sum() > 20:
        centers.append((lo + hi) / 2); fracs.append(labels[m].mean())
centers = np.array(centers); fracs = np.array(fracs)
print("\ncalibration at M=10  (sign-agreement bin -> empirical reliable fraction):")
for c, f in zip(centers, fracs):
    print(f"  sign-agree~{c:.2f} -> {f:.2f} reliable")

np.savez('reliability_calib.npz',
         M_list=np.array(M_LIST),
         auc_sign=np.array([auc_sign_by_M[m] for m in M_LIST]),
         auc_snr=np.array([auc_snr_by_M[m] for m in M_LIST]),
         calib_centers=centers, calib_fracs=fracs, n_trials=N_TRIALS, a_true=a_true)
print("\nsaved reliability_calib.npz")
