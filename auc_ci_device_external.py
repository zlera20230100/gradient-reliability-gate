# -*- coding: utf-8 -*-
# Bootstrap 95% CIs for the reliability AUCs from saved arrays:
#  (1) the n=18 labelled device set (reliability.npz), and
#  (2) the external thin-film TMM benchmark (extbench_tmm.npz: all_sa score vs all_correct label),
# plus the external-benchmark class balance. No retraining.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(7)
NB = 5000

def boot_ci(y, x, nb=NB):
    y = np.asarray(y).astype(int); x = np.asarray(x, dtype=float)
    pt = roc_auc_score(y, x)
    bs = []
    n = len(y)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi, n

print("=== (1) n=18 device-labelled set (reliability.npz) ===")
d = np.load('reliability.npz')
y = d['label']
print(f"  class balance: {int(y.sum())} reliable / {int((1-y).sum())} unreliable  (n={len(y)})")
for name, key in [('sign-agreement','sign_agree'), ('SNR','snr'), ('naive |gradient|','magnitude')]:
    pt, lo, hi, n = boot_ci(y, d[key])
    print(f"  {name:>16}: AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

print("\n=== (2) external thin-film TMM benchmark (extbench_tmm.npz) ===")
e = np.load('extbench_tmm.npz', allow_pickle=True)
sa = np.asarray(e['all_sa']).ravel().astype(float)
correct = np.asarray(e['all_correct']).ravel().astype(int)
npos, nneg = int(correct.sum()), int((1-correct).sum())
print(f"  pooled components: {len(correct)}  | class balance: {npos} sign-correct / {nneg} sign-wrong")
print(f"  effective negatives (sign-wrong) = {nneg}  <-- AUC fragility check")
if nneg >= 1 and npos >= 1:
    pt, lo, hi, n = boot_ci(correct, sa)
    print(f"  sign-agreement gate: AUC={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={n})")
else:
    print("  cannot bootstrap: a class is empty")
print(f"  reported point AUC in npz: {float(e['auc']):.3f}")
