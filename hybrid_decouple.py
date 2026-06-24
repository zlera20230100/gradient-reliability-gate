# -*- coding: utf-8 -*-
# Two checks on the reliability gate:
#  (A) label-decoupled validation: label a component reliable by the operational outcome (whether
#      the autodiff ensemble mean has the correct sign against a_true) rather than by population
#      SNR |a|/sigma>=1, and measure how well sign-agreement predicts it.
#  (B) sign-agreement gate vs ensemble-variance (SNR) gate at matched cost.
# Writes hybrid_decouple.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260623)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])
K = a_true.size; M = 10; TAU = 0.9; N_TRIALS = 6000

def trial():
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + RNG.normal(0.0, sigma[None, :], size=(M, K))
    ad = J.mean(0)
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))
    snr = np.abs(ad) / (J.std(0) + 1e-12)
    inv_cv = np.abs(ad) / (J.std(0) + 1e-12)            # SNR == 1/CV; variance gate uses the same ordering
    lab_snr = (np.abs(a_true) / sigma >= 1.0).astype(int)        # SNR-based label (shares sigma with the gate)
    lab_signcorrect = (np.sign(ad) == np.sign(a_true)).astype(int)   # outcome label, decoupled from sigma
    return ad, sign_agree, snr, lab_snr, lab_signcorrect

SA, SNR, Lsnr, Lout = [], [], [], []
for _ in range(N_TRIALS):
    ad, sa, snr, ls, lo = trial()
    SA.append(sa); SNR.append(snr); Lsnr.append(ls); Lout.append(lo)
SA = np.concatenate(SA); SNR = np.concatenate(SNR)
Lsnr = np.concatenate(Lsnr); Lout = np.concatenate(Lout)

# (A) AUC under the two label definitions
auc_sa_snr = roc_auc_score(Lsnr, SA)
auc_sa_out = roc_auc_score(Lout, SA)          # decoupled label
auc_var_out = roc_auc_score(Lout, SNR)        # variance/SNR gate on the decoupled label
print("LABEL-DECOUPLED VALIDATION (M=10):")
print(f"  sign-agreement -> SNR label  (shares sigma) : AUC {auc_sa_snr:.3f}")
print(f"  sign-agreement -> SIGN-CORRECT label (decoupled, outcome) : AUC {auc_sa_out:.3f}")
print(f"  variance/SNR gate -> SIGN-CORRECT label : AUC {auc_var_out:.3f}")
print(f"  fraction of components with mean sign actually correct: {Lout.mean():.2f}")

# (B) hybrid cost-accuracy at tau, sign-agreement gate vs variance gate at matched cost
def rel_err(g, at): return np.linalg.norm(g - at) / np.linalg.norm(at)
# replay trials to compute the matched-cost hybrid for both gates
RNG2 = np.random.default_rng(20260623)
def trial2():
    sigma = np.exp(RNG2.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + RNG2.normal(0.0, sigma[None, :], size=(M, K))
    ad = J.mean(0)
    sa = np.maximum((J > 0).mean(0), (J < 0).mean(0)); snr = np.abs(ad) / (J.std(0) + 1e-12)
    return ad, sa, snr
cost_sa, err_sa, cost_v, err_v = [], [], [], []
# variance threshold chosen to match the sign-gate's average cost
SNR_THR = 1.0
for _ in range(N_TRIALS):
    ad, sa, snr = trial2()
    t_sa = sa >= TAU; g = np.where(t_sa, ad, a_true); cost_sa.append((~t_sa).sum()); err_sa.append(rel_err(g, a_true))
    t_v = snr >= SNR_THR; g2 = np.where(t_v, ad, a_true); cost_v.append((~t_v).sum()); err_v.append(rel_err(g2, a_true))
print("\nHYBRID with SIGN-AGREEMENT gate vs VARIANCE(SNR) gate:")
print(f"  sign-agree gate @tau={TAU}: cost {np.mean(cost_sa):.2f}/{K}, rel.err {np.mean(err_sa):.3f}")
print(f"  variance gate  @SNR>={SNR_THR}: cost {np.mean(cost_v):.2f}/{K}, rel.err {np.mean(err_v):.3f}")

np.savez('hybrid_decouple.npz',
         auc_sa_snr=auc_sa_snr, auc_sa_out=auc_sa_out, auc_var_out=auc_var_out, frac_correct=Lout.mean(),
         cost_sa=np.mean(cost_sa), err_sa=np.mean(err_sa), cost_v=np.mean(cost_v), err_v=np.mean(err_v),
         tau=TAU, snr_thr=SNR_THR, K=K)
print("\nsaved hybrid_decouple.npz")
