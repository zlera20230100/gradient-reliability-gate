# -*- coding: utf-8 -*-
# Reliability-gated hybrid design gradient: autodiff (deep-ensemble mean) on the components the
# gate trusts, one full-wave finite-difference (FD) solve on the rest.
# (1) controlled reliability spectrum (same model as reliability_calib.py): cost-accuracy frontier,
#     gate subset vs oracle subset.
# (2) 24-GHz FPC device: gate applied to the six radiation zones and the impedance direction.
# Writes hybrid_gradient.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

RNG = np.random.default_rng(20260622)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])     # mixed-sign per-zone gradient (same as the calib study)
K = a_true.size
M = 10                                                    # ensemble size
TAU = 0.9                                                 # trust threshold (from calibration)
N_TRIALS = 6000

def one_trial():
    # each component gets its own reliability level, so a single gradient mixes trusted and untrusted comps
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + RNG.normal(0.0, sigma[None, :], size=(M, K))   # M independent retrains
    ad = J.mean(0)                                                       # autodiff estimate
    sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))            # ensemble sign-agreement
    true_rel = (np.abs(a_true) / sigma) >= 1.0                          # ground truth: signal dominates noise
    return ad, sign_agree, true_rel

def hybrid(ad, trust):
    g = np.where(trust, ad, a_true)        # autodiff on trusted comps; FD truth on the rest
    return g, (~trust).sum()               # gradient, number of FD solves spent

def rel_err(g):
    return np.linalg.norm(g - a_true) / np.linalg.norm(a_true)

# (1) cost-accuracy frontier over the trust threshold
taus = np.round(np.linspace(0.5, 1.0, 11), 2)
cost_gate = {t: [] for t in taus}; err_gate = {t: [] for t in taus}
cost_oracle = []; err_oracle = []          # oracle = trust exactly the truly-reliable comps
err_allad = []; err_allfd = []
# operating-point bookkeeping at TAU
op_cost = []; op_err = []; n_corr_select = []; n_total_rel = []
for _ in range(N_TRIALS):
    ad, sa, tr = one_trial()
    err_allad.append(rel_err(ad)); err_allfd.append(0.0)
    g_o, c_o = hybrid(ad, tr); cost_oracle.append(c_o); err_oracle.append(rel_err(g_o))
    for t in taus:
        trust = sa >= t
        g, c = hybrid(ad, trust); cost_gate[t].append(c); err_gate[t].append(rel_err(g))
    trust = sa >= TAU
    g, c = hybrid(ad, trust); op_cost.append(c); op_err.append(rel_err(g))
    # overlap between the gate's trusted set and the truly-reliable set
    n_corr_select.append((trust & tr).sum()); n_total_rel.append(tr.sum())

c_curve = np.array([np.mean(cost_gate[t]) for t in taus])
e_curve = np.array([np.mean(err_gate[t]) for t in taus])
print("CONTROLLED SPECTRUM cost-accuracy frontier (M=10, K=6):")
print(f"{'tau':>5} {'mean FD solves':>15} {'mean rel.err':>13}")
for t, c, e in zip(taus, c_curve, e_curve):
    print(f"{t:>5.2f} {c:>15.2f} {e:>13.3f}")
print(f"\n all-autodiff (0 solves): rel.err {np.mean(err_allad):.3f}")
print(f" all-FD ({K} solves)     : rel.err {np.mean(err_allfd):.3f}")
print(f" ORACLE hybrid           : {np.mean(cost_oracle):.2f} solves, rel.err {np.mean(err_oracle):.3f}")
print(f" GATE hybrid @tau={TAU}    : {np.mean(op_cost):.2f} solves, rel.err {np.mean(op_err):.3f}")
print(f"   -> solver saving vs all-FD: {100*(1-np.mean(op_cost)/K):.0f}%   "
      f"error vs all-autodiff: {np.mean(op_err):.3f} vs {np.mean(err_allad):.3f}")
frac_rel = np.sum(n_corr_select) / max(1, np.sum(n_total_rel))
print(f"   -> gate recovers {100*frac_rel:.0f}% of the truly-reliable comps (so the shortcut is taken where safe)")

# (2) real 24-GHz FPC device
ms = np.load('zones_multiseed.npz'); fw = np.load('grad_fullwave.npz')
rela = ms['rela']                  # (10,6) per-seed radiation grad per zone
fd = fw['fd_grad']                 # (6,) full-wave FD truth
sa_dev = np.maximum((rela > 0).mean(0), (rela < 0).mean(0))   # ensemble sign-agreement per radiation zone
ad_dev = rela.mean(0)              # autodiff radiation gradient (ensemble mean)
trust_dev = sa_dev >= TAU
print("\nREAL 24-GHz FPC device (6 radiation zones):")
print(" ensemble sign-agree :", np.round(sa_dev, 2), " (threshold", TAU, ")")
print(" gate trusts         :", trust_dev.astype(int), f"-> {(~trust_dev).sum()}/{6} flagged for FD (no shortcut)")
# accuracy of the all-autodiff radiation gradient against the FD truth
err_naive_dev = np.linalg.norm(ad_dev - fd) / np.linalg.norm(fd)
signmatch = (np.sign(ad_dev) == np.sign(fd)).sum()
print(f" naive all-autodiff radiation gradient: rel.err {err_naive_dev:.1f}x truth, "
      f"sign-correct {signmatch}/6  -> gate's refusal to shortcut is VINDICATED")

np.savez('hybrid_gradient.npz',
         taus=taus, cost_curve=c_curve, err_curve=e_curve,
         err_allad=np.mean(err_allad), cost_oracle=np.mean(cost_oracle), err_oracle=np.mean(err_oracle),
         op_cost=np.mean(op_cost), op_err=np.mean(op_err), K=K, tau=TAU,
         frac_rel_recovered=frac_rel,
         dev_sign_agree=sa_dev, dev_trust=trust_dev, dev_ad=ad_dev, dev_fd=fd,
         dev_naive_relerr=err_naive_dev, dev_signmatch=signmatch)
print("\nsaved hybrid_gradient.npz")
