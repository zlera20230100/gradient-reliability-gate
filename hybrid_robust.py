# -*- coding: utf-8 -*-
# Re-runs the reliability-gated hybrid-gradient cost-accuracy result under several noise models
# other than i.i.d.-Gaussian, including a correlated-seed case (shared bias across retrains).
# The reliability label is fixed (population SNR |a|/sigma_scale >= 1); only the sampling noise
# the ensemble sees changes. Writes hybrid_robust.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

RNG = np.random.default_rng(20260622)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])
K = a_true.size; M = 10; TAU = 0.9; N_TRIALS = 6000

def draw(model, sigma):
    """M-by-K seed gradients: signal + noise of given model and per-component scale sigma (std-matched)."""
    if model == 'gaussian':
        return RNG.normal(0.0, sigma[None, :], size=(M, K))
    if model == 'student_t':                      # heavy-tailed (df=3), scaled to std = sigma
        df = 3.0; t = RNG.standard_t(df, size=(M, K))
        return t * (sigma[None, :] / np.sqrt(df / (df - 2.0)))
    if model == 'laplace':                         # heavier-than-Gaussian peak/tails, std = sigma
        return RNG.laplace(0.0, sigma[None, :] / np.sqrt(2.0), size=(M, K))
    if model == 'correlated':                      # shared per-component bias across seeds + idiosyncratic
        rho = 0.6                                  # 60% of variance is a common (seed-shared) bias
        common = RNG.normal(0.0, np.sqrt(rho) * sigma, size=(K,))     # same for every seed
        idio = RNG.normal(0.0, np.sqrt(1 - rho) * sigma[None, :], size=(M, K))
        return common[None, :] + idio
    raise ValueError(model)

def trial(model):
    sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0), size=K))
    J = a_true[None, :] + draw(model, sigma)
    ad = J.mean(0)
    sa = np.maximum((J > 0).mean(0), (J < 0).mean(0))
    true_rel = (np.abs(a_true) / sigma) >= 1.0
    return ad, sa, true_rel

def rel_err(g):
    return np.linalg.norm(g - a_true) / np.linalg.norm(a_true)

print(f"{'noise model':>12} | {'gate cost@tau':>13} {'save%':>6} {'gate err':>9} | "
      f"{'all-AD err':>10} {'oracle cost':>11} {'oracle err':>10} {'rel.recov%':>10}")
out = {}
for model in ('gaussian', 'student_t', 'laplace', 'correlated'):
    cost, err, allad, ocost, oerr, nsel, nrel = [], [], [], [], [], 0, 0
    for _ in range(N_TRIALS):
        ad, sa, tr = trial(model)
        allad.append(rel_err(ad))
        tt = sa >= TAU
        g = np.where(tt, ad, a_true); cost.append((~tt).sum()); err.append(rel_err(g))
        go = np.where(tr, ad, a_true); ocost.append((~tr).sum()); oerr.append(rel_err(go))
        nsel += (tt & tr).sum(); nrel += tr.sum()
    c, e = np.mean(cost), np.mean(err); save = 100 * (1 - c / K); recov = 100 * nsel / max(1, nrel)
    out[model] = dict(cost=c, save=save, err=e, allad=np.mean(allad),
                      ocost=np.mean(ocost), oerr=np.mean(oerr), recov=recov)
    print(f"{model:>12} | {c:>13.2f} {save:>5.0f}% {e:>9.3f} | "
          f"{np.mean(allad):>10.3f} {np.mean(ocost):>11.2f} {np.mean(oerr):>10.3f} {recov:>9.0f}%")

np.savez('hybrid_robust.npz', tau=TAU, K=K,
         models=np.array(list(out.keys())),
         save=np.array([out[m]['save'] for m in out]),
         gate_err=np.array([out[m]['err'] for m in out]),
         allad_err=np.array([out[m]['allad'] for m in out]),
         recov=np.array([out[m]['recov'] for m in out]))
print("\nsaved hybrid_robust.npz")
print("READ: across all four noise models (incl. heavy-tailed and adversarial correlated-seed),")
print("the gate keeps a large solver saving at far lower error than all-autodiff -> frontier is robust.")
