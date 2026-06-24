# -*- coding: utf-8 -*-
# Shared-bias sweep for the reliability-gated hybrid gradient: the noise across the M retrains has a
# fraction rho of common (seed-shared) bias and 1-rho idiosyncratic. rho=0 is the i.i.d. case the gate
# is designed for; rho->1 is the adversarial correlated-seed regime that defeats it. Shows WHERE the
# frontier breaks (gate error climbs toward the all-autodiff error). Writes shared_bias_sweep.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np

RNG = np.random.default_rng(20260624)
a_true = np.array([2.0, -1.5, 0.8, -0.6, 1.3, -0.9])
K = a_true.size; M = 10; TAU = 0.9; N_TRIALS = 6000
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def rel_err(g):
    return np.linalg.norm(g - a_true) / np.linalg.norm(a_true)

print(f"{'rho(shared)':>11} | {'gate cost/K':>11} {'save%':>6} {'gate err':>9} | "
      f"{'all-AD err':>10} {'recovery%':>10}  flag")
rows = []
for rho in RHOS:
    cost, err, allad, nsel, nrel = [], [], [], 0, 0
    for _ in range(N_TRIALS):
        sigma = np.exp(RNG.uniform(np.log(0.05), np.log(8.0), size=K))
        common = RNG.normal(0.0, np.sqrt(rho) * sigma, size=(K,))
        idio = RNG.normal(0.0, np.sqrt(1 - rho) * sigma[None, :], size=(M, K))
        J = a_true[None, :] + common[None, :] + idio
        ad = J.mean(0)
        sa = np.maximum((J > 0).mean(0), (J < 0).mean(0))
        tr = (np.abs(a_true) / sigma) >= 1.0
        allad.append(rel_err(ad))
        tt = sa >= TAU
        g = np.where(tt, ad, a_true); cost.append((~tt).sum()); err.append(rel_err(g))
        nsel += (tt & tr).sum(); nrel += tr.sum()
    c, e, aa = np.mean(cost), np.mean(err), np.mean(allad)
    recov = 100 * nsel / max(1, nrel)
    # "broken" when the gate no longer beats all-autodiff by a clear margin
    flag = 'BREAKS' if e > 0.5 * aa else 'ok'
    rows.append((rho, c, 100 * (1 - c / K), e, aa, recov))
    print(f"{rho:>11.2f} | {c:>11.2f} {100*(1-c/K):>5.0f}% {e:>9.3f} | {aa:>10.3f} {recov:>9.0f}%  {flag}")

rows = np.array(rows)
np.savez('shared_bias_sweep.npz', rho=rows[:, 0], gate_cost=rows[:, 1], save_pct=rows[:, 2],
         gate_err=rows[:, 3], allad_err=rows[:, 4], recovery=rows[:, 5], tau=TAU, K=K, M=M)
print("\nsaved shared_bias_sweep.npz")
