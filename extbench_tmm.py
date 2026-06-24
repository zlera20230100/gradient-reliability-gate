# -*- coding: utf-8 -*-
# Reliability gate on a thin-film multilayer Fabry-Perot benchmark via the transfer-matrix method
# (TMM). The surrogate's autodiff gradient often has the right sign but wrong magnitude; tests
# whether ordering components by sign-agreement and FD-correcting the flagged ones first recovers
# the descent direction (cosine to the TMM gradient) with fewer solves than random ordering.
# Ground truth = central FD on the TMM oracle; gate = deep-ensemble per-component sign-agreement.
# Writes extbench_tmm.npz.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

LAM0 = 1550.0
nH, nL, nsub, n0 = 2.35, 1.45, 1.52, 1.0
IDX = np.array([nH, nL, nH, nL, nH, nL, nH, nL, nH, nL, nH])
QW = LAM0 / (4.0 * IDX); NOMINAL = QW.copy(); NOMINAL[5] *= 2.0
K = IDX.size

def tmm_R(thick, lam):
    M = np.eye(2, dtype=complex)
    for nj, dj in zip(IDX, thick):
        delta = 2.0 * np.pi * nj * dj / lam
        c, s = np.cos(delta), np.sin(delta)
        M = M @ np.array([[c, 1j * s / nj], [1j * nj * s, c]], dtype=complex)
    B = M[0, 0] + M[0, 1] * nsub; C = M[1, 0] + M[1, 1] * nsub
    r = (n0 * B - C) / (n0 * B + C)
    return float(np.abs(r) ** 2)

_scan = np.linspace(LAM0 - 60, LAM0 + 60, 481)
_Rs = np.array([tmm_R(NOMINAL, l) for l in _scan]); _up = _scan > LAM0
LAM_EVAL = float(_scan[_up][np.argmin(np.abs(_Rs[_up] - 0.4))])

def tmm_grad_fd(thick, h=0.5):
    g = np.zeros(K)
    for k in range(K):
        tp = thick.copy(); tp[k] += h; tm = thick.copy(); tm[k] -= h
        g[k] = (tmm_R(tp, LAM_EVAL) - tmm_R(tm, LAM_EVAL)) / (2 * h)
    return g

print(f"working point lambda_eval={LAM_EVAL:.1f} nm, R={tmm_R(NOMINAL, LAM_EVAL):.3f} (resonance flank)")

RNG = np.random.default_rng(7)
NS = int(os.environ.get('NS', '600')); span = 0.30
X = NOMINAL[None, :] * (1.0 + span * (2 * RNG.random((NS, K)) - 1))
Y = np.array([tmm_R(x, LAM_EVAL) for x in X])
xm, xs = X.mean(0), X.std(0); ym, ysd = Y.mean(), Y.std()
Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
Yt = torch.tensor((Y - ym) / ysd, dtype=torch.float32).view(-1, 1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_one(seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=2e-3); lossf = nn.MSELoss()
    n = Xt.size(0); idx = torch.randperm(n, generator=g); tr = idx[:int(0.9 * n)]
    for ep in range(400):
        perm = tr[torch.randperm(tr.numel(), generator=g)]
        for b in perm.split(256):
            opt.zero_grad(); lossf(net(Xt[b]), Yt[b]).backward(); opt.step()
    return net

M = 10
nets = [train_one(s) for s in range(M)]
print(f"ensemble trained (M={M}, NS={NS})")

def ad_grad(net, xphys):
    xn = torch.tensor(((xphys - xm) / xs), dtype=torch.float32).view(1, -1).requires_grad_(True)
    y = net(xn); y.backward()
    return (xn.grad.detach().numpy().ravel()) * (ysd / xs)

# evaluate at query points near the nominal stack (all on the flank)
NQ = 80
QP = NOMINAL[None, :] * (1.0 + 0.12 * (2 * RNG.random((NQ, K)) - 1))
TAU = 0.9
all_sa, all_correct = [], []
# frontiers: cosine vs #FD-solves, for gate-order / random-order / oracle-order
cos_gate = np.zeros((NQ, K + 1)); cos_rand = np.zeros((NQ, K + 1)); cos_orac = np.zeros((NQ, K + 1))
def cossim(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rng2 = np.random.default_rng(123)
n_trust_at_tau = []
for q, xq in enumerate(QP):
    Gm = np.array([ad_grad(net, xq) for net in nets])      # (M,K)
    ad = Gm.mean(0); sa = np.maximum((Gm > 0).mean(0), (Gm < 0).mean(0))
    fd = tmm_grad_fd(xq)
    all_sa.append(sa); all_correct.append((np.sign(ad) == np.sign(fd)).astype(int))
    n_trust_at_tau.append(int((sa >= TAU).sum()))
    # orderings: which components to FD-correct first
    order_gate = np.argsort(sa)                              # least reliable first
    order_orac = np.argsort(-np.abs(ad - fd))                # truly-worst first (oracle)
    for n in range(K + 1):
        g = ad.copy(); g[order_gate[:n]] = fd[order_gate[:n]]; cos_gate[q, n] = cossim(g, fd)
        g = ad.copy(); g[order_orac[:n]] = fd[order_orac[:n]]; cos_orac[q, n] = cossim(g, fd)
        # random order: average over a few perms
        acc = 0.0; R = 8
        for _ in range(R):
            perm = rng2.permutation(K); g = ad.copy(); g[perm[:n]] = fd[perm[:n]]; acc += cossim(g, fd)
        cos_rand[q, n] = acc / R

all_sa = np.concatenate(all_sa); all_correct = np.concatenate(all_correct)
auc = roc_auc_score(all_correct, all_sa) if 0 < all_correct.sum() < all_correct.size else float('nan')
cg, cr, co = cos_gate.mean(0), cos_rand.mean(0), cos_orac.mean(0)
print(f"\npooled over {NQ} query points x {K} comps = {all_correct.size} samples:")
print(f"  fraction of AD components sign-correct: {all_correct.mean():.2f}")
print(f"  gate AUC (sign-agreement predicts AD-sign-correct, label-decoupled): {auc:.3f}")
print(f"  mean #components trusted at tau={TAU}: {np.mean(n_trust_at_tau):.1f}/{K}")
print("\ncosine-to-truth vs #FD solves (descent-direction recovery):")
print(f"{'#FD':>4} {'gate-order':>11} {'random-order':>13} {'oracle-order':>13}")
for n in range(K + 1):
    print(f"{n:>4} {cg[n]:>11.3f} {cr[n]:>13.3f} {co[n]:>13.3f}")
# solver saving: #FD for gate to reach a target cosine vs random
def solves_to(curve, target):
    idx = np.where(curve >= target)[0]
    return int(idx[0]) if len(idx) else K
for tgt in (0.95, 0.99):
    sg, sr = solves_to(cg, tgt), solves_to(cr, tgt)
    print(f"  to reach cosine>={tgt}: gate-order {sg} solves vs random-order {sr} solves "
          f"-> {sr - sg} fewer ({100*(sr-sg)/max(sr,1):.0f}% saving)")

np.savez('extbench_tmm.npz', auc=auc, frac_correct=all_correct.mean(),
         cos_gate=cg, cos_rand=cr, cos_orac=co, K=K, tau=TAU, NS=NS, NQ=NQ,
         n_trust_mean=np.mean(n_trust_at_tau), lam_eval=LAM_EVAL,
         all_sa=all_sa, all_correct=all_correct, cos_ad=float(cg[0]))
print("\nsaved extbench_tmm.npz")
