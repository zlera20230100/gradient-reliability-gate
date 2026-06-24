# -*- coding: utf-8 -*-
# Harder external thin-film TMM benchmark for the reliability gate (companion to extbench_tmm.py).
# extbench_tmm.py runs in a near-saturated regime (~97% of autodiff gradient components already
# sign-correct, base cosine-to-truth ~0.996), so the allocator has little to save. Here the problem
# is made harder; the surrogate training recipe, sign-agreement gate, allocator/ordering logic,
# central-FD ground truth and all metrics are reused from extbench_tmm.py:
#   (1) fewer surrogate training samples (NS=250 vs 600).
#   (2) a half-wave defect in the centre of the stack -> sharper near-band feature.
#   (3) evaluate on a flank of that feature (R~0.4), query points roam to +-10% -> mild
#       extrapolation, ~20% of gradient components become sign-wrong instead of ~3%.
# TMM oracle math is identical to extbench_tmm.py. Writes extbench_tmm_hard.npz.
# CPU-only, tiny MLP, M=10, fixed seeds.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
import numpy as np
import torch, torch.nn as nn
torch.set_num_threads(3)
from sklearn.metrics import roc_auc_score

LAM0 = 1550.0
nH, nL, nsub, n0 = 2.35, 1.45, 1.52, 1.0
# 11-layer quarter-wave stack (same K as the easy bench)
IDX = np.array([nH, nL] * 5 + [nH])
QW = LAM0 / (4.0 * IDX)
NOMINAL = QW.copy()
# half-wave defect in the centre -> sharper near-band resonance
NOMINAL[len(IDX) // 2] *= 2.0
K = IDX.size

def tmm_R(thick, lam):
    # TMM oracle: characteristic-matrix product, normal incidence
    M = np.eye(2, dtype=complex)
    for nj, dj in zip(IDX, thick):
        delta = 2.0 * np.pi * nj * dj / lam
        c, s = np.cos(delta), np.sin(delta)
        M = M @ np.array([[c, 1j * s / nj], [1j * nj * s, c]], dtype=complex)
    B = M[0, 0] + M[0, 1] * nsub; C = M[1, 0] + M[1, 1] * nsub
    r = (n0 * B - C) / (n0 * B + C)
    return float(np.abs(r) ** 2)

# flank of the defect feature, R~0.4 just above LAM0; sharper feature makes the local gradient
# harder for the MLP to learn.
_scan = np.linspace(LAM0 - 80, LAM0 + 80, 1601)
_Rs = np.array([tmm_R(NOMINAL, l) for l in _scan]); _up = _scan > LAM0
LAM_EVAL = float(_scan[_up][np.argmin(np.abs(_Rs[_up] - 0.4))])
print(f"working point lambda_eval={LAM_EVAL:.1f} nm, R={tmm_R(NOMINAL, LAM_EVAL):.3f} "
      f"(flank of centre defect resonance; K={K}-layer stack)")

def tmm_grad_fd(thick, h=0.5):
    g = np.zeros(K)
    for k in range(K):
        tp = thick.copy(); tp[k] += h; tm = thick.copy(); tm[k] -= h
        g[k] = (tmm_R(tp, LAM_EVAL) - tmm_R(tm, LAM_EVAL)) / (2 * h)
    return g

RNG = np.random.default_rng(7)
NS = int(os.environ.get('NS', '250'))          # fewer samples than the 600 in the easy bench
span = 0.30
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
    # same training recipe as extbench_tmm.py
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
print(f"ensemble trained (M={M}, NS={NS}, K={K})")

def ad_grad(net, xphys):
    xn = torch.tensor(((xphys - xm) / xs), dtype=torch.float32).view(1, -1).requires_grad_(True)
    y = net(xn); y.backward()
    return (xn.grad.detach().numpy().ravel()) * (ysd / xs)

# query points roam to +-10% of the nominal stack -> mild extrapolation
NQ = 100
QP = NOMINAL[None, :] * (1.0 + 0.10 * (2 * RNG.random((NQ, K)) - 1))
TAU = 0.9
all_sa, all_correct = [], []
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
    # allocator orderings: which components to FD-correct first
    order_gate = np.argsort(sa)                              # least reliable first (gate-driven)
    order_orac = np.argsort(-np.abs(ad - fd))               # worst first (oracle upper bound)
    for n in range(K + 1):
        g = ad.copy(); g[order_gate[:n]] = fd[order_gate[:n]]; cos_gate[q, n] = cossim(g, fd)
        g = ad.copy(); g[order_orac[:n]] = fd[order_orac[:n]]; cos_orac[q, n] = cossim(g, fd)
        acc = 0.0; R = 8
        for _ in range(R):
            perm = rng2.permutation(K); g = ad.copy(); g[perm[:n]] = fd[perm[:n]]; acc += cossim(g, fd)
        cos_rand[q, n] = acc / R

all_sa = np.concatenate(all_sa); all_correct = np.concatenate(all_correct)
n_pos = int(all_correct.sum()); n_neg = int(all_correct.size - n_pos)
auc = roc_auc_score(all_correct, all_sa) if 0 < n_pos < all_correct.size else float('nan')
cg, cr, co = cos_gate.mean(0), cos_rand.mean(0), cos_orac.mean(0)
frac_wrong = 1.0 - all_correct.mean()

print(f"\npooled over {NQ} query points x {K} comps = {all_correct.size} samples:")
print(f"  class balance: #sign-correct={n_pos}  #sign-wrong={n_neg}  (sign-wrong fraction={frac_wrong:.3f})")
print(f"  gate AUC (sign-agreement predicts AD-sign-correct, label-decoupled): {auc:.3f}")
print(f"  mean #components trusted at tau={TAU}: {np.mean(n_trust_at_tau):.1f}/{K}")
print("\ncosine-to-truth vs #FD(TMM) solves (descent-direction recovery):")
print(f"{'#FD':>4} {'gate-order':>11} {'random-order':>13} {'oracle-order':>13} {'gate-rand':>10}")
for n in range(K + 1):
    print(f"{n:>4} {cg[n]:>11.3f} {cr[n]:>13.3f} {co[n]:>13.3f} {cg[n]-cr[n]:>+10.3f}")

def solves_to(curve, target):
    idx = np.where(curve >= target)[0]
    return int(idx[0]) if len(idx) else K

print("\nALLOCATOR result -- TMM solves to reach a target descent-direction cosine:")
print(f"  base cosine at 0 FD solves (pure surrogate gradient): {cg[0]:.3f}")
print(f"  all-FD baseline = {K} solves (full TMM gradient, cosine=1.000)")
print(f"  {'target':>7} {'gate':>5} {'random':>7} {'all-FD':>7} {'saved-vs-random':>16} {'saved-vs-allFD':>15}")
save_extra = {}
for tgt in (0.84, 0.86, 0.88, 0.90, 0.95, 0.99):
    sg, sr = solves_to(cg, tgt), solves_to(cr, tgt)
    saved = sr - sg
    pct = 100 * saved / max(sr, 1)
    print(f"  {tgt:>7.2f} {sg:>5} {sr:>7} {K:>7} {f'{saved} ({pct:.0f}%)':>16} {f'{K - sg}':>15}")
    save_extra[f'solves_gate_{tgt}'] = sg
    save_extra[f'solves_rand_{tgt}'] = sr

np.savez('extbench_tmm_hard.npz', auc=auc, frac_correct=all_correct.mean(), frac_wrong=frac_wrong,
         n_sign_correct=n_pos, n_sign_wrong=n_neg,
         cos_gate=cg, cos_rand=cr, cos_orac=co, K=K, tau=TAU, NS=NS, NQ=NQ, M=M,
         n_trust_mean=np.mean(n_trust_at_tau), lam_eval=LAM_EVAL,
         all_sa=all_sa, all_correct=all_correct, cos_ad=float(cg[0]), **save_extra)
print("\nsaved extbench_tmm_hard.npz")
