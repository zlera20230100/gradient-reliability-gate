# -*- coding: utf-8 -*-
# Trust-step-verify testbed with a cheap exact oracle. All K components affect the objective;
# some are poorly identifiable (weakly excited + label noise), so the surrogate gradient there
# is large per-model but sign-unstable across seeds. Compares a single-model gradient step (all
# components) against an ensemble gate that steps only the sign-stable components, scored on the
# oracle objective.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np, torch, torch.nn as nn
DEV = torch.device('cpu')
RNG = np.random.default_rng(0)
K = 6
c_true = np.array([1.0, -0.8, 0.6, -0.7, 0.5, -0.4])      # all components affect the objective
WELL = np.array([1, 1, 0, 1, 0, 1], dtype=bool)            # components 2,4 are poorly excited in the data
EXC = np.where(WELL, 1.0, 0.03)                            # components 2,4 essentially unexcited -> unidentifiable
NOISE = 0.15

def oracle_R(g): return np.atleast_2d(g) @ c_true          # cheap exact response
TARGET = 1.0
def oracle_L(g): return float((oracle_R(g)[0] - TARGET) ** 2)
gstar_L = 0.0                                              # target is reachable -> optimum L = 0

NS = 70
Gtr = RNG.uniform(-1.0, 1.0, (NS, K)) * EXC[None, :]       # components 2,4 barely vary -> unidentifiable
Rtr = oracle_R(Gtr) + NOISE * RNG.standard_normal(NS)

class Sur(nn.Module):
    def __init__(s):
        super().__init__(); s.net = nn.Sequential(nn.Linear(K, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(s, x): return s.net(x)

def train_member(seed):
    torch.manual_seed(seed)
    m = Sur().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    X = torch.tensor(Gtr, dtype=torch.float32); Y = torch.tensor(Rtr, dtype=torch.float32).view(-1, 1)
    for _ in range(1500):
        opt.zero_grad(); (((m(X) - Y) ** 2).mean()).backward(); opt.step()
    return m

M = 10
ens = [train_member(s) for s in range(M)]

def member_gradR(model, g):                                # dR/dg from one surrogate at g
    x = torch.tensor(g.reshape(1, K), dtype=torch.float32, requires_grad=True)
    model(x).backward(); return x.grad.numpy().ravel()
def ens_jac(g): return np.array([member_gradR(m, g) for m in ens])   # (M,K)

g0 = np.zeros(K)
J = ens_jac(g0)
sign_agree = np.maximum((J > 0).mean(0), (J < 0).mean(0))
TAU = 0.9; trust = sign_agree >= TAU
print("well-excited (identifiable) components:", WELL.astype(int).tolist())
print("ensemble sign-agreement              :", np.round(sign_agree, 2).tolist())
print("gate TRUST (>=%.1f)                  :" % TAU, trust.astype(int).tolist())
print(f"gate vs identifiability: {(trust==WELL).sum()}/{K} correct; per-seed |grad| on poorly-excited "
      f"comps mean={np.abs(J[:,~WELL]).mean():.2f} (large) but sign-unstable")

def member_R(model, g): return float(model(torch.tensor(g.reshape(1, K), dtype=torch.float32)))

def step_with(grad_fn, mask, steps=40, lr=0.12):
    g = g0.copy()
    for _ in range(steps):
        gR, Rp = grad_fn(g)
        dL = 2.0 * (Rp - TARGET) * gR * mask
        g = np.clip(g - lr * dL, -2.5, 2.5)
    return g

# gate step: ensemble-mean gradient, trusted components only
def gate_grad(g): return ens_jac(g).mean(0), float(np.mean([member_R(m, g) for m in ens]))
g_gate = step_with(gate_grad, trust.astype(float)); L_gate = oracle_L(g_gate)

# naive step: a single model's full gradient (no ensemble), repeated over the 10 members
L_naive = []
for m in ens:
    g_n = step_with(lambda g: (member_gradR(m, g), member_R(m, g)), np.ones(K))
    L_naive.append(oracle_L(g_n))
L_naive = np.array(L_naive)

L0 = oracle_L(g0)
print(f"\noracle-verified objective L (optimum = 0; lower better):")
print(f"  start                         L = {L0:.4f}")
print(f"  GATE (ensemble, trusted-only) L = {L_gate:.4f}")
print(f"  NAIVE (single model, all comp) L = {L_naive.mean():.4f} +- {L_naive.std():.4f}  "
      f"(worst {L_naive.max():.4f}, best {L_naive.min():.4f})")
print(f"  -> gate reduces L by {100*(1-L_gate/L0):.0f}%; naive only {100*(1-L_naive.mean()/L0):.0f}% on "
      f"average and is erratic ({(L_naive>L_gate).sum()}/{M} single-model runs worse than the gate)")
np.savez('endtoend.npz', c_true=c_true, well=WELL, sign_agree=sign_agree, trust=trust, tau=TAU,
         L0=L0, L_gate=L_gate, L_naive=L_naive, M=M)
print("saved endtoend.npz")
