# -*- coding: utf-8 -*-
# quick sweep to locate a DEFENSIBLE harder regime: target sign-wrong fraction ~0.10-0.30 with AUC>>0.5
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
LAM0=1550.0; n0=1.0; nsub=1.52

def build(nH,nL,nlayers,defect):
    IDX=np.array(([nH,nL]*nlayers)[:nlayers]); 
    if nlayers%2==0: IDX=np.array(([nH,nL]*nlayers)[:nlayers])
    IDX=np.array(([nH,nL]* (nlayers//2+1))[:nlayers])
    QW=LAM0/(4.0*IDX); NOM=QW.copy(); NOM[nlayers//2]*=defect
    return IDX,NOM

def run(nH,nL,nlayers,defect,NS,qspan,flank='steep',seed=7):
    IDX,NOMINAL=build(nH,nL,nlayers,defect); K=IDX.size
    def tmm_R(thick,lam):
        M=np.eye(2,dtype=complex)
        for nj,dj in zip(IDX,thick):
            delta=2*np.pi*nj*dj/lam; c,s=np.cos(delta),np.sin(delta)
            M=M@np.array([[c,1j*s/nj],[1j*nj*s,c]],dtype=complex)
        B=M[0,0]+M[0,1]*nsub; C=M[1,0]+M[1,1]*nsub
        r=(n0*B-C)/(n0*B+C); return float(abs(r)**2)
    sc=np.linspace(LAM0-80,LAM0+80,1601); Rs=np.array([tmm_R(NOMINAL,l) for l in sc])
    if flank=='steep':
        dR=np.gradient(Rs,sc); LAM=float(sc[np.argmax(np.abs(dR))])
    else:
        up=sc>LAM0; LAM=float(sc[up][np.argmin(np.abs(Rs[up]-0.4))])
    def fd(thick,h=0.5):
        g=np.zeros(K)
        for k in range(K):
            tp=thick.copy(); tp[k]+=h; tm=thick.copy(); tm[k]-=h
            g[k]=(tmm_R(tp,LAM)-tmm_R(tm,LAM))/(2*h)
        return g
    RNG=np.random.default_rng(seed); span=0.30
    X=NOMINAL[None,:]*(1+span*(2*RNG.random((NS,K))-1)); Y=np.array([tmm_R(x,LAM) for x in X])
    xm,xs=X.mean(0),X.std(0); ym,ysd=Y.mean(),Y.std()
    Xt=torch.tensor((X-xm)/xs,dtype=torch.float32); Yt=torch.tensor((Y-ym)/ysd,dtype=torch.float32).view(-1,1)
    class MLP(nn.Module):
        def __init__(s):
            super().__init__(); s.net=nn.Sequential(nn.Linear(K,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
        def forward(s,x): return s.net(x)
    def train(sd):
        torch.manual_seed(sd); g=torch.Generator().manual_seed(sd)
        net=MLP(); opt=torch.optim.Adam(net.parameters(),lr=2e-3); lf=nn.MSELoss()
        n=Xt.size(0); idx=torch.randperm(n,generator=g); tr=idx[:int(0.9*n)]
        for ep in range(400):
            perm=tr[torch.randperm(tr.numel(),generator=g)]
            for b in perm.split(256):
                opt.zero_grad(); lf(net(Xt[b]),Yt[b]).backward(); opt.step()
        return net
    nets=[train(s) for s in range(10)]
    def ag(net,xp):
        xn=torch.tensor(((xp-xm)/xs),dtype=torch.float32).view(1,-1).requires_grad_(True)
        net(xn).backward(); return xn.grad.detach().numpy().ravel()*(ysd/xs)
    NQ=80; QP=NOMINAL[None,:]*(1+qspan*(2*RNG.random((NQ,K))-1))
    sa_all,cor_all=[],[]; cg=np.zeros((NQ,K+1)); crnd=np.zeros((NQ,K+1))
    def cs(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    r2=np.random.default_rng(123)
    for q,xq in enumerate(QP):
        Gm=np.array([ag(net,xq) for net in nets]); ad=Gm.mean(0)
        sa=np.maximum((Gm>0).mean(0),(Gm<0).mean(0)); f=fd(xq)
        sa_all.append(sa); cor_all.append((np.sign(ad)==np.sign(f)).astype(int))
        og=np.argsort(sa)
        for n in range(K+1):
            g=ad.copy(); g[og[:n]]=f[og[:n]]; cg[q,n]=cs(g,f)
            acc=0
            for _ in range(8):
                p=r2.permutation(K); g=ad.copy(); g[p[:n]]=f[p[:n]]; acc+=cs(g,f)
            crnd[q,n]=acc/8
    sa_all=np.concatenate(sa_all); cor_all=np.concatenate(cor_all)
    auc=roc_auc_score(cor_all,sa_all) if 0<cor_all.sum()<cor_all.size else float('nan')
    cgm,crm=cg.mean(0),crnd.mean(0)
    def st(curve,t):
        i=np.where(curve>=t)[0]; return int(i[0]) if len(i) else K
    out={'K':K,'fw':1-cor_all.mean(),'auc':auc,'base':cgm[0]}
    for t in (0.9,0.95,0.99): out[f'g{t}']=st(cgm,t); out[f'r{t}']=st(crm,t)
    return out

configs=[
 ("orig-but-steep",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=600,qspan=0.12,flank='steep')),
 ("NS200_steep",   dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=200,qspan=0.15,flank='steep')),
 ("NS150_K13",     dict(nH=2.35,nL=1.45,nlayers=13,defect=2.0,NS=150,qspan=0.15,flank='steep')),
 ("NS200_strongC", dict(nH=2.55,nL=1.45,nlayers=13,defect=2.0,NS=200,qspan=0.15,flank='steep')),
 ("NS250_K13_q12", dict(nH=2.55,nL=1.45,nlayers=13,defect=2.0,NS=250,qspan=0.12,flank='steep')),
]
print(f"{'name':>16} {'K':>3} {'frac_wrong':>10} {'auc':>6} {'base':>6} | {'g.90':>4}{'r.90':>5} {'g.95':>4}{'r.95':>5} {'g.99':>4}{'r.99':>5}")
for nm,kw in configs:
    o=run(**kw)
    print(f"{nm:>16} {o['K']:>3} {o['fw']:>10.3f} {o['auc']:>6.3f} {o['base']:>6.3f} | "
          f"{o['g0.9']:>4}{o['r0.9']:>5} {o['g0.95']:>4}{o['r0.95']:>5} {o['g0.99']:>4}{o['r0.99']:>5}")

print("\n--- finer grid toward moderate regime ---")
configs2=[
 ("K11_NS300_flank",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=300,qspan=0.12,flank='flank')),
 ("K11_NS200_flank",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=200,qspan=0.12,flank='flank')),
 ("K11_NS150_flank",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=150,qspan=0.12,flank='flank')),
 ("K11_NS150_q15f", dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=150,qspan=0.15,flank='flank')),
 ("K11_NS180_steep",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=180,qspan=0.12,flank='steep')),
 ("K11_NS250_steep",dict(nH=2.35,nL=1.45,nlayers=11,defect=2.0,NS=250,qspan=0.12,flank='steep')),
 ("K9_NS150_steep", dict(nH=2.35,nL=1.45,nlayers=9, defect=2.0,NS=150,qspan=0.12,flank='steep')),
]
print(f"{'name':>16} {'K':>3} {'frac_wrong':>10} {'auc':>6} {'base':>6} | {'g.90':>4}{'r.90':>5} {'g.95':>4}{'r.95':>5} {'g.99':>4}{'r.99':>5}")
for nm,kw in configs2:
    o=run(**kw)
    print(f"{nm:>16} {o['K']:>3} {o['fw']:>10.3f} {o['auc']:>6.3f} {o['base']:>6.3f} | "
          f"{o['g0.9']:>4}{o['r0.9']:>5} {o['g0.95']:>4}{o['r0.95']:>5} {o['g0.99']:>4}{o['r0.99']:>5}")
