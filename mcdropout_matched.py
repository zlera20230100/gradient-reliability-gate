# -*- coding: utf-8 -*-
# Capacity- and compute-matched MC-dropout baseline.
#
# Matched against a single deep-ensemble member:
#   * Model    -- same ZonePINN as zones_multiseed.py: NearFieldNet(4+K, hidden=128,
#                 4 layers) + FarFieldNet(4+K, Fourier features -> hidden=256, 4 layers),
#                 same coord/output scaling, blending probe, K=6 zone conditioning.
#   * Training -- same optimiser (Adam, lr=1e-3), iterations (N_ITERS, default 3000),
#                 collocation counts, loss weights, grad-clip (1.0), and random
#                 source/aperture sampling. The model here is one ensemble member plus
#                 Dropout layers.
#   * Compute  -- stochastic forward passes T set to ensemble size M (from
#                 zones_multiseed.npz, M=10). Both spend M test-time forward passes.
#
# Dropout placement (see add_dropout / enable_mc_dropout):
#   Dropout(p) after every Tanh in the hidden trunk of both sub-networks. With
#   num_layers=4 each trunk has 3 Tanh activations, so 3 Dropout layers per sub-net = 6.
#   Not applied to the input/Fourier-feature encoding (keeps FarFieldNet's high-frequency
#   basis deterministic) or to the final output Linear (Gal & Ghahramani 2016). This is
#   the standard "dropout on every hidden layer" placement.
#
# Output: mcdropout_matched.npz with the same sign-agreement / SNR indicators as the
#   deep ensemble and the same labelled AUC over-confidence diagnostic as reliability.py
#   (responsive=reliable, null + real-radiation = unreliable), with MC-dropout's
#   radiation indicator substituted for the ensemble's.
#
# Usage:  python mcdropout_matched.py [n_iters=3000] [p_drop=0.10]
# Needs the project PINN framework (pinn_model.py, config.py, main.py), same as
# mcdropout.py / zones_multiseed.py.
import os, sys, time, numpy as np, torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch.nn as nn
CODE = r"D:\实践三号“延安”\实践三号“延安”\代码"
PAPER = r"D:\实践三号“延安”\论文"
REPO  = os.path.dirname(os.path.abspath(__file__))   # repo_paper2 (npz inputs live here)
sys.path.insert(0, CODE); os.chdir(CODE)
import logging; logging.disable(logging.INFO)
from config import FREQUENCY, EXCITATION_CONFIG, Z_INTERFACE, device
from main import setup_environment, load_antenna_geometry
from pinn_model import NearFieldNet, FarFieldNet, MaxwellEquationsMicrostrip

# ---- deep-ensemble member training schedule -------------------------------
N_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000   # ensemble per-member iters
P_DROP  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
K = 6; G_LO, G_HI = 0.80, 1.20

# ---- compute-matched: T stochastic passes = ensemble size M --------------
de_path = os.path.join(REPO, 'zones_multiseed.npz')
if not os.path.exists(de_path):
    de_path = os.path.join(PAPER, 'zones_multiseed.npz')
de = np.load(de_path)
M_ENS = int(np.array(de['rela']).shape[0])                 # ensemble size M (=10)
T_MC  = M_ENS                                              # compute-matched
print(f"device={device} N_ITERS={N_ITERS} p_drop={P_DROP} "
      f"M_ensemble={M_ENS} T_MC(matched)={T_MC} K={K}", flush=True)

setup_environment()
mesh = load_antenna_geometry(substrate_path="../模型/1.1.STL", patch_path="../模型/1-1.stl", headless=True)
points, entities, scale = mesh.sample_points()
bres = mesh.sample_boundary_points(); b_points, b_ent, b_normals = bres[:3]
pp = mesh.patch_params
def T(a): return torch.tensor(a, dtype=torch.float32, device=device)
cx = np.array([c['cx'] for c in pp])*1e-3; cy = np.array([c['cy'] for c in pp])*1e-3
loc = np.array(EXCITATION_CONFIG['location'])
r_cell = np.sqrt((cx-loc[0])**2 + (cy-loc[1])**2)
order = np.argsort(r_cell); zone = np.zeros(len(pp), dtype=np.int64)
for k, idx in enumerate(np.array_split(order, K)): zone[idx] = k
ALL_X=T(points[:,0:1]*scale); ALL_Y=T(points[:,1:2]*scale); ALL_Z=T(points[:,2:3]*scale)
mask_pec=(b_ent=='ground_plane')|(b_ent=='patch'); mask_abc=(b_ent=='air_box')|(b_ent=='substrate')
def pack(m):
    idx=np.where(m)[0]
    return dict(x=T(b_points[idx,0:1]*scale),y=T(b_points[idx,1:2]*scale),z=T(b_points[idx,2:3]*scale),
                nx=T(b_normals[idx,0:1]),ny=T(b_normals[idx,1:2]),nz=T(b_normals[idx,2:3]))
PEC=pack(mask_pec); ABC=pack(mask_abc)

# ---- ZonePINN, same as zones_multiseed.py (one ensemble member) ----------
class ZonePINN(nn.Module):
    def __init__(self, scale, pp, zone, K):
        super().__init__()
        self.coord_scale_k0=scale; self.freq_base=24.0e9; self.output_scale=200000.0; self.K=K
        self.near_net=NearFieldNet(4+K); self.far_net=FarFieldNet(4+K, coord_scale_k0=scale)
        self.register_buffer('probe_loc', torch.tensor(EXCITATION_CONFIG['location']).float())
        self.blending_sigma_phys=0.003
        self.register_buffer('cx', torch.tensor([c['cx'] for c in pp]).float()*1e-3)
        self.register_buffer('cy', torch.tensor([c['cy'] for c in pp]).float()*1e-3)
        self.register_buffer('Lc', torch.tensor([c['L'] for c in pp]).float()*1e-3)
        self.register_buffer('Wc', torch.tensor([c['W'] for c in pp]).float()*1e-3)
        self.register_buffer('zone', torch.tensor(zone, dtype=torch.long))
        self.mask_sharp=2.0e6; self._g=torch.ones(K, device=device)
    def set_g(self, g): self._g = g if torch.is_tensor(g) else torch.tensor(g, dtype=torch.float32, device=device)
    def forward(self, x, y, z, freq):
        f_norm=freq/self.freq_base
        gcols=self._g.view(1,-1).expand(x.shape[0], self.K)
        inputs=torch.cat([x,y,z,f_norm,gcols], dim=1)
        E_near=self.near_net(inputs); E_far=self.far_net(inputs)
        s=self.coord_scale_k0
        px,py,pz=self.probe_loc[0]*s,self.probe_loc[1]*s,self.probe_loc[2]*s; sig=self.blending_sigma_phys*s
        W=torch.exp(-((x-px)**2+(y-py)**2+(z-pz)**2)/(2*sig**2+1e-30))
        return (W*E_near+(1-W)*E_far)*self.output_scale
    def get_patch_mask(self, x_phys, y_phys):
        s_cell=self._g[self.zone]
        mx=torch.sigmoid(self.mask_sharp*((self.Lc*s_cell)/2.0 - torch.abs(x_phys-self.cx)))
        my=torch.sigmoid(self.mask_sharp*((self.Wc*s_cell)/2.0 - torch.abs(y_phys-self.cy)))
        return torch.clamp((mx*my).sum(1,keepdim=True),0.0,1.0)

def add_dropout(seq, p):
    # insert Dropout(p) after each Tanh in the hidden trunk; reuse the existing
    # Linear layers unchanged. The output Linear (not preceded by a Tanh) gets no dropout.
    new=[]
    for m in seq:
        new.append(m)
        if isinstance(m, nn.Tanh): new.append(nn.Dropout(p))
    return nn.Sequential(*new)
def enable_mc_dropout(model):
    # Gal & Ghahramani (2016): keep the model in eval() but switch the Dropout
    # layers to train() so masks stay stochastic at test time.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

rad=EXCITATION_CONFIG['probe_radius']; n_src=4000
zmin=points[entities=='substrate'][:,2].min(); zmax=points[entities=='substrate'][:,2].max()
z_prs=Z_INTERFACE['sub2_top']; BATCH=4096
torch.manual_seed(0); np.random.seed(0)
r=rad*np.sqrt(np.random.rand(n_src,1)); th=2*np.pi*np.random.rand(n_src,1)
SRC_X=T((loc[0]+r*np.cos(th))*scale); SRC_Y=T((loc[1]+r*np.sin(th))*scale); SRC_Z=T(np.random.uniform(zmin,zmax,(n_src,1))*scale)

# ---- build one matched member, then add Dropout layers --------------------
model=ZonePINN(scale,pp,zone,K).to(device)
model.near_net.net=add_dropout(model.near_net.net, P_DROP).to(device)
model.far_net.net =add_dropout(model.far_net.net, P_DROP).to(device)
n_drop=sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
n_params=sum(p.numel() for p in model.parameters())
print(f"matched member: {n_params} params, {n_drop} Dropout layers "
      f"(after every hidden Tanh in near+far nets)", flush=True)

eq=MaxwellEquationsMicrostrip(FREQUENCY); eq.set_coord_scale(scale)
opt=torch.optim.Adam(model.parameters(), lr=1e-3)
cxmin,cxmax=model.cx.min().item(),model.cx.max().item(); cymin,cymax=model.cy.min().item(),model.cy.max().item()
t0=time.time()
for it in range(1,N_ITERS+1):
    model.train(); opt.zero_grad()
    g=torch.tensor(np.random.uniform(G_LO,G_HI,K), dtype=torch.float32, device=device); model.set_g(g)
    ia=torch.randint(0,ALL_X.shape[0],(BATCH,),device=device)
    xa=ALL_X[ia].detach().requires_grad_(True); ya=ALL_Y[ia].detach().requires_grad_(True); za=ALL_Z[ia].detach().requires_grad_(True)
    fa=torch.full_like(xa,FREQUENCY)
    l_all,_=eq.compute_physics_residuals(model,xa,ya,za,fa); l_div=eq.compute_divergence_residual(model,xa,ya,za,fa)
    isrc=torch.randint(0,SRC_X.shape[0],(1024,),device=device)
    xs=SRC_X[isrc].detach().requires_grad_(True); ys=SRC_Y[isrc].detach().requires_grad_(True); zs=SRC_Z[isrc].detach().requires_grad_(True)
    l_src,_=eq.compute_physics_residuals(model,xs,ys,zs,torch.full_like(xs,FREQUENCY))
    ip=torch.randint(0,PEC['x'].shape[0],(2048,),device=device)
    l_pec=eq.compute_pec_residuals(model,PEC['x'][ip],PEC['y'][ip],PEC['z'][ip],torch.full_like(PEC['x'][ip],FREQUENCY),PEC['nx'][ip],PEC['ny'][ip],PEC['nz'][ip])
    ib=torch.randint(0,ABC['x'].shape[0],(2048,),device=device)
    xb=ABC['x'][ib].detach().requires_grad_(True); yb=ABC['y'][ib].detach().requires_grad_(True); zb=ABC['z'][ib].detach().requires_grad_(True)
    l_abc=eq.compute_boundary_residuals(model,xb,yb,zb,torch.full_like(xb,FREQUENCY),ABC['nx'][ib],ABC['ny'][ib],ABC['nz'][ib])
    Np=2048; pad=0.0005
    xp=(torch.rand(Np,1,device=device)*(cxmax-cxmin+2*pad)+cxmin-pad)*scale
    yp=(torch.rand(Np,1,device=device)*(cymax-cymin+2*pad)+cymin-pad)*scale
    zp=torch.full_like(xp,z_prs*scale)
    Ep=model(xp,yp,zp,torch.full_like(xp,FREQUENCY)); Er_p,Ei_p=Ep[:,0:3],Ep[:,3:6]
    E_tan_sq=(Er_p[:,0:1]**2+Er_p[:,1:2]**2+Ei_p[:,0:1]**2+Ei_p[:,1:2]**2)/(model.output_scale**2)
    l_mpec=torch.mean(E_tan_sq*model.get_patch_mask(xp/scale,yp/scale))
    total=l_all*1.0+l_src*20.0+l_pec*1.0+l_abc*1.0+l_mpec*10.0+l_div*1.0
    total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if it%max(1,N_ITERS//5)==0 or it==1:
        print(f"  it {it:5d}/{N_ITERS} tot {float(total):.2e} all {float(l_all):.2e} ({(time.time()-t0)/it*1000:.0f}ms/it)", flush=True)
train_sec=time.time()-t0

# ---- MC-dropout sampling of the per-zone Jacobian (T_MC = M passes) -------
nft=4096
rr=torch.sqrt(torch.tensor(0.00015**2,device=device)+(0.00025**2-0.00015**2)*torch.rand(nft,1,device=device))
ang=2*np.pi*torch.rand(nft,1,device=device)
feed_xy=torch.cat([(loc[0]+rr*torch.cos(ang))*scale,(loc[1]+rr*torch.sin(ang))*scale],1)
ng=48; gx=torch.linspace(cxmin,cxmax,ng,device=device)*scale; gy=torch.linspace(cymin,cymax,ng,device=device)*scale
GX,GY=torch.meshgrid(gx,gy,indexing='ij'); ap_xy=torch.stack([GX.reshape(-1),GY.reshape(-1)],1)
def jac(points_xy, zval):
    gvec=torch.ones(K,device=device,requires_grad=True); model.set_g(gvec)
    X=points_xy[:,0:1]; Y=points_xy[:,1:2]; Zc=torch.full_like(X,zval*scale)
    E=model(X,Y,Zc,torch.full_like(X,FREQUENCY)); Q=(E[:,2]**2+E[:,5]**2).mean()
    gg=torch.autograd.grad(Q,gvec)[0].detach().cpu().numpy(); return Q.item(), gg/(Q.item()+1e-30)
enable_mc_dropout(model)
mc_relf=[]; mc_rela=[]
for t in range(T_MC):
    _,rf=jac(feed_xy, EXCITATION_CONFIG['location'][2]+0.0005); _,ra=jac(ap_xy, z_prs)
    mc_relf.append(rf); mc_rela.append(ra)
mc_relf=np.array(mc_relf); mc_rela=np.array(mc_rela)            # (T_MC, K)

# ---- same reliability indicators as the deep ensemble --------------------
def stats(J):
    sa=np.maximum((J>0).mean(0),(J<0).mean(0)); snr=np.abs(J.mean(0))/(J.std(0)+1e-12); return sa,snr
sa_mc, snr_mc = stats(mc_rela)
sa_de, snr_de = stats(de['rela'])

print("\n===== MATCHED MC-DROPOUT vs DEEP ENSEMBLE on the real-device RADIATION gradient =====")
print(f"  matched MC-dropout (1 member, {T_MC} passes): sign-agree mean {sa_mc.mean():.2f}  SNR mean {snr_mc.mean():.2f}")
print(f"  deep ensemble ({M_ENS} members)             : sign-agree mean {sa_de.mean():.2f}  SNR mean {snr_de.mean():.2f}")
print(f"  per-zone MC sign-agree: {np.round(sa_mc,2)}")
print(f"  per-zone DE sign-agree: {np.round(sa_de,2)}")
print(f"  full-wave verdict: radiation gradient UNRELIABLE (sign-match 3/6). Trust threshold 0.9.")
print(f"  -> matched MC-dropout flags distrust? {(sa_mc.mean()<0.9)}   deep ensemble flags distrust? {(sa_de.mean()<0.9)}")

# ---- labelled AUC over-confidence diagnostic, as in reliability.py --------
#   responsive (reliable) = label 1; null + real-radiation = label 0.
#   Score = the reliability indicator. AUC computed twice: with the deep
#   ensemble's radiation indicator, then with MC-dropout's substituted.
auc_mc_sign = auc_mc_snr = auc_de_sign = auc_de_snr = float('nan')
try:
    from sklearn.metrics import roc_auc_score
    ct = np.load(os.path.join(REPO, 'protocol_control.npz')) if os.path.exists(os.path.join(REPO,'protocol_control.npz')) \
         else np.load(os.path.join(PAPER, 'protocol_control.npz'))
    def ens_stats(J):
        mean=J.mean(0); std=J.std(0)
        return np.maximum((J>0).mean(0),(J<0).mean(0)), np.abs(mean)/(std+1e-12)
    sa_resp, snr_resp = ens_stats(ct['J_responsive'])     # reliable
    sa_null, snr_null = ens_stats(ct['J_null'])           # unreliable (null)
    label = np.concatenate([np.ones(K), np.zeros(K), np.zeros(K)]).astype(int)
    # deep-ensemble diagnostic (radiation block = ensemble indicator)
    auc_de_sign = roc_auc_score(label, np.concatenate([sa_resp,  sa_null,  sa_de]))
    auc_de_snr  = roc_auc_score(label, np.concatenate([snr_resp, snr_null, snr_de]))
    # matched MC-dropout diagnostic (radiation block = MC indicator)
    auc_mc_sign = roc_auc_score(label, np.concatenate([sa_resp,  sa_null,  sa_mc]))
    auc_mc_snr  = roc_auc_score(label, np.concatenate([snr_resp, snr_null, snr_mc]))
    print("\nover-confidence diagnostic (AUC: separate reliable from unreliable; higher=better):")
    print(f"  deep ensemble     : AUC(sign-agree)={auc_de_sign:.3f}  AUC(SNR)={auc_de_snr:.3f}")
    print(f"  matched MC-dropout: AUC(sign-agree)={auc_mc_sign:.3f}  AUC(SNR)={auc_mc_snr:.3f}")
    print(f"  (MC AUC < DE AUC => MC-dropout is over-confident on the unreliable radiation gradient)")
except Exception as e:
    print(f"\n[warn] AUC diagnostic skipped ({type(e).__name__}: {e}); reliability indicators still saved.")

np.savez(os.path.join(REPO, 'mcdropout_matched.npz'),
         mc_relf=mc_relf, mc_rela=mc_rela, sa_mc=sa_mc, snr_mc=snr_mc, sa_de=sa_de, snr_de=snr_de,
         p_drop=P_DROP, T_mc=T_MC, M_ensemble=M_ENS, n_iters=N_ITERS,
         n_params=n_params, n_dropout_layers=n_drop, train_sec=train_sec,
         auc_mc_sign=auc_mc_sign, auc_mc_snr=auc_mc_snr,
         auc_de_sign=auc_de_sign, auc_de_snr=auc_de_snr)
print("\nsaved mcdropout_matched.npz", flush=True)
