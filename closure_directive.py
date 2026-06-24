# -*- coding: utf-8 -*-
# Full-wave FPC beam-steering closure on a degree-grid NF2FF.
# Earlier versions (closure.py / closure_multiangle.py) passed theta in radians to
# CalcNF2FF, which expects degrees, so the far-field sweep spanned only +-1.57 deg and read as
# a near-constant pattern. Here NF2FF is evaluated on a degree grid, as in _fix_patterns.py.
#
# Uses the directive model (SUB=24, AIRGAP=5.4, PATCH=2.8) that gives an 11.1 dBi broadside
# beam (fpc_result.npz), not the SUB=16 closure model. Two coding routes for each target angle
# theta0 in {0,10,20,30}:
#   (A) physics phase-gradient: required aperture phase phi(x) = -k0 sin(theta0) x, mapped to a
#       per-column cell size scale via the unit-cell size->phase curve (_uc_ref.npz, 211 deg swing).
#   (B) surrogate g: per-column scales from inverse.npz (g_uniform/g_broadside/g_steer20)
#       and closure_multiangle.npz (steer10_g, steerm15_g).
#
# Usage:
#   python closure_directive.py validate         # one uniform run, confirm ~11 dBi degree-grid beam
#   python closure_directive.py grad             # physics phase-gradient sweep
#   python closure_directive.py surr             # surrogate-g sweep
#   python closure_directive.py all              # everything
import os, sys, json, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.add_dll_directory(r"D:\openEMS_pkg\openEMS")
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

def P(*a): print(*a, flush=True)
PAPER = os.path.dirname(os.path.abspath(__file__))

# geometry / model constants (directive model)
G = json.load(open(os.path.join(PAPER, "_geom.json")))
cells = np.array([c[:4] for c in G["cells"]])               # (96,4) x0,y0,x1,y1 [mm]
feed = G["feed"][0]; fx = 0.5*(feed[0]+feed[2]); fy = 0.5*(feed[1]+feed[3])
cxc = 0.5*(cells[:,0]+cells[:,2])
xu = np.unique(np.round(cxc,3)); col = np.searchsorted(xu, np.round(cxc,3))   # 0..7 columns by x
xcol_centers = xu * 1e-3                                     # column x positions [m]
P(f"{len(cells)} cells; 8 columns; feed=({fx:.2f},{fy:.2f}); col x-centers(mm)={np.round(xu,2)}")

unit = 1e-3; f0, fc = 24e9, 12e9
SUB = 24.0; PATCH = 2.8; AIRGAP = 5.4; margin = 7.0
z_g, z_s1 = 0.0, 1.0; z_air = z_s1 + AIRGAP; z_s2 = z_air + 1.0
epsR, tand = 4.4, 0.02
APFINE = 0.06                                                # aperture fine step (as in _fix_patterns)
k0 = 2*np.pi*f0/C0

# degree theta grid
THD = np.linspace(-90, 90, 181)

# unit-cell size->phase curve for the physics route
UC = np.load(os.path.join(PAPER, "_uc_ref.npz"))
ucL = UC['L']; ucPh = UC['phase_unwrap']                     # deg

def required_g_for_angle(theta0_deg):
    """Map required per-column reflection phase phi(x)=-k0 sin(theta0) x to a per-column size
    scale g via the unit-cell size->phase curve. Phase referenced to broadside. Returns g (len 8)."""
    th0 = np.radians(theta0_deg)
    phi_req = -k0 * np.sin(th0) * xcol_centers               # rad
    phi_req = np.degrees(phi_req)
    phi_req = phi_req - phi_req.mean()                        # only relative gradient matters
    # invert phase->L on the monotonic-ish portion; uc curve defined for patch L 2..6mm.
    # Translate the needed delta-phase to a delta-size, then to a multiplicative scale g about
    # the nominal column size.
    ph = ucPh.copy(); Lc = ucL.copy()
    # make phase monotonic decreasing for a clean inverse over the usable swing
    idx = np.argsort(ph)
    ph_s = ph[idx]; L_s = Lc[idx]
    Lreq = np.interp(np.clip(phi_req, ph_s.min(), ph_s.max()), ph_s, L_s)
    Lref = np.interp(0.0, ph_s, L_s)
    g = Lreq / Lref
    return g

# the directive FPC model + NF2FF (degree grid)
def run_design(gvec, tag, target):
    sim = rf"D:\openEMS_pkg\sim_dir_{tag}"
    FDTD = openEMS(EndCriteria=1e-3, NrTS=40000); FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PML_8']*6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX); g = CSX.GetGrid(); g.SetDeltaUnit(unit)
    fr4 = CSX.AddMaterial('FR4', epsilon=epsR, kappa=2*np.pi*f0*EPS0*epsR*tand)
    gnd = CSX.AddMetal('gnd'); prs = CSX.AddMetal('prs'); via = CSX.AddMetal('via'); dp = CSX.AddMetal('dpatch')
    fr4.AddBox([-SUB/2,-SUB/2,z_g],  [SUB/2,SUB/2,z_s1], priority=1)
    fr4.AddBox([-SUB/2,-SUB/2,z_air],[SUB/2,SUB/2,z_s2], priority=1)
    gnd.AddBox([-SUB/2,-SUB/2,z_g],[SUB/2,SUB/2,z_g], priority=10)
    xe=set(); ye=set()
    for i,(x0,y0,x1,y1) in enumerate(cells):
        s = float(gvec[col[i]]); cxi=0.5*(x0+x1); cyi=0.5*(y0+y1)
        Lx=(x1-x0)*s; Ly=(y1-y0)*s
        nx0,nx1,ny0,ny1 = cxi-Lx/2,cxi+Lx/2,cyi-Ly/2,cyi+Ly/2
        prs.AddBox([nx0,ny0,z_s2],[nx1,ny1,z_s2], priority=10)
        xe.update([nx0,nx1,cxi]); ye.update([ny0,ny1,cyi])
    hp=0.10; zgap=0.12
    port = FDTD.AddLumpedPort(1,50.0,[fx-hp,fy-hp,z_g],[fx+hp,fy+hp,zgap],'z',1.0,priority=5,edges2grid='xy')
    via.AddBox([fx-hp,fy-hp,zgap],[fx+hp,fy+hp,z_s1], priority=10)
    xe.update([fx-hp,fx+hp]); ye.update([fy-hp,fy+hp])
    Lpy=PATCH; Lpx=PATCH*0.85
    dp.AddBox([-Lpx/2,-Lpy/2,z_s1],[Lpx/2,Lpy/2,z_s1], priority=12)
    xe.update([-Lpx/2,Lpx/2,0.0]); ye.update([-Lpy/2,Lpy/2,0.0])
    def build(crit, lo, hi, tol=0.02):
        L = sorted(set([lo,hi]+list(crit))); out=[L[0]]
        for v in L[1:]:
            if v-out[-1] >= tol: out.append(v)
        return out
    ap = list(np.arange(-3.6, 3.6001, APFINE))
    xcrit = build(xe | set(ap) | {fx-hp,fx+hp}, -SUB/2-margin, SUB/2+margin)
    ycrit = build(ye | set(ap) | {fy-hp,fy+hp}, -SUB/2-margin, SUB/2+margin)
    g.AddLine('x', xcrit); g.AddLine('y', ycrit)
    g.AddLine('z', [-margin*1.4, z_g, zgap, z_s1, z_air, z_s2, z_s2+margin*2.0])
    g.SmoothMeshLines('x',0.5,ratio=1.45); g.SmoothMeshLines('y',0.5,ratio=1.45); g.SmoothMeshLines('z',0.45,ratio=1.45)
    def dedupe(dirn, floor=0.035):
        L=list(g.GetLines(dirn)); keep=[L[0]]
        for v in L[1:-1]:
            if v-keep[-1] >= floor: keep.append(v)
        if L[-1]-keep[-1] >= floor: keep.append(L[-1])
        else: keep[-1]=L[-1]
        g.SetLines(dirn, keep)
    for d in ('x','y','z'): dedupe(d)
    nf2ff = FDTD.CreateNF2FFBox()
    nx=len(g.GetLines('x')); ny=len(g.GetLines('y')); nz=len(g.GetLines('z'))
    P(f"[{tag}] mesh {nx}x{ny}x{nz}={nx*ny*nz/1e6:.2f}M  running FDTD...")
    FDTD.Run(sim, cleanup=True, verbose=0)
    f = np.linspace(18e9,40e9,441); port.CalcPort(sim, f)
    s11 = 20*np.log10(np.abs(port.uf_ref/port.uf_inc)); i0 = np.argmin(np.abs(f-f0))
    # degree-grid NF2FF
    rE = nf2ff.CalcNF2FF(sim, f0, THD, np.array([0.0]),  outfile=f'nfE_{tag}.h5', read_cached=False)  # phi=0 xz (E/steer plane)
    rH = nf2ff.CalcNF2FF(sim, f0, THD, np.array([90.0]), outfile=f'nfH_{tag}.h5', read_cached=False)  # phi=90 yz
    Dmax = 10*np.log10(rE.Dmax[0])
    gE = 10*np.log10(np.maximum(rE.E_norm[0][:,0]**2 / np.max(rE.E_norm[0][:,0]**2), 1e-9) * rE.Dmax[0])
    gH = 10*np.log10(np.maximum(rH.E_norm[0][:,0]**2 / np.max(rH.E_norm[0][:,0]**2), 1e-9) * rH.Dmax[0])
    ipk = int(np.argmax(gE)); pk = THD[ipk]
    above = THD[gE > gE.max()-3.0]
    bw = float(above.max()-above.min()) if above.size>1 else 0.0
    P(f"[{tag}] target={target:+.0f}d  S11@24={s11[i0]:.2f}dB  D@24={Dmax:.2f}dBi  E-peak={pk:+.1f}d  3dBbw={bw:.0f}d")
    return dict(theta=THD, gE=gE, gH=gH, Dmax=float(Dmax), peak=float(pk),
                bw=bw, s11_24=float(s11[i0]), g=np.asarray(gvec,float), target=float(target))


def main():
    mode = sys.argv[1] if len(sys.argv)>1 else 'all'
    out = {}

    if mode in ('validate','all'):
        P("\n=== VALIDATE: uniform (g=1) on directive model, DEGREE grid ===")
        out['uniform'] = run_design(np.ones(8), 'uniform', 0.0)

    if mode in ('grad','all'):
        P("\n=== ROUTE A: physics phase-gradient coding ===")
        for t in (10,20,30):
            g = required_g_for_angle(t)
            P(f"  target {t:+d}d -> g={np.round(g,3)}")
            out[f'grad{t}'] = run_design(g, f'grad{t}', t)

    if mode in ('gradstrong','all'):
        P("\n=== ROUTE A-strong: forced large per-column size gradient (stress test) ===")
        # Impose a strong monotonic size taper across the 8 columns as a coded-aperture
        # phase gradient, ignoring the saturated uc_ref mapping. amp = peak-to-peak fraction.
        for t, amp in [(20, 0.30), (30, 0.50)]:
            sign = -np.sign(np.sin(np.radians(t)))   # taper direction toward +theta
            ramp = np.linspace(-0.5, 0.5, 8)
            g = 1.0 + sign*amp*ramp
            P(f"  target {t:+d}d amp={amp} -> g={np.round(g,3)} contrast={g.max()/g.min():.2f}")
            out[f'gstrong{t}'] = run_design(g, f'gstrong{t}', t)

    if mode in ('surr','all'):
        P("\n=== ROUTE B: surrogate-prescribed coding ===")
        INV = np.load(os.path.join(PAPER, "inverse.npz"))
        MA  = np.load(os.path.join(PAPER, "closure_multiangle.npz")) \
              if os.path.exists(os.path.join(PAPER,"closure_multiangle.npz")) else None
        surr = {'surr_broadside': (INV['g_broadside'], 0.0),
                'surr_steer20':   (INV['g_steer20'], 20.0)}
        if MA is not None:
            if 'steer10_g' in MA.files:   surr['surr_steer10']  = (MA['steer10_g'], 10.0)
            if 'steerm15_g' in MA.files:  surr['surr_steerm15'] = (MA['steerm15_g'], -15.0)
        for tag,(gv,tt) in surr.items():
            out[tag] = run_design(gv, tag, tt)

    # save
    save = {}
    for tag,d in out.items():
        for k,v in d.items():
            save[f"{tag}_{k}"] = np.asarray(v)
    save['col'] = col
    np.savez(os.path.join(PAPER, "closure_directive.npz"), **save)
    P("\nsaved closure_directive.npz")

    # summary table
    P("\n===== CORRECTED DIRECTIVE CLOSURE (E-plane / phi=0 xz) =====")
    P(f"{'design':16s} {'target':>7s} {'peak':>7s} {'3dBbw':>7s} {'D@24':>7s} {'S11@24':>8s}")
    for tag,d in out.items():
        P(f"{tag:16s} {d['target']:+7.0f} {d['peak']:+7.1f} {d['bw']:7.0f} {d['Dmax']:7.2f} {d['s11_24']:8.2f}")

if __name__ == '__main__':
    main()
