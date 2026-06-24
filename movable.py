# -*- coding: utf-8 -*-
# Sweeps the 24 GHz coded-metasurface FPC antenna for a full-wave-movable inverse-design
# objective. Scaling all 96 PRS cells by a global size factor g shifts the Fabry-Perot
# resonance; tests whether the gain-vs-frequency peak and S11 resonance move even when
# broadside D at 24 GHz is roughly stable.
#
# Directive model (SUB=24, AIRGAP=5.4, PATCH=2.8) from closure_directive.py / openems_fpc.py,
# with NF2FF on a degree grid. For each g in {0.7,0.85,1.0,1.15,1.3}: one FDTD run, broadside
# directivity D(f) over 20..28 GHz (17 pts), |S11|(f) over the same band, and extraction of
# f_peak, peak-D, D@24, S11 resonance freq, -10dB BW, 1dB-gain BW. Then two spatial codings
# (radial taper increasing/decreasing) at the same band.
#
# Usage:
#   python movable.py global   # the g-sweep
#   python movable.py spatial  # 2 radial-taper codings
#   python movable.py all
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
cxc = 0.5*(cells[:,0]+cells[:,2]); cyc = 0.5*(cells[:,1]+cells[:,3])
rad = np.sqrt(cxc**2 + cyc**2)                              # cell radial distance [mm]
rad_n = (rad - rad.min())/(rad.max()-rad.min())            # 0..1 normalized radius

unit = 1e-3; f0, fc = 24e9, 12e9
SUB = 24.0; PATCH = 2.8; AIRGAP = 5.4; margin = 7.0
z_g, z_s1 = 0.0, 1.0; z_air = z_s1 + AIRGAP; z_s2 = z_air + 1.0
epsR, tand = 4.4, 0.02
APFINE = 0.06

# frequency band for the gain/S11 sweep
FBAND = np.linspace(20e9, 28e9, 17)                         # NF2FF gain points
FS11  = np.linspace(18e9, 32e9, 561)                        # dense S11 grid

# the directive FPC model + frequency-band NF2FF
def run_design(gvec96, tag):
    """gvec96: per-cell (len 96) multiplicative size scale. Returns band metrics."""
    sim = rf"D:\openEMS_pkg\sim_mov_{tag}"
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
        s = float(gvec96[i]); cxi=0.5*(x0+x1); cyi=0.5*(y0+y1)
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

    # S11 over a dense grid
    port.CalcPort(sim, FS11)
    s11 = 20*np.log10(np.maximum(np.abs(port.uf_ref/port.uf_inc), 1e-12))
    i24 = np.argmin(np.abs(FS11-f0))
    ires = int(np.argmin(s11)); fres = FS11[ires]; s11res = float(s11[ires])
    # -10 dB BW around the deepest resonance
    below = FS11[s11 < -10.0]
    bw10 = float(below.max()-below.min())/1e9 if below.size>1 else 0.0

    # broadside directivity over the band (theta=0, phi=0)
    Df = np.zeros(len(FBAND))
    for k,fk in enumerate(FBAND):
        r = nf2ff.CalcNF2FF(sim, fk, np.array([0.0]), np.array([0.0]),
                            outfile=f'nf_{tag}_{int(fk/1e9)}.h5', read_cached=False)
        Df[k] = 10*np.log10(r.Dmax[0])
    ipk = int(np.argmax(Df)); fpk = FBAND[ipk]; Dpk = float(Df[ipk])
    # interpolate D@24
    D24 = float(np.interp(f0, FBAND, Df))
    # 1-dB gain bandwidth (band where D >= Dpk-1)
    above = FBAND[Df >= Dpk-1.0]
    bw1d = float(above.max()-above.min())/1e9 if above.size>1 else 0.0

    P(f"[{tag}] f_peak={fpk/1e9:.2f}GHz Dpk={Dpk:.2f}dBi D@24={D24:.2f}dBi "
      f"S11res={s11res:.1f}dB@{fres/1e9:.2f}GHz S11@24={s11[i24]:.1f}dB "
      f"BW-10={bw10:.2f}GHz BW1dG={bw1d:.2f}GHz")
    return dict(Df=Df, s11=s11, fpk=float(fpk), Dpk=Dpk, D24=D24,
                fres=float(fres), s11res=s11res, s11_24=float(s11[i24]),
                bw10=bw10, bw1d=bw1d)


def main():
    mode = sys.argv[1] if len(sys.argv)>1 else 'all'
    G_GLOBAL = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    res = {}

    if mode in ('global','all'):
        P("\n=== GLOBAL PRS cell-scale sweep ===")
        for gg in G_GLOBAL:
            res[f'g{gg:.2f}'] = run_design(np.full(96, gg), f"g{int(gg*100)}")

    if mode in ('spatial','all'):
        P("\n=== SPATIAL coding: radial taper (mean scale held ~1.0) ===")
        amp = 0.30  # peak-to-peak about 1.0
        # increasing with radius: small at center, large at edge
        g_inc = 1.0 + amp*(rad_n - rad_n.mean())
        # decreasing with radius: large at center, small at edge
        g_dec = 1.0 - amp*(rad_n - rad_n.mean())
        res['rad_inc'] = run_design(g_inc, 'radinc')
        res['rad_dec'] = run_design(g_dec, 'raddec')

    # save
    save = dict(FBAND=FBAND, FS11=FS11, G_GLOBAL=G_GLOBAL,
                rad_n=rad_n, cxc=cxc, cyc=cyc)
    for tag,d in res.items():
        for k,v in d.items():
            save[f"{tag}__{k}"] = np.asarray(v)
    np.savez(os.path.join(PAPER, "movable.npz"), **save)
    P("\nsaved movable.npz")

    # summary table
    P("\n===== MOVABLE-OBJECTIVE SWEEP =====")
    P(f"{'design':10s} {'f_peak':>8s} {'peakD':>7s} {'D@24':>7s} {'S11res_f':>9s} {'S11res':>7s} {'BW-10':>7s} {'BW1dG':>7s}")
    for tag,d in res.items():
        P(f"{tag:10s} {d['fpk']/1e9:8.2f} {d['Dpk']:7.2f} {d['D24']:7.2f} "
          f"{d['fres']/1e9:9.2f} {d['s11res']:7.1f} {d['bw10']:7.2f} {d['bw1d']:7.2f}")

if __name__ == '__main__':
    main()
