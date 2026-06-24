# -*- coding: utf-8 -*-
# fig_hybrid (2 panels) for the reliability-gated hybrid design gradient.
# (a) controlled spectrum: cost-accuracy frontier. the gate spends one full-wave finite-difference (FD)
#     solve on flagged components and uses the autodiff estimate elsewhere; operating point vs the oracle.
# (b) real 24-GHz FPC: per-zone sign-agreement; the gate flags all six radiation zones for FD, while the
#     impedance direction is taken from autodiff.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'
d = np.load(os.path.join(DIR, 'hybrid_gradient.npz'))
K = int(d['K']); tau = float(d['tau'])

fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={'width_ratios': [1.12, 1.0]})

# (a) cost-accuracy frontier
c = d['cost_curve']; e = d['err_curve']
# keep distinct (cost,err) points along the frontier
keep = np.concatenate([[True], np.abs(np.diff(c)) > 1e-9])
a.plot(c[keep], e[keep], '-o', color=SIG, ms=5, lw=1.6, zorder=3, label='gate frontier (sweep $\\tau$)')
# endpoints and references
a.scatter([0.0], [float(d['err_allad'])], marker='X', s=110, c=ACC, edgecolors='k', linewidths=0.5, zorder=5,
          label='all-autodiff (0 solves, wrong)')
a.scatter([K], [0.0], marker='s', s=70, c='0.25', edgecolors='k', linewidths=0.5, zorder=5,
          label=f'all-FD ({K} solves, exact)')
a.scatter([float(d['cost_oracle'])], [float(d['err_oracle'])], marker='*', s=190, c=GRN, edgecolors='k',
          linewidths=0.5, zorder=6, label='oracle hybrid (best subset)')
a.scatter([float(d['op_cost'])], [float(d['op_err'])], marker='o', s=95, facecolors='none', edgecolors=ORG,
          linewidths=2.0, zorder=7, label=f'gate @ $\\tau={tau}$')
a.annotate(f"{100*(1-float(d['op_cost'])/K):.0f}% fewer solves\nthan all-FD,\n"
           f"{float(d['err_allad'])/float(d['op_err']):.1f}$\\times$ lower error\nthan all-autodiff",
           xy=(float(d['op_cost']), float(d['op_err'])), xytext=(2.7, 0.34), fontsize=8.2, color='0.2',
           arrowprops=dict(arrowstyle='->', color='0.45', lw=1.0))
a.set_xlabel('full-wave FD solves spent (of %d components)' % K)
a.set_ylabel('relative error of the design gradient')
a.set_xlim(-0.3, K + 0.3); a.set_ylim(-0.03, 0.58)
a.legend(loc='upper right', frameon=False, fontsize=7.9)
a.set_title('(a) the gate buys near-exact gradients cheaply', loc='left', fontsize=9.4, fontweight='bold')

# (b) real device, per-zone sign-agreement
sa = d['dev_sign_agree']; trust = d['dev_trust']
x = np.arange(1, K + 1)
cols = [GRN if t else ACC for t in trust]
b.bar(x, sa, color=cols, edgecolor='k', linewidth=0.5, width=0.66, zorder=3)
b.axhline(tau, color='0.4', ls='--', lw=1.1, zorder=2)
b.text(K + 0.5, tau + 0.008, f'trust threshold {tau}', ha='right', va='bottom', fontsize=7.8, color='0.3')
b.set_xticks(x); b.set_xticklabels([f'z{i}' for i in x])
b.set_xlabel('radiation zone (real 24-GHz FPC)')
b.set_ylabel('ensemble sign-agreement (10 seeds)')
b.set_ylim(0.45, 1.12)
b.text(0.03, 0.93,
       f"all {K}/{K} zones < threshold $\\Rightarrow$ FD on all\n"
       f"naive autodiff shortcut: only {int(d['dev_signmatch'])}/{K} signs correct,\n"
       f"{float(d['dev_naive_relerr']):.1f}$\\times$ the full-wave magnitude\n"
       f"(impedance direction certified free, 0 solves)",
       transform=b.transAxes, fontsize=8.0, va='top', ha='left', color='0.2',
       bbox=dict(boxstyle='round,pad=0.4', fc='#f7f2ea', ec='0.7', lw=0.7))
b.set_title('(b) on an inert real device the gate refuses the shortcut', loc='left', fontsize=9.4, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_hybrid_gradient.{ext}'), dpi=320, bbox_inches='tight')
print('wrote fig_hybrid_gradient.pdf/.png')
