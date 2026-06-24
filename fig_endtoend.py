# -*- coding: utf-8 -*-
# fig_endtoend (2 panels) on a solver-validated testbed.
# (a) per-component sign-agreement; the gate drops the seed-unstable components.
# (b) oracle-verified objective after a single-model step vs the ensemble gate step.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; NEU = '#444444'
d = np.load(os.path.join(DIR, 'endtoend.npz'))
sa = d['sign_agree']; trust = d['trust']; tau = float(d['tau']); K = len(sa)
L0 = float(d['L0']); Lg = float(d['L_gate']); Ln = np.atleast_1d(d['L_naive'])
rng = np.random.default_rng(2)

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={'width_ratios': [1.05, 0.95]})

# (a) per-component ensemble sign-agreement; gate drops the seed-unstable ones
cols = [GRN if t else ACC for t in trust]
a.bar(np.arange(K), sa, color=cols, edgecolor='k', lw=0.6, width=0.66)
a.axhline(tau, color='0.4', ls='--', lw=1.0); a.text(K - 0.5, tau + 0.008, f'trust threshold {tau:.1f}', ha='right', va='bottom', fontsize=7.8, color='0.3')
a.set_xticks(np.arange(K)); a.set_xticklabels([f'$g_{{{k}}}$' for k in range(K)])
a.set_ylim(0.45, 1.05); a.set_ylabel('ensemble sign-agreement (10 seeds)'); a.set_xlabel('design-parameter component')
from matplotlib.patches import Patch
a.legend(handles=[Patch(fc=GRN, ec='k', label='trusted (used in step)'), Patch(fc=ACC, ec='k', label='distrusted (dropped)')],
         frameon=False, fontsize=7.8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
a.set_title('(a) gate drops seed-unstable components', loc='left', fontsize=9.3, fontweight='bold')

# (b) oracle-verified objective after the step: ensemble gate vs single-model
a.set_axisbelow(True)
xb = 1 + 0.12 * rng.standard_normal(len(Ln))
b.scatter(xb, Ln, s=42, c=ACC, edgecolors='k', linewidths=0.4, zorder=3, label='naive: single-model step ($\\times$10)')
_m = float(Ln.mean()); b.plot([1 - 0.22, 1 + 0.22], [_m, _m], '-', color='#7d1f17', lw=2.8, zorder=5, solid_capstyle='round')
b.text(1 + 0.27, _m, 'mean', va='center', ha='left', fontsize=7.6, color='#7d1f17')
b.scatter([0], [Lg], s=120, marker='D', c=GRN, edgecolors='k', linewidths=0.6, zorder=4, label='ensemble gate step')
b.axhline(L0, color=NEU, ls=':', lw=1.2); b.text(1.45, L0 * 1.05, 'start', ha='right', va='bottom', fontsize=8, color=NEU)
b.set_yscale('log'); b.set_xlim(-0.5, 1.5); b.set_xticks([0, 1]); b.set_xticklabels(['ensemble\ngate', 'single\nmodel'])
b.set_ylabel('oracle-verified objective $L$ (optimum $=0$)')
_yl = b.get_ylim(); b.set_ylim(_yl[0], _yl[1] * 2.6)   # headroom so the title clears the top point
b.legend(frameon=False, fontsize=7.8, loc='lower left')
b.set_title('(b) gate reliable; single-model erratic', loc='left', fontsize=9.3, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(DIR, 'fig_endtoend.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_endtoend.pdf'), bbox_inches='tight')
print(f'saved fig_endtoend; gate L={Lg:.3f}, naive L={Ln.mean():.3f}+-{Ln.std():.3f} (worst {Ln.max():.3f})')
