# -*- coding: utf-8 -*-
# fig_reliability (3 panels) comparing the ensemble sign-agreement indicator against MC-dropout.
# (a) sign-agreement by group, with MC-dropout points overlaid.
# (b) reliability-classification AUC with bootstrap 95% CI for three indicators.
# (c) MC-dropout sign-agreement across dropout rates vs the deep ensemble.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'; ORG = '#e67e22'
d = np.load(os.path.join(DIR, 'reliability.npz'))
mc = np.load(os.path.join(DIR, 'mcdropout.npz'))
ci = np.load(os.path.join(DIR, 'reliability_ci.npz'))
thr = float(d['thr']); rng = np.random.default_rng(1)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={'width_ratios': [1.25, 1.0, 1.0]})

# (a) ensemble sign-agreement by group + MC-dropout points
groups = [('responsive\n(reliable)', d['sa_resp'], GRN), ('null\n(unreliable)', d['sa_null'], ACC),
          ('real radiation\n(unreliable)', d['sa_rad'], ACC)]
for i, (lab, vals, col) in enumerate(groups):
    a.scatter(i + 0.10 * rng.standard_normal(len(vals)), vals, s=34, c=col, edgecolors='k', linewidths=0.4, zorder=3, alpha=0.9)
    a.scatter([i], [vals.mean()], marker='_', s=900, c='k', zorder=4)
a.axhline(thr, color='0.4', ls='--', lw=1.0, zorder=1)
a.text(2.45, thr + 0.006, f'trust threshold {thr:.1f}', ha='right', va='bottom', fontsize=7.6, color='0.3')
xm = 2 + 0.10 * rng.standard_normal(len(mc['sa_mc']))
a.scatter(xm, mc['sa_mc'], s=40, marker='^', c=ORG, edgecolors='k', linewidths=0.4, zorder=5, label='MC-dropout (1 model): false trust')
a.legend(loc='center left', bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=7.4)
a.set_xticks(range(3)); a.set_xticklabels([g[0] for g in groups], fontsize=8.4)
a.set_ylabel('ensemble sign-agreement (10 seeds)'); a.set_ylim(0.45, 1.03)
a.set_title('(a) cheap ensemble indicator separates reliable / unreliable', loc='left', fontsize=9.2, fontweight='bold')

# (b) reliability-classification AUC with bootstrap CI
names = ['ensemble\nsign-agree', 'ensemble\nSNR', 'naive\n|gradient|']
keys = ['sign-agreement', 'SNR', 'naive_gradient']
pts = [float(ci[k][0]) for k in keys]; los = [float(ci[k][1]) for k in keys]; his = [float(ci[k][2]) for k in keys]
cols = [GRN, GRN, SIG]
b.bar(range(3), pts, color=cols, edgecolor='k', lw=0.6, width=0.62)
b.errorbar(range(3), pts, yerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
           fmt='none', ecolor='k', elinewidth=1.0, capsize=4, zorder=5)
for i, v in enumerate(pts):
    b.text(i, min(v + 0.02, 1.04), f'{v:.3f}', ha='center', va='bottom', fontsize=8.5)
b.axhline(0.5, color=ACC, ls=':', lw=1.1); b.text(2.45, 0.52, 'chance', ha='right', va='bottom', fontsize=7.8, color=ACC)
b.set_xticks(range(3)); b.set_xticklabels(names, fontsize=8.4)
b.set_ylabel('reliability AUC (95% CI, $n{=}18$)'); b.set_ylim(0.0, 1.12)
b.set_title('(b) predicts the full-wave verdict, no solver', loc='left', fontsize=9.2, fontweight='bold')

# (c) MC-dropout sign-agreement across dropout rates
ps = [0.05, 0.10, 0.20, 0.50]; sa_sweep = []
for p in ps:
    dd = np.load(os.path.join(DIR, f'mcdropout_p{p:.2f}.npz')); sa_sweep.append(float(dd['sa_mc'].mean()))
c.plot(ps, sa_sweep, 'o-', color=ORG, lw=1.9, ms=7, label='MC-dropout (1 model)')
c.axhline(float(d['sa_rad'].mean()), color=GRN, ls='-', lw=1.8, label='deep ensemble (10 seeds)')
c.axhline(thr, color='0.4', ls='--', lw=1.0); c.text(0.5, thr - 0.045, 'trust threshold', ha='right', fontsize=7.6, color='0.3')
c.set_xscale('log'); c.set_xticks(ps); c.set_xticklabels([f'{p:g}' for p in ps])
c.set_xlabel('MC-dropout rate $p$'); c.set_ylabel('radiation-gradient sign-agreement'); c.set_ylim(0.45, 1.05)
c.legend(loc='center left', frameon=False, fontsize=8.0)
c.set_title('(c) dropout is over-confident at every rate', loc='left', fontsize=9.2, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(DIR, 'fig_reliability.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(DIR, 'fig_reliability.pdf'), bbox_inches='tight')
print('saved fig_reliability (3-panel); sweep sign-agree =', [round(s, 2) for s in sa_sweep])
