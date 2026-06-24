# -*- coding: utf-8 -*-
# fig_extbench (2 panels) for the thin-film transfer-matrix (TMM) benchmark.
# (a) ROC of the sign-agreement gate predicting whether each surrogate gradient component
#     is sign-correct against the TMM gradient, pooled over query points.
# (b) sign-agreement separated by sign-correct vs sign-wrong components, with the trust threshold.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.9})
DIR = os.path.dirname(os.path.abspath(__file__)); GRN = '#1e7a45'; ACC = '#c0392b'; SIG = '#1f5fa6'
d = np.load(os.path.join(DIR, 'extbench_tmm.npz'))
sa = d['all_sa']; corr = d['all_correct']; auc = float(d['auc']); tau = float(d['tau'])
ntr = float(d['n_trust_mean']); K = int(d['K']); cos_ad = float(d['cos_ad'])

fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={'width_ratios': [1.0, 1.05]})

# (a) ROC -- shaded area shows the AUC; the operating point sits at the trust threshold tau
fpr, tpr, thr = roc_curve(corr, sa)
a.fill_between(fpr, tpr, color=SIG, alpha=0.12, zorder=1)
a.plot(fpr, tpr, '-', color=SIG, lw=2.3, zorder=3, solid_capstyle='round')
a.plot([0, 1], [0, 1], '--', color='0.65', lw=1.0, zorder=2)
j = int(np.argmin(np.abs(thr - tau)))
a.plot(fpr[j], tpr[j], 'o', ms=7.5, mfc='white', mec=SIG, mew=1.8, zorder=4)
a.annotate(rf'$\tau={tau}$', xy=(fpr[j], tpr[j]), xytext=(fpr[j] + 0.16, tpr[j] - 0.13),
           fontsize=8.5, color=SIG, va='center',
           arrowprops=dict(arrowstyle='-', color=SIG, lw=0.8))
a.text(0.96, 0.07, f'AUC $=$ {auc:.2f}', ha='right', va='bottom', fontsize=12,
       color=SIG, fontweight='bold')
a.text(0.78, 0.70, 'chance', rotation=45, rotation_mode='anchor',
       color='0.55', fontsize=8, va='bottom', ha='center')
a.set_xlabel('false positive rate'); a.set_ylabel('true positive rate')
a.set_xlim(0, 1); a.set_ylim(0, 1.02)
a.set_xticks([0, 0.5, 1]); a.set_yticks([0, 0.5, 1])
a.set_title('(a) ROC on the external TMM benchmark',
            loc='left', fontsize=9.0, fontweight='bold')

# (b) separation -- trusted band, jittered points, and a coloured mean line per group
b.axhspan(tau, 1.03, color=GRN, alpha=0.06, zorder=0)
b.axhline(tau, color='0.45', ls='--', lw=1.1, zorder=2)
b.text(1.47, tau + 0.008, rf'trust threshold $\tau={tau}$', ha='right', va='bottom',
       fontsize=7.8, color='0.35')
rng = np.random.default_rng(0)
grp = [('sign-correct', sa[corr == 1], GRN), ('sign-wrong', sa[corr == 0], ACC)]
for i, (lab, vals, col) in enumerate(grp):
    if not len(vals):
        continue
    b.scatter(i + 0.085 * rng.standard_normal(len(vals)), vals, s=18, c=col,
              edgecolors='none', alpha=0.5, zorder=3)
    m = float(vals.mean())
    b.plot([i - 0.24, i + 0.24], [m, m], '-', color=col, lw=3.0, zorder=5, solid_capstyle='round')
    b.annotate(f'mean {m:.2f}', xy=(i + 0.24, m), xytext=(i + 0.30, m),
               va='center', ha='left', fontsize=8.3, color=col, fontweight='bold')
b.set_xticks([0, 1]); b.set_xticklabels([g[0] for g in grp]); b.set_xlim(-0.5, 1.55)
b.set_ylabel('ensemble sign-agreement (10 seeds)'); b.set_ylim(0.45, 1.03)
b.text(0.5, 0.035,
       rf'$\tau={tau}$: ${ntr:.1f}/{K}$ trusted $\Rightarrow$ hybrid follows the free gradient '
       rf'(cosine {cos_ad:.3f}) at $<\!1$ solve/grad',
       transform=b.transAxes, fontsize=7.4, va='bottom', ha='center', color='0.35')
b.set_title('(b) reliable vs. unreliable components', loc='left',
            fontsize=9.0, fontweight='bold')

fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(DIR, f'fig_extbench.{ext}'), dpi=320, bbox_inches='tight')
print('wrote fig_extbench.pdf/.png ; AUC', round(auc, 3), 'trusted', round(ntr, 1), '/', K)
