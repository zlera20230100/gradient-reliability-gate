# -*- coding: utf-8 -*-
# Elsevier graphical abstract for paper 2. landscape banner, single composite.
# three left-to-right stages: ensemble Jacobian -> sign-agreement gate -> hybrid gradient.
# numbers are the honest manuscript figures (external AUC 0.906~0.91, tau=0.9, 63% controlled).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

DIR = os.path.dirname(os.path.abspath(__file__))

# palette matches the paper's figures
NEU = '#444444'
SIG = '#1f5fa6'   # neutral / curve
ACC = '#c0392b'   # flagged / unreliable
GRN = '#1e7a45'   # trusted

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# canvas: ~1340 x 540 px at 100 dpi -> clears the >=531x1328 px floor
fig = plt.figure(figsize=(13.4, 5.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 134); ax.set_ylim(0, 54); ax.axis('off')

# light dividers between the three stages
for xd in (45.5, 90.0):
    ax.plot([xd, xd], [6, 47], color='#dddddd', lw=1.0, zorder=0)

# faint stage banners along the top
def stage_header(x, n, txt, col):
    ax.add_patch(Circle((x, 50.4), 1.35, facecolor=col, edgecolor='none', zorder=4))
    ax.text(x, 50.4, n, ha='center', va='center', fontsize=12, color='white',
            fontweight='bold', zorder=5)
    ax.text(x + 2.6, 50.4, txt, ha='left', va='center', fontsize=12.5,
            color=NEU, fontweight='bold')

stage_header(3.5, '1', 'Deep ensemble of differentiable surrogates', NEU)
stage_header(48.5, '2', 'Solver-free sign-agreement gate', NEU)
stage_header(93.0, '3', 'Cost-optimal hybrid gradient', NEU)

# ============================ STAGE 1 ============================
# M independently retrained surrogates -> a per-component design Jacobian,
# drawn as a small fan of gradient arrows with seed spread.
sx = 6.0
# stacked surrogate cards
for k in range(4):
    off = k * 0.9
    ax.add_patch(FancyBboxPatch((sx + off, 28 - off), 9.5, 7.0,
                 boxstyle='round,pad=0.1,rounding_size=0.4',
                 facecolor='#f5f5f5' if k < 3 else '#eef3f9',
                 edgecolor=SIG if k == 3 else '#cccccc', lw=1.2, zorder=3 + k))
ax.text(sx + 2.7 + 5.0, 31.5, r'PINN$_m$', ha='center', va='center',
        fontsize=13, fontweight='bold', color=NEU, zorder=8)
ax.text(sx + 2.7 + 5.0, 28.0, r'$\Theta_m,\;m=1\dots M$', ha='center', va='center',
        fontsize=9.5, color=SIG, zorder=8)
ax.text(sx + 5.0, 21.3, 'independently retrained\n(M random seeds)',
        ha='center', va='top', fontsize=9.0, color='#666666')

# fan of per-component gradient arrows (ensemble spread)
gx = 24.0; gy = 31.5
rng = np.random.default_rng(7)
angles = np.array([18, 8, -2, -12, 26])      # spread of the ensemble gradient
for j, a in enumerate(angles):
    r = 12.5 + rng.uniform(-0.6, 0.6)
    th = np.deg2rad(a)
    col = GRN if j == 2 else '#9bbf9b'        # central direction in trusted green
    lw = 2.6 if j == 2 else 1.3
    ax.add_patch(FancyArrowPatch((gx, gy), (gx + r * np.cos(th), gy + r * np.sin(th)),
                 arrowstyle='-|>', mutation_scale=15, color=col, lw=lw, zorder=6))
ax.text(gx + 8.0, gy + 9.5, 'design Jacobian', ha='center', va='center',
        fontsize=10.5, color=NEU)
ax.text(gx + 8.0, gy + 7.4, r'$\partial \mathbf{r}/\partial g_k$ (free autodiff)',
        ha='center', va='center', fontsize=10.0, color=NEU)
ax.text(gx + 8.5, gy - 8.0, 'per-component spread\nacross seeds',
        ha='center', va='top', fontsize=8.8, color='#777777', style='italic')

# ============================ STAGE 2 ============================
# sign-agreement gate scores each component; trust threshold tau=0.9.
# two example components: impedance (trusted) vs radiation (flagged).
ax.text(48.0, 43.0, r'sign-agreement score  $a_k\!\in[0,1]$',
        ha='left', va='center', fontsize=10.5, color=NEU)

# horizontal score axis
bx0, bx1 = 49.5, 86.0
by_imp, by_rad = 33.0, 21.5
ax.plot([bx0, bx1], [by_imp - 4.5, by_imp - 4.5], color='#bbbbbb', lw=0.0)  # spacer

def score_bar(y, label, sub, score, color, verdict, vcol):
    ax.text(bx0 - 0.5, y + 2.2, label, ha='left', va='center',
            fontsize=10.5, color=NEU)
    ax.text(bx0 - 0.5, y + 0.2, sub, ha='left', va='center',
            fontsize=8.6, color='#777777', style='italic')
    track_y = y - 2.0
    ax.add_patch(Rectangle((bx0, track_y), bx1 - bx0, 1.6,
                 facecolor='#eeeeee', edgecolor='#cccccc', lw=0.8, zorder=2))
    ax.add_patch(Rectangle((bx0, track_y), (bx1 - bx0) * score, 1.6,
                 facecolor=color, edgecolor='none', zorder=3))
    ax.text(bx1 + 0.7, track_y + 0.8, f'{score:.2f}', ha='left', va='center',
            fontsize=10, color=color, fontweight='bold')
    # verdict tag
    ax.add_patch(FancyBboxPatch((bx0, y - 5.2), 18.0, 2.0,
                 boxstyle='round,pad=0.05,rounding_size=0.3',
                 facecolor=vcol, edgecolor='none', zorder=4))
    ax.text(bx0 + 9.0, y - 4.2, verdict, ha='center', va='center',
            fontsize=8.6, color='white', fontweight='bold', zorder=5)

score_bar(by_imp, 'impedance gradient', 'seed-stable',
          0.97, GRN, 'TRUSTED: free autodiff', GRN)
score_bar(by_rad, 'radiation gradient', 'seed-unstable',
          0.62, ACC, 'FLAGGED: 1 full-wave solve', ACC)

# trust threshold tau = 0.9
tau = 0.9
xt = bx0 + (bx1 - bx0) * tau
ax.plot([xt, xt], [by_rad - 3.0, by_imp + 2.0], color='#222222', ls='--', lw=1.4, zorder=6)
ax.text(xt, by_imp + 3.2, r'$\tau = 0.9$', ha='center', va='bottom',
        fontsize=10.5, color='#222222', fontweight='bold')

# ============================ STAGE 3 ============================
# outcome: cost-optimal hybrid gradient + honest numbers.
hx = 94.5
ax.add_patch(FancyBboxPatch((hx, 27.5), 13.0, 9.0,
             boxstyle='round,pad=0.1,rounding_size=0.5',
             facecolor='#eef3f9', edgecolor=SIG, lw=1.4, zorder=3))
ax.text(hx + 6.5, 33.7, 'hybrid', ha='center', va='center',
        fontsize=12.5, fontweight='bold', color=SIG)
ax.text(hx + 6.5, 31.3, 'gradient', ha='center', va='center',
        fontsize=12.5, fontweight='bold', color=SIG)
ax.text(hx + 6.5, 29.0, r'$\nabla_g \mathcal{L}$', ha='center', va='center',
        fontsize=12, color=NEU)

# two feed-in legend lines: autodiff on trusted, FD only on flagged
ax.add_patch(Rectangle((hx, 23.5), 1.6, 1.4, facecolor=GRN, edgecolor='none'))
ax.text(hx + 2.2, 24.2, 'free autodiff on trusted',
        ha='left', va='center', fontsize=9.2, color=NEU)
ax.add_patch(Rectangle((hx, 20.8), 1.6, 1.4, facecolor=ACC, edgecolor='none'))
ax.text(hx + 2.2, 21.5, 'finite-difference on flagged only',
        ha='left', va='center', fontsize=9.2, color=NEU)

# honest headline numbers
ny = 14.5
ax.add_patch(FancyBboxPatch((hx - 0.5, ny - 7.0), 38.0, 9.6,
             boxstyle='round,pad=0.1,rounding_size=0.4',
             facecolor='#fafafa', edgecolor='#dddddd', lw=1.0, zorder=2))
ax.text(hx + 0.6, ny + 1.0, 'AUC ', ha='left', va='center', fontsize=12, color=NEU)
ax.text(hx + 5.2, ny + 1.0, '0.91', ha='left', va='center',
        fontsize=15, fontweight='bold', color=GRN)
ax.text(hx + 9.6, ny + 1.0, 'predicting the full-wave verdict',
        ha='left', va='center', fontsize=9.6, color=NEU)
ax.text(hx + 9.6, ny - 1.0, 'on an external transfer-matrix benchmark',
        ha='left', va='center', fontsize=8.6, color='#777777', style='italic')

ax.text(hx + 0.6, ny - 4.0, '~63%', ha='left', va='center',
        fontsize=15, fontweight='bold', color=SIG)
ax.text(hx + 7.5, ny - 4.0, 'fewer solves', ha='left', va='center',
        fontsize=9.6, color=NEU)
ax.text(hx + 7.5, ny - 5.9, 'on a controlled spectrum',
        ha='left', va='center', fontsize=8.6, color='#777777', style='italic')

# ============================ stage-to-stage arrows ============================
ax.add_patch(FancyArrowPatch((43.0, 31.5), (48.0, 31.5), arrowstyle='-|>',
             mutation_scale=22, color=NEU, lw=2.2, zorder=7))
ax.add_patch(FancyArrowPatch((88.0, 27.5), (93.5, 31.0), arrowstyle='-|>',
             mutation_scale=22, color=NEU, lw=2.2, zorder=7))

# bottom one-line takeaway
ax.text(67.0, 3.2,
        'A solver-free deep-ensemble gradient-reliability gate: trust free autodiff where seeds agree, verify only where they do not.',
        ha='center', va='center', fontsize=10.0, color=NEU, style='italic')

out_pdf = os.path.join(DIR, 'fig_graphical_p2.pdf')
out_png = os.path.join(DIR, 'fig_graphical_p2.png')
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=110)
plt.close(fig)

# report pixel size of the png
try:
    from PIL import Image
    w, h = Image.open(out_png).size
    print(f'fig_graphical_p2: PNG {w}x{h} px (w x h), PDF + PNG written to {DIR}')
except Exception:
    print(f'fig_graphical_p2: written to {DIR}')
