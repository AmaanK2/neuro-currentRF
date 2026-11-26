"""Compose ChampLasso (shared vmax) and MNRidge (own vmax) plots side by side."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path('.')
COLS = [('ChampLasso', ROOT / 'vol_src_comparison', 'vmax = 5e-11'),
        ('MNRidge', ROOT / 'vol_src_v051_snr30', 'snr=30, own vmax = 1.5e-11')]
ROWS = [('common_butterfly', 'Common: all sources'),
        ('common_glass_50ms', 'Common, 50 ms'),
        ('common_glass_95ms', 'Common, 95 ms'),
        ('common_glass_170ms', 'Common, 170 ms'),
        ('contrast_butterfly', 'Contrast: all sources'),
        ('contrast_glass_190ms', 'Contrast, 190 ms')]

fig, axes = plt.subplots(len(ROWS), 2, figsize=(10, 2.3 * len(ROWS)))
for c, (solver, folder, scale) in enumerate(COLS):
    for r, (key, label) in enumerate(ROWS):
        ax = axes[r, c]
        ax.imshow(mpimg.imread(folder / f'{solver}_{key}.png'))
        ax.axis('off')
        if r == 0:
            ax.set_title(f'{solver}\n({scale})', fontsize=12, fontweight='bold')
        if c == 0:
            ax.text(-0.02, 0.5, label, transform=ax.transAxes, ha='right', va='center', fontsize=9)
fig.tight_layout()
fig.savefig('vol_src_side_by_side_snr30.png', dpi=150, bbox_inches='tight')
print('saved')
