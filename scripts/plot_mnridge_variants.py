"""Plot sim_localization/mnridge_variants.csv next to the ChampLasso results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

v = pd.read_csv('sim_localization/mnridge_variants.csv')
r = pd.read_csv('sim_localization/results.csv')
champ = r[(r.solver == 'ChampLasso') & (r.snr == 1.0)].assign(variant='ChampLasso')
sc = pd.read_csv('sim_localization/spatial_components.csv')
sc = sc[sc.variant == 'all L components'].assign(variant='spatial components (paper)')
df = pd.concat([champ[v.columns], v, sc[['scenario','variant','peak_err_mm','disp_mm','near15']]])
variants = ['ChampLasso', 'plain', 'dSPM', 'dSPM+subspace (paper)', 'spatial components (paper)']
colors = ['#1f77b4', '#d95f02', '#e6a15c', '#b35806', '#6a3d9a']
scen = ['heschl-lh', 'heschl-rh', 'shallow', 'mid', 'deep', 'pair-heschl']
metrics = [('peak_err_mm', 'Peak localization error [mm] (lower is better)'),
           ('disp_mm', 'Spatial dispersion [mm] (lower = more focal)'),
           ('near15', 'Energy within 15 mm of truth (higher is better)')]
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
w = 0.8 / len(variants)
for ax, (k, title) in zip(axes, metrics):
    for i, (name, c) in enumerate(zip(variants, colors)):
        s = df[df.variant == name].set_index('scenario')
        ax.bar(np.arange(len(scen)) + (i - len(variants) / 2 + .5) * w, [s[k][x] for x in scen], w, color=c, label=name)
    ax.set_title(title, fontsize=10, loc='left')
axes[0].legend(ncol=3, fontsize=8)
axes[-1].set_xticks(range(len(scen)), scen)
fig.tight_layout()
fig.savefig('sim_localization/mnridge_variants.png', dpi=110)
