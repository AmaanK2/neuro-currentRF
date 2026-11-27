"""Bar chart of sim_localization/mnridge_improvements.csv (mean +- s.e.m. over seeds x scenarios)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('sim_localization/mnridge_improvements.csv')
order = list(df.variant.unique())
metrics = [('peak_err_mm', 'Peak localization error [mm] (lower is better)'),
           ('disp_mm', 'Spatial dispersion [mm] (lower = more focal)'),
           ('near15', 'Energy within 15 mm of truth (higher is better)')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (k, title) in zip(axes, metrics):
    g = df.groupby('variant')[k]
    m, s = g.mean()[order], g.sem()[order]
    colors = ['#9e9e9e' if o.startswith('baseline') else '#d95f02' for o in order]
    ax.barh(range(len(order)), m, xerr=s, color=colors)
    ax.set_yticks(range(len(order)), order if k == 'peak_err_mm' else [''] * len(order))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, loc='left')
fig.tight_layout()
fig.savefig('sim_localization/mnridge_improvements.png', dpi=110)
