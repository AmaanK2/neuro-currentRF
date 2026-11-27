"""Example estimated maps next to the true source, for ChampLasso and MNRidge.

Each panel is a top-down (axial) max-projection of the estimated source strength,
normalized to that map's own maximum. Green star = true source, dashed circle =
15 mm around it, white x = strongest estimated voxel.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import simulate_localization as sl
from ncrf import ChampLasso, MNRidge, fit_ncrf

SCENARIOS = ['heschl-lh', 'deep', 'pair-heschl']


def projection(coords, energy, step=10.):
    """Max-projection over z onto a regular (x, y) grid."""
    ix = np.round((coords[:, 0] - coords[:, 0].min()) / step).astype(int)
    iy = np.round((coords[:, 1] - coords[:, 1].min()) / step).astype(int)
    grid = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    for a, b, e in zip(ix, iy, energy):
        if np.isnan(grid[b, a]) or e > grid[b, a]:
            grid[b, a] = e
    extent = (coords[:, 0].min() - step / 2, coords[:, 0].max() + step / 2,
              coords[:, 1].min() - step / 2, coords[:, 1].max() + step / 2)
    return grid, extent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--sim-snr', type=float, default=1.0)
    args = parser.parse_args()
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(args.out)
    whitener = sl.whitening_filter_for(lf, cov)
    coords = lf.get_dim('source').coordinates * 1000.
    scenarios = sl.make_scenarios(lf, 0)
    solvers = {'ChampLasso': ChampLasso(mu='auto', n_iter=5), 'MNRidge': MNRidge(beta='auto')}

    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(8, 3.9 * len(SCENARIOS)))
    for r, name in enumerate(SCENARIOS):
        sources = scenarios[name]
        meg, stim, _, trfs = sl.simulate(lf, cov, sources, args.sim_snr, 60., 1, whitener)
        for c, (solver_name, solver) in enumerate(solvers.items()):
            fit = fit_ncrf(meg, stim, lf, cov, tstart=sl.TSTART, tstop=sl.TSTOP, solver=solver)
            h = fit.model.h
            x = h.get_data(('source', 'space', 'time'))
            energy = np.sqrt((x ** 2).sum(axis=(1, 2)))
            metrics = sl.evaluate(h, lf, sources, trfs)
            grid, extent = projection(coords, energy / energy.max())
            ax = axes[r, c]
            ax.imshow(grid, origin='lower', extent=extent, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
            ax.set_facecolor('#1a1a1a')
            peak = coords[int(np.argmax(energy))]
            ax.plot(*peak[:2], 'x', color='white', ms=9, mew=2)
            for voxel, _ in sources:
                t = coords[voxel]
                ax.plot(*t[:2], '*', color='#00e676', ms=15, mec='k')
                ax.add_patch(plt.Circle(t[:2], 15, fill=False, ls='--', ec='#00e676', lw=1.3))
            ax.set_title(f'{solver_name}\npeak error {metrics["peak_err_mm"]:.0f} mm\n'
                         f'{100 * metrics["near15"]:.0f}% of energy within 15 mm', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel({'heschl-lh': 'one source (auditory cortex)', 'deep': 'one deep source',
                               'pair-heschl': 'two sources (left + right)'}[name], fontsize=10)
        print(name, 'done', flush=True)
    fig.suptitle('Estimated source maps, top-down view\nGreen star = true source, dashed circle = 15 mm, white x = strongest estimated spot',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / 'example_maps.png', dpi=120)
    print('saved')


if __name__ == '__main__':
    main()
