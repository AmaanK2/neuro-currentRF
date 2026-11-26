"""Paper-style spatial components on the localization simulation.

For each scenario, fits the Donhauser & Baillet spatial-component pipeline
(dSPM resolution subspace, ridge, GEP spatial filters, bootstrap hyperparameters)
and measures how well the component source maps localize the simulated sources.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import simulate_localization as sl
from ncrf import NCRFEstimator, RegressionData, spatial_components


def map_metrics(energy, lf, sources):
    """Peak error, dispersion and energy within 15 mm for a per-source magnitude map."""
    coords = lf.get_dim('source').coordinates * 1000.
    truth = np.array([coords[v] for v, _ in sources])
    remaining = energy.copy()
    peaks = []
    for _ in sources:
        p = int(np.argmax(remaining))
        peaks.append(p)
        remaining[np.linalg.norm(coords - coords[p], axis=1) < 20] = -1
    d = np.linalg.norm(coords[peaks][:, None] - truth[None], axis=2)
    peak_err = min(sum(d[i, j] for i, j in enumerate(perm))
                   for perm in itertools.permutations(range(len(sources)))) / len(sources)
    nearest = np.linalg.norm(coords[:, None] - truth[None], axis=2).min(1)
    e2 = energy ** 2
    return dict(peak_err_mm=peak_err, disp_mm=float(np.sqrt((e2 * nearest ** 2).sum() / e2.sum())),
                near15=float(e2[nearest < 15].sum() / e2.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--snr', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--n-boot', type=int, default=15, help='bootstrap repetitions (paper: 15)')
    parser.add_argument('--scenarios', nargs='*')
    args = parser.parse_args()
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(args.out)
    scenarios = sl.make_scenarios(lf, 0)
    if args.scenarios:
        scenarios = {k: scenarios[k] for k in args.scenarios}
    whitener = sl.whitening_filter_for(lf, cov)
    estimator = NCRFEstimator.from_lead_field(lf, cov)
    rows = []
    for name, sources in scenarios.items():
        meg, stim, _, _ = sl.simulate(lf, cov, sources, args.snr, args.duration, 1, whitener)
        ds = RegressionData.from_data([meg], [[stim]], sl.TSTART, sl.TSTOP, 1, 'spectral', True)
        forward = estimator._forward_for(ds.sensor_dim)
        data = ds.whiten(forward)
        t0 = time.time()
        sc = spatial_components(data, forward, n_boot=args.n_boot)
        sec = time.time() - t0
        weights = np.clip(sc.correlations[:sc.n_components], 0, None) ** 2
        pooled = np.sqrt((sc.source_maps ** 2 * weights).sum(1))
        for label, energy in [('component 1', sc.source_maps[:, 0]), (f'all L components', pooled)]:
            row = dict(scenario=name, variant=label, L=sc.n_components, r1=float(sc.correlations[0]),
                       alpha=sc.alpha, beta=sc.beta, seconds=sec, **map_metrics(energy, lf, sources))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(args.out / 'spatial_components.csv', index=False)


if __name__ == '__main__':
    main()
