"""Do snr='auto', depth weighting or sLORETA improve MNRidge's localization? Multi-seed test.

Each seed re-draws the noise, the stimulus and the source orientations.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import simulate_localization as sl
from ncrf import MNRidge, fit_ncrf
from ncrf._solvers import mn_ridge

VARIANTS = {
    'baseline (snr 30)': dict(),
    'snr auto (GCV)': dict(snr='auto'),
    'depth 0': dict(depth=0.0),
    'depth 0.4': dict(depth=0.4),
    'depth 1.0': dict(depth=1.0),
    'dSPM': dict(dspm=True),
    'sLORETA': dict(sloreta=True),
    'sLORETA + snr auto': dict(sloreta=True, snr='auto'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--sim-snr', type=float, default=1.0)
    parser.add_argument('--seeds', type=int, default=4)
    args = parser.parse_args()
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(args.out)
    whitener = sl.whitening_filter_for(lf, cov)

    chosen = []
    original = mn_ridge._gcv_snr
    mn_ridge._gcv_snr = lambda *a, **k: chosen.append(original(*a, **k)) or chosen[-1]  # final fit runs in this process

    rows = []
    for seed in range(args.seeds):
        for name, sources in sl.make_scenarios(lf, seed).items():
            meg, stim, _, trfs = sl.simulate(lf, cov, sources, args.sim_snr, 60., seed + 1, whitener)
            for vname, kwargs in VARIANTS.items():
                chosen.clear()
                fit = fit_ncrf(meg, stim, lf, cov, tstart=sl.TSTART, tstop=sl.TSTOP,
                               solver=MNRidge(beta='auto', n_beta=8, **kwargs))
                row = dict(seed=seed, scenario=name, variant=vname, snr_used=chosen[-1] if chosen else kwargs.get('snr', 30.),
                           **sl.evaluate(fit.model.h, lf, sources, trfs))
                rows.append(row)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(args.out / 'mnridge_improvements.csv', index=False)
    summarize(pd.DataFrame(rows))


def summarize(df):
    g = df.groupby('variant', sort=False)
    out = pd.DataFrame({
        'peak_err_mm': g.peak_err_mm.mean(), 'peak_sem': g.peak_err_mm.sem(),
        'disp_mm': g.disp_mm.mean(), 'near15': g.near15.mean(), 'near15_sem': g.near15.sem(),
    })
    print(out.round(3).to_string())
    print('\nnear15 by scenario')
    print(df.pivot_table(index='scenario', columns='variant', values='near15').round(2)[list(g.groups.keys()) if False else df.variant.unique()].to_string())
    print('\npeak error by scenario')
    print(df.pivot_table(index='scenario', columns='variant', values='peak_err_mm').round(1)[df.variant.unique()].to_string())
    auto = df[df.variant.str.contains('auto')]
    print('\nGCV-chosen snr:', auto.snr_used.round(1).describe()[['min', '50%', 'max']].to_dict())


if __name__ == '__main__':
    main()
