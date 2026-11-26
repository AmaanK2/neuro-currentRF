"""MNRidge variants (paper-faithful options) on the localization simulation.

Reuses the scenarios and metrics of simulate_localization.py and compares the
plain MNRidge with dSPM, resolution-subspace truncation and other ``snr`` values.
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd

import simulate_localization as sl
from ncrf import MNRidge, fit_ncrf

VARIANTS = {
    'plain': MNRidge(beta='auto'),
    'dSPM': MNRidge(beta='auto', dspm=True),
    'subspace': MNRidge(beta='auto', subspace=0.999),
    'dSPM+subspace (paper)': MNRidge(beta='auto', dspm=True, subspace=0.999),
    'dSPM+subspace, snr=10': MNRidge(beta='auto', dspm=True, subspace=0.999, snr=10.),
    'dSPM+subspace, snr=30': MNRidge(beta='auto', dspm=True, subspace=0.999, snr=30.),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--snr', type=float, default=1.0)
    parser.add_argument('--duration', type=float, default=60.)
    args = parser.parse_args()
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(args.out)
    scenarios = sl.make_scenarios(lf, 0)
    whitener = sl.whitening_filter_for(lf, cov)
    rows = []
    for (name, sources) in scenarios.items():
        meg, stim, _, trfs = sl.simulate(lf, cov, sources, args.snr, args.duration, 1, whitener)
        for vname, solver in VARIANTS.items():
            fit = fit_ncrf(meg, stim, lf, cov, tstart=sl.TSTART, tstop=sl.TSTOP, solver=solver)
            row = dict(scenario=name, variant=vname, **sl.evaluate(fit.model.h, lf, sources, trfs))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / 'mnridge_variants.csv', index=False)
    print(df.pivot(index='scenario', columns='variant', values='peak_err_mm').round(1).to_string())
    print(df.pivot(index='scenario', columns='variant', values='near15').round(2).to_string())
    print(df.pivot(index='scenario', columns='variant', values='disp_mm').round(1).to_string())


if __name__ == '__main__':
    main()
