"""Effect of MNRidge's assumed ``snr`` (inverse-operator regularization) on localization."""
import argparse
from pathlib import Path

import pandas as pd

import simulate_localization as sl
from ncrf import MNRidge, fit_ncrf

SNRS = (1., 3., 10., 30., 100., 300.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--sim-snr', type=float, default=1.0)
    args = parser.parse_args()
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(args.out)
    whitener = sl.whitening_filter_for(lf, cov)
    rows = []
    for name, sources in sl.make_scenarios(lf, 0).items():
        meg, stim, _, trfs = sl.simulate(lf, cov, sources, args.sim_snr, 60., 1, whitener)
        for dspm in (False, True):
            for snr in SNRS:
                fit = fit_ncrf(meg, stim, lf, cov, tstart=sl.TSTART, tstop=sl.TSTOP,
                               solver=MNRidge(beta='auto', snr=snr, dspm=dspm))
                rows.append(dict(scenario=name, dspm=dspm, snr=snr, **sl.evaluate(fit.model.h, lf, sources, trfs)))
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / 'mnridge_snr.csv', index=False)
    for dspm in (False, True):
        sub = df[df.dspm == dspm]
        print(f'\n=== dspm={dspm}: mean over scenarios ===')
        print(sub.groupby('snr')[['peak_err_mm', 'disp_mm', 'near15']].mean().round(2).to_string())
        print('peak error per scenario'); print(sub.pivot(index='scenario', columns='snr', values='peak_err_mm').round(1).to_string())
        print('energy within 15 mm'); print(sub.pivot(index='scenario', columns='snr', values='near15').round(2).to_string())


if __name__ == '__main__':
    main()
