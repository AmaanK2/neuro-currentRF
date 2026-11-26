"""Sanity check of simulate_localization.py: does MNRidge find the source at very high SNR?"""
from pathlib import Path

import numpy as np

import simulate_localization as sl
from ncrf import MNRidge, fit_ncrf


def main():
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(Path(__file__).resolve().parent.parent / 'sim_smoke')
    src = sl.make_scenarios(lf, 0)['heschl-lh']
    coords = lf.get_dim('source').coordinates * 1000
    truth = src[0][0]
    whitener = sl.whitening_filter_for(lf, cov)
    print('true voxel', truth, coords[truth], flush=True)
    for snr in (1.0, 30.0, 1000.0):
        meg, stim, lags, trfs = sl.simulate(lf, cov, src, snr, 60., 1, whitener)
        fit = fit_ncrf(meg, stim, lf, cov, tstart=0, tstop=0.5, solver=MNRidge(beta='auto'))
        x = fit.model.h.get_data(('source', 'space', 'time'))
        e = np.sqrt((x ** 2).sum((1, 2)))
        top = np.argsort(e)[::-1][:5]
        print(f'snr={snr}: beta={fit.solver.beta:.3g} top5 voxels {top.tolist()} '
              f'dist(mm) {np.round(np.linalg.norm(coords[top] - coords[truth], axis=1)).tolist()} '
              f'rank of truth {int((e > e[truth]).sum())}/{len(e)}', flush=True)
        print('   ', {k: round(float(v), 3) for k, v in sl.evaluate(fit.model.h, lf, src, trfs).items()}, flush=True)


if __name__ == '__main__':
    main()
