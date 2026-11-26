"""Why is MNRidge's amplitude much lower than ChampLasso's? Decompose it on simulated data.

Quantities are in theta (basis-coefficient) space, so they are comparable across solvers:
  peak      max over sources of ||theta_s||            (what a glass-brain peak shows)
  oracle    ||theta|| of the least-squares single-source fit at the TRUE voxel
  sensor    ||L theta|| / ||B_ridge||  (how much of the sensor-space TRF the source model reproduces)
"""
import numpy as np

import simulate_localization as sl
from ncrf import ChampLasso, MNRidge, NCRFEstimator, RegressionData, fit_ncrf
from ncrf._solvers.mn_ridge import _pooled_cross_products, _ridge


def main():
    lf, _ = sl.load_forward()
    cov = sl.noise_covariance(__import__('pathlib').Path('sim_localization'))
    sources = sl.make_scenarios(lf, 0)['heschl-lh']
    voxel = sources[0][0]
    meg, stim, _, _ = sl.simulate(lf, cov, sources, 1.0, 60., 1, sl.whitening_filter_for(lf, cov))
    ds = RegressionData.from_data([meg], [[stim]], sl.TSTART, sl.TSTOP, 1, 'spectral', True)
    forward = NCRFEstimator.from_lead_field(lf, cov)._forward_for(ds.sensor_dim)
    data = ds.whiten(forward)
    bE, EtE = _pooled_cross_products(data)
    g = forward.whitened_lead_field  # (rank, n_src * dc)
    dc = forward.dc

    def stats(theta):
        per_source = np.linalg.norm(theta.reshape(-1, dc, theta.shape[1]), axis=(1, 2))
        return per_source.max(), np.sqrt((per_source ** 2).sum()), np.linalg.norm(g @ theta)

    fits = {'ChampLasso': fit_ncrf(meg, stim, lf, cov, tstart=0, tstop=.5, solver=ChampLasso(mu='auto', n_iter=5))}
    for snr in (3., 10., 30., 100.):
        fits[f'MNRidge snr={snr:g}'] = fit_ncrf(meg, stim, lf, cov, tstart=0, tstop=.5, solver=MNRidge(beta='auto', snr=snr))
    beta = fits['MNRidge snr=3'].solver.beta
    b_ridge = _ridge(bE, EtE, beta)
    b_ols = _ridge(bE, EtE, 0.)
    # oracle: best single-source (true voxel, any orientation) explanation of the sensor TRF
    # (along the TRUE orientation: the radial direction is nearly silent to MEG and makes a free fit unstable)
    cols = slice(voxel * dc, voxel * dc + dc)
    direction = g[:, cols] @ sources[0][1]
    coef = (direction @ b_ridge) / (direction @ direction)  # (n_coefficients,)
    theta_oracle = np.zeros((g.shape[1], b_ridge.shape[1]))
    theta_oracle[cols] = np.outer(sources[0][1], coef)
    peak_o, total_o, sens_o = stats(theta_oracle)
    print(f'selected beta = {beta:.4g};  ||B(beta)|| / ||B(0)|| = {np.linalg.norm(b_ridge) / np.linalg.norm(b_ols):.3f}  (ridge shrinkage)')
    print(f'oracle single-source fit at the true voxel: peak {peak_o:.3g}, sensor fraction {sens_o / np.linalg.norm(b_ridge):.3f}\n')
    print(f'{"solver":18s} {"peak":>9s} {"peak/oracle":>12s} {"total":>9s} {"sensor frac":>12s} {"peak/total":>11s}')
    for name, fit in fits.items():
        peak, total, sens = stats(fit.model.theta)
        print(f'{name:18s} {peak:9.3g} {peak / peak_o:12.3f} {total:9.3g} {sens / np.linalg.norm(b_ridge):12.3f} {peak / total:11.3f}')


if __name__ == '__main__':
    main()
