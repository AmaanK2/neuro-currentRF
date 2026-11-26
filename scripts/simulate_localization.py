"""Localization accuracy of ChampLasso vs MNRidge on simulated sources.

Sensor data are simulated from the real Brainstorm auditory forward model
(10 mm volume source space) with sources at *known* positions and a known TRF,
plus Gaussian noise drawn from the real empty-room noise covariance. Both
solvers are then fit on the simulated data and the estimated source distribution
is compared with the ground truth.

Why this experiment: sensor-space held-out scores cannot tell a focal estimate
from a smeared one (see scripts/compare_solvers.py), so localization has to be
measured against known positions.

Scenarios
---------
single sources at the left/right transverse temporal gyrus (Heschl; the
realistic auditory case) and at voxels spread over the depth range of the source
space, plus one pair of simultaneous sources (left + right Heschl) with
different TRF latencies.

Metrics (per fit; ``e`` is the per-source energy ``sqrt(sum h**2)`` over
orientation and time)
-------
peak_err_mm   distance between the strongest estimated source(s) and the truth
              (several sources: peaks are found greedily and matched)
disp_mm       spatial dispersion: energy-weighted RMS distance from the nearest
              true source -- how far the estimate is smeared
near15        share of estimated energy within 15 mm of a true source
trf_corr      correlation of the estimated time course at the true voxel (along
              the true orientation) with the true TRF

Usage::

    python scripts/simulate_localization.py --out sim_localization
    python scripts/simulate_localization.py --scenarios heschl-lh --snr 0.5 --quick

Simulation limits: Gaussian noise (real recordings have structured noise and
source correlations), one noise realization per cell, a linear forward model
identical to the one the solvers assume (no forward-model error).
"""
import argparse
import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import eelbrain
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from ncrf import ChampLasso, MNRidge, NCRFEstimator, fit_ncrf

SFREQ = 100.
TSTART, TSTOP = 0., 0.5


def true_trf(t, peak, width=0.025):
    """Biphasic TRF: a positive lobe at ``peak`` and a smaller opposite lobe 100 ms later."""
    return np.exp(-0.5 * ((t - peak) / width) ** 2) - 0.6 * np.exp(-0.5 * ((t - peak - 0.1) / (1.6 * width)) ** 2)


def noise_covariance(cache_dir):
    """Empty-room covariance, preprocessed like examples/00_example-vol-src.py."""
    path = cache_dir / 'empty-room-cov.fif'
    if path.exists():
        return mne.read_cov(path, verbose=False)
    bst = Path(mne.datasets.brainstorm.bst_auditory.data_path(download=False)) / 'MEG' / 'bst_auditory'
    eog_proj = mne.read_proj(bst / 'bst_auditory-eog-proj.fif')
    raw = mne.io.read_raw_ctf(bst / 'S01_Noise_20131218_01.ds', preload=True, verbose=False)
    raw.add_proj(eog_proj)
    raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin', verbose=False)
    raw.filter(1.0, 20.0, fir_design='firwin', verbose=False)
    raw.resample(SFREQ, npad='auto')
    cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, method='shrunk', rank=None, verbose=False)
    mne.write_cov(path, cov, overwrite=True, verbose=False)
    return cov


def load_forward():
    data_path = Path(mne.datasets.brainstorm.bst_auditory.data_path(download=False))
    subjects_dir = data_path / 'subjects'
    fwd = mne.read_forward_solution(subjects_dir / 'bst_auditory' / 'bem' / 'bst_auditory-vol-10-fwd.fif', verbose=False)
    lf = eelbrain.load.mne.forward_operator(fwd, src='vol-10', subjects_dir=subjects_dir, parc='aparc+aseg')
    return lf, fwd['info']['bads']


def make_scenarios(lf, seed):
    """Name -> list of (voxel index, orientation) true sources."""
    rng = np.random.RandomState(seed)
    src = lf.get_dim('source')
    coords = src.coordinates
    parc = np.asarray(src.parc)

    def pick(label):
        idx = np.flatnonzero(parc == label)
        if not len(idx):
            raise ValueError(f"no sources in {label!r}")
        center = coords[idx].mean(0)
        return int(idx[np.argmin(np.linalg.norm(coords[idx] - center, axis=1))])

    def orientation():
        v = rng.randn(3)
        return v / np.linalg.norm(v)

    left, right = pick('ctx-lh-transversetemporal'), pick('ctx-rh-transversetemporal')
    depth = np.linalg.norm(coords - coords.mean(0), axis=1)
    scenarios = {
        'heschl-lh': [(left, orientation())],
        'heschl-rh': [(right, orientation())],
    }
    # voxels spread over the depth range: shallow (near the skull) to deep
    order = np.argsort(depth)
    for name, q in [('shallow', 0.95), ('mid', 0.5), ('deep', 0.08)]:
        scenarios[name] = [(int(order[int(q * (len(order) - 1))]), orientation())]
    scenarios['pair-heschl'] = [(left, orientation()), (right, orientation())]
    return scenarios


def whitening_filter_for(lf, cov):
    """The whitening filter NCRF applies to data on all of the lead field's sensors."""
    estimator = NCRFEstimator.from_lead_field(lf, cov)
    return estimator._forward_for(lf.get_dim('sensor')).whitening_filter


def simulate(lf, cov, sources, snr, duration, seed, whitening_filter):
    """Simulated sensor data (NDVar), stimulus (NDVar) and the true TRFs of each source."""
    rng = np.random.RandomState(seed)
    sensor = lf.get_dim('sensor')
    n = int(duration * SFREQ)
    time_dim = eelbrain.UTS(0, 1 / SFREQ, n)

    # impulse train with jittered inter-stimulus intervals
    onsets = np.cumsum(rng.uniform(0.6, 1.4, size=int(duration / 0.6)))
    onsets = onsets[onsets < duration - TSTOP]
    stim = np.zeros(n)
    stim[np.round(onsets * SFREQ).astype(int)] = 1.
    stim_nd = eelbrain.NDVar(stim, (time_dim,), 'common')

    lags = np.arange(int(round(TSTART * SFREQ)), int(round(TSTOP * SFREQ)) + 1) / SFREQ
    peaks = [0.10, 0.18]  # different latency per source of a multi-source scenario
    signal = 0.
    trfs = []
    for k, (voxel, orientation) in enumerate(sources):
        h = true_trf(lags, peaks[k % 2])
        trfs.append(h)
        response = np.convolve(stim, h)[:n]
        pattern = lf.x[:, voxel, :].astype(np.float64) @ orientation  # (n_sensors,)
        signal = signal + np.outer(pattern, response)

    # noise from the empty-room covariance, restricted to the data's sensors
    names = sensor.names
    index = [cov.ch_names.index(name) for name in names]
    C = cov.data[np.ix_(index, index)]
    w, v = np.linalg.eigh(C)
    w = np.clip(w, 0, None)
    noise = (v * np.sqrt(w)) @ rng.randn(len(names), n)
    # Scale the signal so that, after whitening, signal power / noise power == snr.
    # The whitener must be the one NCRF fits with: the empty-room covariance is
    # nearly rank-deficient, and a whitener that divides by every eigenvalue
    # lets near-null noise directions dominate the signal power, which leaves
    # almost no signal in the directions NCRF actually uses.
    whiten = whitening_filter
    scale = np.sqrt(snr * (whiten @ noise).var() / (whiten @ signal).var())
    meg = eelbrain.NDVar(signal * scale + noise, (sensor, time_dim), 'meg')
    return meg, stim_nd, lags, trfs


def evaluate(h, lf, sources, trfs):
    """Localization metrics for an estimated NCRF ``h`` (source, space, time NDVar)."""
    coords = lf.get_dim('source').coordinates * 1000.  # mm
    x = h.get_data(('source', 'space', 'time'))
    energy = np.sqrt((x ** 2).sum(axis=(1, 2)))
    truth = np.array([coords[v] for v, _ in sources])
    dist_to_truth = np.linalg.norm(coords[:, None, :] - truth[None], axis=2)  # (n_src, n_true)

    # greedy peaks, suppressing 20 mm around each found one
    remaining = energy.copy()
    peaks = []
    for _ in sources:
        p = int(np.argmax(remaining))
        peaks.append(p)
        remaining[np.linalg.norm(coords - coords[p], axis=1) < 20] = -1
    peak_xyz = coords[peaks]
    pair_dist = np.linalg.norm(peak_xyz[:, None, :] - truth[None], axis=2)
    peak_err = min(sum(pair_dist[i, j] for i, j in enumerate(perm))
                   for perm in itertools.permutations(range(len(sources)))) / len(sources)

    e2 = energy ** 2
    nearest = dist_to_truth.min(axis=1)
    disp = np.sqrt((e2 * nearest ** 2).sum() / e2.sum())
    near15 = e2[nearest < 15].sum() / e2.sum()

    corr = []
    n_lags = len(trfs[0])
    for (voxel, orientation), h_true in zip(sources, trfs):
        course = np.tensordot(orientation, x[voxel], axes=(0, 0))  # (time,)
        t0 = int(round(TSTART * SFREQ))
        est = course[t0:t0 + n_lags]
        corr.append(np.corrcoef(est, h_true)[0, 1] if est.std() > 0 else 0.)
    return dict(peak_err_mm=peak_err, disp_mm=disp, near15=near15, trf_corr=float(np.mean(corr)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=Path('sim_localization'))
    parser.add_argument('--scenarios', nargs='*', help='subset of scenario names')
    parser.add_argument('--snr', nargs='*', type=float, default=[0.3, 1.0],
                        help='whitened signal/noise power ratio (default: 0.3 1.0)')
    parser.add_argument('--duration', type=float, default=60., help='seconds of simulated data')
    parser.add_argument('--champ-iter', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick', action='store_true', help='fewer iterations and a short recording (smoke test)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.duration, args.champ_iter = 20., 2

    lf, _ = load_forward()
    cov = noise_covariance(args.out)
    scenarios = make_scenarios(lf, args.seed)
    if args.scenarios:
        scenarios = {k: scenarios[k] for k in args.scenarios}
    coords = lf.get_dim('source').coordinates * 1000.
    for name, sources in scenarios.items():
        print(f'{name}: voxels {[v for v, _ in sources]} at {[tuple(np.round(coords[v]).astype(int)) for v, _ in sources]} mm', flush=True)

    whitener = whitening_filter_for(lf, cov)
    solvers = {'ChampLasso': ChampLasso(mu='auto', n_iter=args.champ_iter), 'MNRidge': MNRidge(beta='auto')}
    rows = []
    for (name, sources), snr in itertools.product(scenarios.items(), args.snr):
        meg, stim, _, trfs = simulate(lf, cov, sources, snr, args.duration, args.seed + 1, whitener)
        for solver_name, solver in solvers.items():
            start = time.time()
            fit = fit_ncrf(meg, stim, lf, cov, tstart=TSTART, tstop=TSTOP, solver=solver)
            seconds = time.time() - start
            row = dict(scenario=name, snr=snr, solver=solver_name, seconds=seconds, **evaluate(fit.model.h, lf, sources, trfs))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(args.out / 'results.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n', df.round(3).to_string(index=False))
    plot(df, args.out / 'localization_summary.png')
    print('saved to', args.out.resolve())


def plot(df, path):
    metrics = [('peak_err_mm', 'Peak localization error [mm]  (lower is better)'),
               ('disp_mm', 'Spatial dispersion [mm]  (lower = more focal)'),
               ('near15', 'Energy within 15 mm of truth  (higher is better)'),
               ('trf_corr', 'TRF time-course correlation  (higher is better)')]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), sharex=True)
    scenarios = list(dict.fromkeys(df['scenario']))
    colors = {'ChampLasso': '#1f77b4', 'MNRidge': '#d95f02'}
    width = 0.8 / (2 * df['snr'].nunique())
    for ax, (key, title) in zip(axes, metrics):
        for i, solver in enumerate(['ChampLasso', 'MNRidge']):
            for j, snr in enumerate(sorted(df['snr'].unique())):
                sub = df[(df.solver == solver) & (df.snr == snr)].set_index('scenario')
                x = np.arange(len(scenarios)) + (i * df['snr'].nunique() + j - df['snr'].nunique() + 0.5) * width
                ax.bar(x, [sub[key].get(s, np.nan) for s in scenarios], width,
                       color=colors[solver], alpha=1.0 if j == 0 else 0.55,
                       label=f'{solver}, SNR {snr:g}')
        ax.set_title(title, fontsize=10, loc='left')
    axes[-1].set_xticks(np.arange(len(scenarios)), scenarios)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


if __name__ == '__main__':
    main()
