"""Volume-source example (examples/00_example-vol-src.py) with two solvers.

Runs the same preprocessing, noise covariance and lead field once, then fits
the oddball NCRFs with ``ChampLasso`` and with ``MNRidge`` and saves matched
plots (same time points, same ``vmax``) for both.

Usage::

    python scripts/compare_vol_src.py [--out DIR]

Requires the Brainstorm auditory dataset (``mne.datasets.brainstorm.bst_auditory``).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import eelbrain
import mne
import numpy as np
import pandas as pd

from ncrf import ChampLasso, MNRidge, fit_ncrf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('vol_src_comparison'))
    parser.add_argument('--own-vmax', action='store_true', help='scale each solver by its own peak instead of a shared vmax')
    parser.add_argument('--only', choices=['ChampLasso', 'MNRidge'], help='plot only this solver')
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Preprocessing (as in the example) --------------------------------------
    data_path = Path(mne.datasets.brainstorm.bst_auditory.data_path())
    bst = data_path / 'MEG' / 'bst_auditory'
    raw = mne.io.read_raw_ctf(bst / 'S01_AEF_20131218_01.ds', preload=False)
    eog_proj = mne.read_proj(bst / 'bst_auditory-eog-proj.fif')
    raw.info['bads'] = ['MLO52-4408', 'MRT51-4408', 'MLO42-4408', 'MLO43-4408']

    annotations_df = pd.read_csv(bst / 'events_bad_01.csv', header=None,
                                 names=['onset', 'duration', 'id', 'label'])
    raw.set_annotations(mne.Annotations(
        annotations_df['onset'].values / raw.info['sfreq'],
        annotations_df['duration'].values / raw.info['sfreq'],
        annotations_df['label'].values,
    ))

    events = mne.find_events(raw, stim_channel='UPPT001')
    sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
    onsets = np.where(np.abs(sound_data) > 2. * np.std(sound_data))[0]
    min_diff = int(0.5 * raw.info['sfreq'])
    diffs = np.concatenate([[min_diff + 1], np.diff(onsets)])
    onsets = onsets[diffs > min_diff]
    assert len(onsets) == len(events)
    sound_events = pd.DataFrame({
        'time': onsets / raw.info['sfreq'],
        'label': pd.Categorical.from_codes(events[:, 2], ['', 'frequent', 'infrequent']),
    })

    raw.add_proj(eog_proj).apply_proj()
    raw.load_data()
    raw.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
    raw.filter(1.0, 20.0, fir_design='firwin')
    raw.crop(sound_events['time'].min() - 0.2, sound_events['time'].max() + 1.0)
    raw.resample(100, npad='auto')
    meg = eelbrain.load.mne.raw_ndvar(raw)

    # --- Predictors --------------------------------------------------------------
    stim1 = eelbrain.NDVar.zeros(meg.time, 'common')
    stim1[sound_events['time'].values] = 1.
    stim2 = stim1.copy('contrast')
    stim2[sound_events['time'].values[sound_events['label'] == 'frequent']] = -1

    # --- Noise covariance --------------------------------------------------------
    raw_er = mne.io.read_raw_ctf(bst / 'S01_Noise_20131218_01.ds', preload=True)
    raw_er.add_proj(eog_proj)
    raw_er.notch_filter(np.arange(60, 181, 60), fir_design='firwin')
    raw_er.filter(1.0, 20.0, fir_design='firwin')
    raw_er.resample(100, npad='auto')
    noise_cov = mne.compute_raw_covariance(raw_er, tmin=0, tmax=None, method='shrunk', rank=None)

    # --- Forward model -----------------------------------------------------------
    subjects_dir = data_path / 'subjects'
    subject = 'bst_auditory'
    trans = bst / 'bst_auditory-trans.fif'
    bem_dir = subjects_dir / subject / 'bem'
    srcfile = bem_dir / 'bst_auditory-vol-10-src.fif'
    if srcfile.is_file():
        src = mne.read_source_spaces(srcfile)
    else:
        src = mne.setup_volume_source_space(
            subject, subjects_dir=subjects_dir, surface=bem_dir / 'inner_skull.surf',
            pos=10.0, add_interpolator=True)
        mne.write_source_spaces(srcfile, src, overwrite=True)

    fwdfile = bem_dir / 'bst_auditory-vol-10-fwd.fif'
    if fwdfile.is_file():
        fwd = mne.read_forward_solution(fwdfile)
    else:
        model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                        meg=True, eeg=False, mindist=5.0, n_jobs=2)
        mne.write_forward_solution(fwdfile, fwd)

    lf = eelbrain.load.mne.forward_operator(fwd, src='vol-10', subjects_dir=subjects_dir, parc='aparc+aseg')

    # --- Fit both solvers --------------------------------------------------------
    SOLVERS = {
        # same settings as the example (fixed mu, shortened iterations)
        'ChampLasso': ChampLasso(mu=4.958456130470556e-06, n_iter=5),
        # beta selected by cross-validation
        'MNRidge': MNRidge(beta='auto'),
    }
    results = {}
    for name, solver in SOLVERS.items():
        cache = bst / f'oddball_ncrf_vol_{name}_v051.pickle'
        if cache.exists():
            results[name] = eelbrain.load.unpickle(cache)
        else:
            print(f'Fitting {name} ...', flush=True)
            results[name] = fit_ncrf(meg, [stim1, stim2], lf, noise_cov, tstart=0, tstop=0.5, solver=solver)
            eelbrain.save.pickle(results[name], cache)
        print(name, results[name], flush=True)

    # --- Morph to fsaverage (once, shared) ---------------------------------------
    mne.datasets.fetch_fsaverage(subjects_dir)
    src_fs = mne.read_source_spaces(subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-vol-5-src.fif')
    morph = mne.compute_source_morph(
        fwd['src'], subject_from=subject, subjects_dir=subjects_dir,
        niter_affine=[10, 10, 5], niter_sdr=[10, 10, 5], src_to=src_fs, verbose=True)


    def morph_to_fsaverage(h):
        data = h.get_data(('source', 'space', 'time'))
        time = h.get_dim('time')
        stc = mne.VolVectorSourceEstimate(data, [fwd['src'][0]['vertno']], time.tmin, time.tstep, subject)
        return eelbrain.load.mne.stc_ndvar(morph.apply(stc), 'fsaverage', 'vol-5')


    # --- Matched plots -----------------------------------------------------------
    # (predictor index, label, peak times); vmax is shared across solvers so the
    # colour scales are comparable.
    PLOTS = [(0, 'common', (0.05, 0.095, 0.170)), (1, 'contrast', (0.190,))]
    for name, result in results.items():
        if args.only and name != args.only:
            continue
        hs = [morph_to_fsaverage(h.smooth('source', 0.01, 'gaussian')) for h in result.model.h]
        vmax = 5e-11
        if args.own_vmax:
            vmax = float(f"{max(float(h.norm('space').max()) for h in hs):.1e}")
            print(f'{name}: using vmax={vmax:.1e}', flush=True)
        for i, label, times in PLOTS:
            bf = eelbrain.plot.Butterfly(hs[i].norm('space'), axtitle=f'{name}: {label}', h=2,
                                         vmax=vmax, ylabel='Amplitude')
            for t in times:
                bf.add_vline(t, color='k', linestyle='--')
            bf.save(OUT / f'{name}_{label}_butterfly.png')
            for t in times:
                gb = eelbrain.plot.GlassBrain(hs[i].sub(time=t), vmax=vmax, display_mode='lr',
                                              title=f'{name} {label} [{t * 1000:.0f} ms]')
                gb.save(OUT / f'{name}_{label}_glass_{t * 1000:.0f}ms.png')
        print(f'{name}: peak |h| common={float(hs[0].norm("space").max()):.3e} '
              f'contrast={float(hs[1].norm("space").max()):.3e}', flush=True)

    print('Saved to', OUT.resolve())


if __name__ == '__main__':
    main()
