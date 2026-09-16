"""Compare ChampLasso and MNRidge under matched conditions.

Both solvers search their own hyperparameter grid by cross-validation on the
same folds, and are then compared on held-out scores only. This is what makes
the comparison fair: cross-validation folds are a deterministic function of the
data and ``n_splits``, so two separate ``fit()`` calls see identical folds, and
each solver is represented by the configuration *it* would have chosen rather
than by a hand-picked one.

Training scores are reported too, but only to show the gap to the held-out
scores; they are not a basis for preferring one solver.

Usage::

    python scripts/compare_solvers.py                 # packaged test data
    python scripts/compare_solvers.py --quick         # fewer candidates/iterations
    python scripts/compare_solvers.py --seconds 60 --n-splits 5
    python scripts/compare_solvers.py --csv out.csv

Caveats this script cannot remove:

* Held-out prediction in *sensor* space does not measure localization accuracy.
  A spatially smeared estimate can predict sensors as well as a focal one. To
  separate the solvers on localization, simulate sources with known positions
  and compare the recovered distributions -- a different experiment.
* The packaged test dataset is a few seconds long and far too short for the
  result to mean anything. Point ``--seconds`` at real data.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import numpy as np

from ncrf import ChampLasso, CrossValidation, MNRidge, NCRFEstimator, RegressionData

#: Held-out scores both solvers report, and the direction that is better.
SHARED_SCORES = {
    'explained_variance': 'higher',
    'l2_error': 'lower',
    'estimation_stability': 'lower',
}

#: Below this, a score is indistinguishable from not fitting anything, and the
#: ranking between two solvers is noise no matter how large the relative gap.
#: An explained variance of 0.001 is 0.1% of the variance.
NEGLIGIBLE = {'explained_variance': 0.01}


@dataclass
class Outcome:
    """Everything recorded for one solver family."""

    name: str
    parameter: str  # the hyperparameter that was searched
    value: float  # the value cross-validation selected
    seconds: float
    cv_results: list
    held_out: dict[str, float]
    training: dict[str, float]
    theta: np.ndarray

    @property
    def density(self) -> float:
        """Fraction of sources above 1% of the peak amplitude."""
        amplitude = np.abs(self.theta).max(axis=1)
        if not amplitude.max():
            return 0.0
        return float(np.mean(amplitude > 0.01 * amplitude.max()))


def _load(name: str):
    """Load one packaged fixture, reusing the extracted dataset when present.

    ``ncrf.tests.fetch.load`` re-fetches the whole 122 MB archive on every call:
    the tarball published for release 0.2 still contains a ``version.txt``
    saying ``0.1``, so ``fetch_dataset``'s freshness check never passes. Four
    fixtures would mean four downloads, and GitHub starts throttling. Read the
    already-extracted pickles directly when they are there.
    """
    import pickle
    from pathlib import Path

    cached = Path(__file__).resolve().parent.parent / 'ncrf-testing-data' / f'{name}.pickled'
    if cached.exists():
        with open(cached, 'rb') as f:
            return pickle.load(f)
    from ncrf.tests.fetch import load
    return load(name)


def load_test_data(seconds: float):
    """Load the packaged regression-test dataset."""
    meg = _load('meg').sub(time=(0, seconds))
    stim = _load('stim').sub(time=(0, seconds))
    return meg, stim, _load('fwd_sol'), _load('emptyroom')


def run(estimator, data, solver, parameter, cv, name):
    """Fit one solver, and record its selection, timing, and both score sets."""
    print(f'--- {name}: searching {parameter} ---', flush=True)
    start = time.time()
    fit = estimator.fit(data, solver, cv=cv)
    seconds = time.time() - start

    cv_results = fit._cv_results or []
    selected = getattr(fit.solver, parameter)
    # the held-out scores of the configuration the search settled on
    held_out = next(
        (result.scores for result in cv_results if result.solver == fit.solver),
        {},
    )
    print(f'    selected {parameter}={selected:g} in {seconds:.1f}s '
          f'from {len(cv_results)} candidates', flush=True)
    return Outcome(name, parameter, float(selected), seconds, cv_results, held_out, fit.scores, fit.solver_fit.theta)


def print_grid(outcome):
    """Every candidate's held-out scores, so the search itself can be checked."""
    print(f'\n{outcome.name} grid (held-out, averaged over folds):')
    header = f'  {outcome.parameter:>12} ' + ' '.join(f'{key:>20}' for key in SHARED_SCORES)
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for result in sorted(outcome.cv_results, key=lambda r: getattr(r.solver, outcome.parameter)):
        value = getattr(result.solver, outcome.parameter)
        marker = '*' if value == outcome.value else ' '
        cells = ' '.join(f'{result.scores.get(key, float("nan")):>20.6f}' for key in SHARED_SCORES)
        print(f'  {value:>12.5g}{marker}{cells}')
    boundary = [getattr(r.solver, outcome.parameter) for r in outcome.cv_results]
    if boundary and outcome.value in (min(boundary), max(boundary)):
        print(f'  WARNING: selected {outcome.parameter} sits on a grid boundary; '
              f'the search range may be too narrow')


def print_comparison(outcomes):
    """The actual head-to-head, on held-out scores."""
    print('\n' + '=' * 78)
    print('HELD-OUT COMPARISON (each solver at its own cross-validated optimum)')
    print('=' * 78)
    width = max(len(o.name) for o in outcomes)
    print(f'{"":<{width}}  ' + ' '.join(f'{key:>20}' for key in SHARED_SCORES))
    for outcome in outcomes:
        cells = ' '.join(f'{outcome.held_out.get(key, float("nan")):>20.6f}' for key in SHARED_SCORES)
        print(f'{outcome.name:<{width}}  {cells}')

    # Each solver picks its winner by its own criterion: MNRidge by a held-out
    # prediction score, ChampLasso by its Bayesian cross-fit likelihood. Comparing
    # the two selections therefore also compares the two selection rules, and
    # favours whichever one optimizes the score being reported. The best value
    # each solver's grid contains is free of that asymmetry.
    print('\nBEST ACHIEVABLE over each grid (removes the selection-criterion asymmetry)')
    print(f'{"":<{width}}  ' + ' '.join(f'{key:>20}' for key in SHARED_SCORES))
    best_of_grid = {}
    for outcome in outcomes:
        cells = []
        for key, direction in SHARED_SCORES.items():
            values = [r.scores[key] for r in outcome.cv_results if key in r.scores]
            best = (max if direction == 'higher' else min)(values) if values else float('nan')
            best_of_grid.setdefault(key, {})[outcome.name] = best
            cells.append(f'{best:>20.6f}')
        print(f'{outcome.name:<{width}}  ' + ' '.join(cells))

    for key, direction in SHARED_SCORES.items():
        by_solver = best_of_grid.get(key, {})
        if len(by_solver) < 2:
            continue
        pick = max if direction == 'higher' else min
        winner = pick(by_solver, key=by_solver.get)
        selected = [o for o in outcomes if key in o.held_out]
        if selected:
            chosen = (max if direction == 'higher' else min)(selected, key=lambda o: o.held_out[key]).name
            if chosen != winner:
                print(f'  NOTE: on {key}, {winner} has the better grid value but '
                      f'{chosen} has the better *selected* value;\n'
                      f'        the two searches are optimizing different criteria.')

    print('\nVerdict per score (as selected):')
    for key, direction in SHARED_SCORES.items():
        scored = [o for o in outcomes if key in o.held_out]
        if len(scored) < 2:
            continue
        pick = max if direction == 'higher' else min
        best = pick(scored, key=lambda o: o.held_out[key])
        other = next(o for o in scored if o is not best)
        margin = abs(best.held_out[key] - other.held_out[key])
        scale = max(abs(best.held_out[key]), abs(other.held_out[key]), 1e-12)
        floor = NEGLIGIBLE.get(key)
        if floor is not None and scale < floor:
            # Both solvers are at the noise floor: a large relative gap between
            # two numbers that are both ~0 says nothing at all.
            note = f'   (both below {floor:g}: neither solver fit anything)'
        elif margin / scale <= 0.05:
            note = '   (difference < 5%: not meaningful)'
        else:
            note = ''
        print(f'  {key:<22} {best.name:<12} better by {margin:.6g}{note}')

    print('\nOther properties (not a ranking):')
    print(f'{"":<{width}}  {"search time":>14} {"selected":>14} {"density":>10} {"train EV":>12}')
    for outcome in outcomes:
        print(f'{outcome.name:<{width}}  {outcome.seconds:>13.1f}s '
              f'{outcome.parameter + "=" + format(outcome.value, ".4g"):>14} '
              f'{outcome.density:>9.1%} {outcome.training.get("explained_variance", float("nan")):>12.6f}')
    print('\n"density" is the fraction of sources above 1% of peak amplitude: a '
          'structural\ndifference between the two algorithms, not a quality '
          'measure. Sensor-space\nprediction cannot tell a focal estimate from a '
          'smeared one.')


def write_csv(path, outcomes):
    """Every candidate of every solver, for plotting elsewhere."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['solver', 'parameter', 'value', 'selected', *SHARED_SCORES])
        for outcome in outcomes:
            for result in outcome.cv_results:
                value = getattr(result.solver, outcome.parameter)
                writer.writerow([
                    outcome.name, outcome.parameter, value, value == outcome.value,
                    *(result.scores.get(key, '') for key in SHARED_SCORES),
                ])
    print(f'\nwrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=5.0, help='seconds of test data to use (default: 5)')
    parser.add_argument('--tstop', type=float, default=0.2, help='TRF length in seconds (default: 0.2)')
    parser.add_argument('--n-splits', type=int, default=3, help='cross-validation folds (default: 3)')
    parser.add_argument('--n-workers', type=int, default=None, help='worker processes (0 runs inline)')
    parser.add_argument('--n-beta', type=int, default=12, help='MNRidge grid points (default: 12)')
    parser.add_argument('--champ-iter', type=int, default=30, help='ChampLasso outer iterations (default: 30)')
    parser.add_argument('--quick', action='store_true', help='small grids and few iterations, for a smoke test')
    parser.add_argument('--csv', help='write per-candidate scores to this file')
    args = parser.parse_args()

    if args.quick:
        args.n_beta, args.champ_iter = 6, 3

    print(f'loading {args.seconds}s of test data...', flush=True)
    meg, stim, lead_field, noise = load_test_data(args.seconds)
    # the fixtures carry a case dimension; RegressionData wants one entry per segment
    meg_trials = list(meg) if meg.has_case else [meg]
    stim_trials = [[s] for s in stim] if stim.has_case else [[stim]]
    data = RegressionData.from_data(meg_trials, stim_trials, tstart=0, tstop=args.tstop, scale='l1', stim_is_single=True)
    estimator = NCRFEstimator.from_lead_field(lead_field, noise)
    cv = CrossValidation(n_splits=args.n_splits, n_workers=args.n_workers)
    print(f'{data!r}\ncross-validation: {args.n_splits} folds\n', flush=True)

    outcomes = [
        run(estimator, data, MNRidge(beta='auto', n_beta=args.n_beta), 'beta', cv, 'MNRidge'),
        run(estimator, data, ChampLasso(mu='auto', n_iter=args.champ_iter), 'mu', cv, 'ChampLasso'),
    ]

    for outcome in outcomes:
        print_grid(outcome)
    print_comparison(outcomes)

    if args.csv:
        write_csv(args.csv, outcomes)


if __name__ == '__main__':
    main()
