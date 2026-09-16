"""Tests for the minimum-norm/ridge solver."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import linalg

from ncrf import ForwardModel, MNRidge, MNRidgeFit, RegressionData
from ncrf._metrics import merge_scores
from ncrf._solvers import mn_ridge
from ncrf._solvers.mn_ridge import _inverse_operator, _ridge, select_by_criterion


def _forward(n_sensors=6, n_sources=3, dc=1, seed=0):
    """A small forward model; ``source``/``sensor`` only need a length here."""
    rng = np.random.RandomState(seed)
    source = MagicMock()
    source.__len__.return_value = n_sources
    sensor = MagicMock()
    sensor.__len__.return_value = n_sensors
    if dc == 1:
        space = None
    else:
        space = MagicMock()
        space.__len__.return_value = dc
    return ForwardModel(
        lead_field=rng.randn(n_sensors, n_sources * dc),
        noise_covariance=np.eye(n_sensors),
        source=source,
        sensor=sensor,
        space=space,
    )


def _data(meg, covariates):
    """A dataset carrying just what the solver reads: ``meg`` and ``covariates``."""
    return RegressionData(
        meg=list(meg),
        covariates=list(covariates),
        design=MagicMock(),
        sensor_dim=MagicMock(),
        whitener=np.eye(meg[0].shape[0]),
    )


def _random_data(forward, n_times=40, n_coefficients=5, n_segments=2, seed=1):
    rng = np.random.RandomState(seed)
    n_sensors = forward.lead_field.shape[0]
    meg = [rng.randn(n_sensors, n_times) for _ in range(n_segments)]
    covariates = [rng.randn(n_times, n_coefficients) for _ in range(n_segments)]
    return _data(meg, covariates)


@pytest.mark.parametrize(
    'beta, expected',
    [
        (0.1, (0.1,)),
        (1, (1.0,)),
        ([0.1], (0.1,)),
        ((0.1, 0.2), (0.1, 0.2)),
        (np.array([0.1, 0.2]), (0.1, 0.2)),
    ],
)
def test_candidates(beta, expected):
    candidates = MNRidge(beta).candidates(None, None)
    assert tuple(candidate.beta for candidate in candidates) == expected


@pytest.mark.parametrize('beta', [[], 'invalid', [0.1, 'invalid'], -1])
def test_candidates_invalid(beta):
    with pytest.raises((TypeError, ValueError)):
        MNRidge(beta).candidates(None, None)


def test_auto_candidates_scales_with_the_design():
    """The grid is anchored on the design, so rescaling covariates rescales beta."""
    forward = _forward()
    data = _random_data(forward)
    scaled = _data(data.meg, [covariate * 10 for covariate in data.covariates])

    betas = np.array([candidate.beta for candidate in MNRidge().auto_candidates(forward, data)])
    scaled_betas = np.array([candidate.beta for candidate in MNRidge().auto_candidates(forward, scaled)])

    assert len(betas) == MNRidge().n_beta
    assert (np.diff(betas) > 0).all()
    np.testing.assert_allclose(scaled_betas, betas * 10, rtol=1e-10)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError, match='criterion='):
        MNRidge(criterion='bogus')
    with pytest.raises(ValueError, match='n_beta='):
        MNRidge(n_beta=0)


def test_solve_requires_fixed_beta():
    with pytest.raises(ValueError, match='requires a fixed numeric beta'):
        MNRidge('auto').solve(_forward(), None)


def test_ridge_matches_the_svd_formula():
    """``_ridge`` has to agree with equations 16-18 of Donhauser & Baillet (2020)."""
    rng = np.random.RandomState(2)
    E = rng.randn(40, 5)  # design matrix ("M" in the paper)
    X = rng.randn(40, 6)  # data, time by sensor
    beta = 0.7

    u, s, vh = linalg.svd(E, full_matrices=False)
    d = s / (s ** 2 + beta ** 2)  # equation 17
    expected = vh.T @ np.diag(d) @ u.T @ X  # equation 18, (coefficient, sensor)

    B = _ridge(X.T @ E, E.T @ E, beta)

    np.testing.assert_allclose(B, expected.T, rtol=1e-8, atol=1e-10)


def test_ridge_unregularized_is_the_pseudo_inverse():
    """beta=0 on a rank-deficient design must not blow up."""
    rng = np.random.RandomState(3)
    E = rng.randn(40, 4)
    E = np.hstack([E, E[:, :1]])  # exactly collinear column
    X = rng.randn(40, 6)

    B = _ridge(X.T @ E, E.T @ E, 0.0)

    assert np.isfinite(B).all()
    # the minimum-norm solution still reproduces the fitted values
    np.testing.assert_allclose(B @ E.T, (linalg.pinv(E) @ X).T @ E.T, rtol=1e-6, atol=1e-8)


def test_ridge_shrinks_towards_zero():
    forward = _forward()
    data = _random_data(forward)
    bE, EtE = mn_ridge._pooled_cross_products(data)

    norms = [np.linalg.norm(_ridge(bE, EtE, beta)) for beta in (0.0, 1.0, 10.0, 100.0)]

    assert norms == sorted(norms, reverse=True)
    assert norms[-1] < norms[0]


def test_inverse_operator_matches_the_minimum_norm_formula():
    forward = _forward()
    initializer = forward.mne_initializer
    w, s = initializer.w, initializer.s
    snr = 3.0
    lambda2 = (s[0] / snr) ** 2
    L = forward.whitened_lead_field
    R = np.diag(w ** 2)  # source covariance implied by the depth weighting
    expected = R @ L.T @ linalg.inv(L @ R @ L.T + lambda2 * np.eye(L.shape[0]))

    k = _inverse_operator(forward, snr, dspm=False)

    np.testing.assert_allclose(k, expected, rtol=1e-8, atol=1e-10)


def test_dspm_normalizes_noise_sensitivity():
    """With whitened data every dSPM source has unit projected noise variance."""
    forward = _forward()

    k = _inverse_operator(forward, snr=3.0, dspm=True)

    np.testing.assert_allclose((k ** 2).sum(axis=1), 1.0, rtol=1e-8)


def test_dspm_preserves_orientation_ratios():
    """Free-orientation sources are normalized per source, not per component."""
    forward = _forward(n_sensors=9, n_sources=3, dc=3)

    plain = _inverse_operator(forward, snr=3.0, dspm=False)
    normalized = _inverse_operator(forward, snr=3.0, dspm=True)

    ratio = np.linalg.norm(normalized, axis=1) / np.linalg.norm(plain, axis=1)
    for source in range(3):
        block = ratio[forward.source_block(source)]
        np.testing.assert_allclose(block, block[0], rtol=1e-8)


def test_inverse_operator_rejects_invalid_snr():
    with pytest.raises(ValueError, match='snr='):
        _inverse_operator(_forward(), snr=0.0, dspm=False)


def test_inverse_operator_is_cached_per_configuration():
    forward = _forward()

    first = mn_ridge._operator(forward, 3.0, True)

    assert mn_ridge._operator(forward, 3.0, True) is first
    assert mn_ridge._operator(forward, 3.0, False) is not first
    assert mn_ridge._operator(forward, 2.0, True) is not first


def test_solve_recovers_a_noiseless_trf():
    """An overdetermined, noise-free problem has to come back exactly."""
    forward = _forward(n_sensors=8, n_sources=3)
    rng = np.random.RandomState(4)
    theta = rng.randn(3, 5)
    covariates = rng.randn(60, 5)
    meg = forward.whitened_lead_field @ theta @ covariates.T
    data = _data([meg], [covariates])

    # snr -> large and beta -> 0 make the solver a plain least-squares inverse
    fit = MNRidge(beta=0.0, snr=1e8, dspm=False).solve(forward, data)

    assert isinstance(fit, MNRidgeFit)
    assert fit.theta.shape == theta.shape
    np.testing.assert_allclose(fit.theta, theta, rtol=1e-5, atol=1e-7)


def test_default_preserves_the_generative_model():
    """The default fit has to reproduce the data it was fit on.

    dSPM rescales each source by its noise sensitivity, which breaks
    ``meg == lead_field @ theta @ covariates.T`` and makes every downstream
    metric meaningless, so it must not be the default.
    """
    assert MNRidge().dspm is False

    forward = _forward(n_sensors=8, n_sources=3)
    rng = np.random.RandomState(6)
    theta = rng.randn(3, 5)
    covariates = rng.randn(60, 5)
    meg = forward.whitened_lead_field @ theta @ covariates.T
    data = _data([meg], [covariates])

    plain = MNRidge(beta=0.0, snr=1e8).solve(forward, data)
    normalized = MNRidge(beta=0.0, snr=1e8, dspm=True).solve(forward, data)

    def residual(fit):
        prediction = forward.whitened_lead_field @ fit.theta @ covariates.T
        return np.linalg.norm(meg - prediction) / np.linalg.norm(meg)

    np.testing.assert_allclose(residual(plain), 0.0, atol=1e-6)
    assert residual(normalized) > 0.1


def test_solve_pools_segments():
    """Two identical segments have to give the same answer as one of them."""
    forward = _forward()
    rng = np.random.RandomState(5)
    meg = rng.randn(6, 30)
    covariates = rng.randn(30, 4)
    solver = MNRidge(beta=0.5)

    one = solver.solve(forward, _data([meg], [covariates]))
    two = solver.solve(forward, _data([meg, meg], [covariates, covariates]))

    np.testing.assert_allclose(one.theta, two.theta, rtol=1e-8, atol=1e-10)


def test_solve_is_linear_in_the_data():
    """Both stages are linear, so scaling the data scales the estimate."""
    forward = _forward()
    data = _random_data(forward)
    scaled = _data([meg * 3 for meg in data.meg], data.covariates)
    solver = MNRidge(beta=0.5)

    np.testing.assert_allclose(
        solver.solve(forward, scaled).theta,
        solver.solve(forward, data).theta * 3,
        rtol=1e-8,
    )


def test_fit_scores_do_not_shadow_model_metrics():
    forward = _forward()
    data = _random_data(forward)
    fit = MNRidge(beta=0.5).solve(forward, data)

    scores = fit.score(forward, data)

    assert set(scores) == {'ridge_penalty', 'ridge_objective'}
    assert scores['ridge_penalty'] > 0
    # merge_scores raises if a solver score collides with a model metric
    merge_scores({'explained_variance': 0.0, 'l2_error': 0.0}, scores)


def test_fit_penalty_is_zero_without_regularization():
    forward = _forward()
    data = _random_data(forward)

    fit = MNRidge(beta=0.0).solve(forward, data)

    assert fit.score(forward, data)['ridge_penalty'] == 0.0


def _cv_result(beta, *, l2_error=0.0, explained_variance=0.0):
    from ncrf._crossvalidation import CVResult
    return CVResult(MNRidge(beta=beta), {
        'l2_error': l2_error,
        'explained_variance': explained_variance,
        'estimation_stability': 0.0,
        'ridge_penalty': 0.0,
        'ridge_objective': 0.0,
    })


def test_select_by_criterion():
    results = [
        _cv_result(0.1, l2_error=3.0, explained_variance=0.1),
        _cv_result(1.0, l2_error=1.0, explained_variance=0.2),
        _cv_result(10.0, l2_error=2.0, explained_variance=0.9),
    ]

    assert select_by_criterion(results, 'l2').beta == 1.0
    assert select_by_criterion(results, 'explained-variance').beta == 10.0
    with pytest.raises(ValueError, match='criterion='):
        select_by_criterion(results, 'bogus')


@pytest.mark.parametrize(
    'best, expected_direction',
    [(0.1, 'below'), (10.0, 'above'), (1.0, None)],
)
def test_extend_grid(best, expected_direction):
    """The grid is extended only when the winner sits on a boundary."""
    results = [_cv_result(beta, l2_error=0.0 if beta == best else 1.0) for beta in (0.1, 1.0, 10.0)]

    extension = MNRidge()._extend_grid(results)

    if expected_direction is None:
        assert extension == ()
    else:
        betas = [candidate.beta for candidate in extension]
        assert betas
        if expected_direction == 'below':
            assert max(betas) < best
        else:
            assert min(betas) > best


def test_extend_grid_cannot_go_below_zero():
    results = [_cv_result(beta, l2_error=0.0 if beta == 0.0 else 1.0) for beta in (0.0, 1.0)]

    assert MNRidge()._extend_grid(results) == ()


def test_cv_table_marks_the_selected_solver():
    results = [_cv_result(beta, l2_error=1.0 / beta) for beta in (0.1, 1.0, 10.0)]
    selected = select_by_criterion(results, 'l2')

    text = str(selected.cv_table(results))

    assert 'beta' in text
    assert '*' in text
