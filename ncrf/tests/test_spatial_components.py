import numpy as np
import pytest

from ncrf import spatial_components
from ncrf._solvers.spatial_components import _loaded, resolution_subspace
from ncrf.tests.test_mn_ridge import _data, _forward


def _planted(source=7, n_sensors=14, n_sources=20, n_coefficients=4, seed=3, noise=0.3):
    rng = np.random.RandomState(seed)
    forward = _forward(n_sensors, n_sources, seed=seed)
    pattern = forward.whitened_lead_field[:, source]
    h = rng.randn(n_coefficients)
    meg, covariates = [], []
    for _ in range(3):
        e = rng.randn(1200, n_coefficients)
        meg.append(np.outer(pattern, e @ h) + noise * rng.randn(n_sensors, 1200))
        covariates.append(e)
    return forward, _data(meg, covariates), source


def test_resolution_subspace_shapes_and_orthonormality():
    forward = _forward(10, 30)

    u, projection = resolution_subspace(forward, dspm=False, fraction=0.9)

    assert u.shape[0] == 30 and projection.shape == (10, u.shape[1])
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-8)
    assert u.shape[1] < resolution_subspace(forward, dspm=False, fraction=1.0)[0].shape[1]


def test_resolution_subspace_matches_explicit_gamma():
    """The QR route reproduces the singular vectors of the explicit Gamma = K G."""
    from ncrf._solvers.mn_ridge import _operator
    forward = _forward(8, 20)
    gamma = _operator(forward, 3.0, True, None) @ forward.whitened_lead_field

    u, _ = resolution_subspace(forward, 3.0, True, 1.0)

    full_u, s, _ = np.linalg.svd(gamma)
    n = len(s[s > 1e-10 * s[0]])
    # same column space as the leading singular vectors
    np.testing.assert_allclose(u[:, :n] @ u[:, :n].T, full_u[:, :n] @ full_u[:, :n].T, atol=1e-6)


def test_loading_blends_towards_scaled_identity():
    c = np.diag([1., 2., 6.])

    np.testing.assert_allclose(_loaded(c, 0.), c)
    np.testing.assert_allclose(_loaded(c, 1.), 3. * np.eye(3))


def test_recovers_a_planted_source():
    forward, data, source = _planted()

    result = spatial_components(data, forward, dspm=False, n_boot=5, tstep=0.01, chunk=4.0,
                                alphas=np.logspace(-4, -1, 3), betas=np.logspace(-1, 1, 4))

    assert result.n_components >= 1
    assert result.correlations[0] > 0.8
    assert int(np.argmax(result.source_maps[:, 0])) == source
    assert result.weights.shape == (result.n_components, 4)


def test_rejects_too_little_data():
    forward, data, _ = _planted()

    with pytest.raises(ValueError, match='fewer than 2 chunks'):
        spatial_components(data, forward, dspm=False, tstep=0.01, chunk=100.)
