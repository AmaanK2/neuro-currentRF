"""Spatial-component optimization of :cite:`donhauserTwoDistinctNeural2020`.

The paper does not read source TRFs off a single regression. After a minimum-norm
(dSPM) operator it:

1. projects the sensor data onto the leading singular vectors of the resolution
   matrix ``Gamma = K @ G`` (99.9 % of the singular-value spectrum; Eq. 13-14),
2. fits ridge regression in that subspace (Eq. 15-20),
3. finds spatial filters ``W`` that maximize explained over unexplained variance,
   a generalized eigenvalue problem ``C1 w = lambda C2 w`` with diagonal loading
   ``alpha`` (Eq. 21-25),
4. chooses ``alpha``, ``beta`` and the number of components by bootstrap on the
   held-out correlation (Eq. 30-31), and
5. maps the spatial patterns ``P = W^-1`` back to source space through the
   subspace basis (``U_Gamma``).

:func:`spatial_components` implements 1-5 for a single participant. Group
alignment with CCA (Eq. 26) and the coherence evaluation (Eq. 27-29) are not
implemented, and the volume/free-orientation source spaces used by NCRF have no
cortical normals, so the operator is not orientation-constrained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg, stats

from .._typing import FloatArray
from .mn_ridge import _operator

if TYPE_CHECKING:
    from .._data import RegressionData
    from .._forward import ForwardModel


def resolution_subspace(
        forward: ForwardModel,
        snr: float = 3.0,
        dspm: bool = True,
        fraction: float = 0.999,
) -> tuple[FloatArray, FloatArray]:
    """Leading singular vectors of the resolution matrix ``Gamma = K @ G``.

    ``Gamma`` is ``n_sources * dc`` square but has rank at most ``forward.rank``;
    it is decomposed through QR factors of ``K`` and ``G.T`` without forming it.

    Returns
    -------
    u_gamma
        Source-space basis ``U_Gamma``, shape ``(n_sources * dc, M)``, where ``M``
        is the smallest number of singular vectors whose singular values sum to
        ``fraction`` of the spectrum.
    projection
        ``K.T @ U_Gamma``, shape ``(rank, M)``: maps whitened sensor data into the
        subspace (``X_proj = X @ projection``, Eq. 14).
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"{fraction=}: must be in (0, 1]")
    k = _operator(forward, snr, dspm, None)  # (n_src * dc, rank)
    g = forward.whitened_lead_field  # (rank, n_src * dc)
    qk, rk = np.linalg.qr(k)
    qg, rg = np.linalg.qr(g.T)
    u, s, _ = np.linalg.svd(rk @ rg.T)  # Gamma = qk @ (rk @ rg.T) @ qg.T
    spectrum = np.cumsum(s) / s.sum()
    n_keep = int(np.searchsorted(spectrum, fraction) + 1)
    u_gamma = qk @ u[:, :n_keep]
    return u_gamma, k.T @ u_gamma


def _ridge_path(bE: FloatArray, EtE: FloatArray, betas: FloatArray) -> list[FloatArray]:
    """Ridge weights ``(M, K)`` for each beta from one eigendecomposition of ``EtE``."""
    eigenvalues, q = linalg.eigh(EtE)
    bEq = bE @ q
    out = []
    for beta in betas:
        denominator = eigenvalues + beta ** 2
        keep = denominator > np.finfo(float).eps * max(denominator.max(), 1.0) * len(EtE)
        out.append((bEq[:, keep] / denominator[keep]) @ q[:, keep].T)
    return out


def _loaded(c: FloatArray, alpha: float) -> FloatArray:
    """Diagonal loading, Eq. 25: ``(1 - alpha) C + alpha * tr(C) / M * I``."""
    return (1 - alpha) * c + alpha * np.trace(c) / len(c) * np.eye(len(c))


def _spatial_filters(bb, bE, EtE, b, alpha) -> FloatArray:
    """Filters ``W (M, M)``, ordered by decreasing explained/unexplained variance (Eq. 22-23)."""
    c1 = b @ EtE @ b.T  # X_hat' X_hat
    c2 = bb - bE @ b.T - b @ bE.T + c1  # (X - X_hat)' (X - X_hat)
    c1, c2 = (c1 + c1.T) / 2, (c2 + c2.T) / 2
    _, w = linalg.eigh(_loaded(c1, alpha), _loaded(c2, alpha))
    return w[:, ::-1]


def _component_correlations(w, bb, bE, EtE, b) -> FloatArray:
    """Correlation of observed and predicted component signals, for every column of ``w``."""
    sxx = np.einsum('ij,ik,kj->j', w, bb, w)
    sxh = np.einsum('ij,ik,kj->j', w, bE @ b.T, w)
    shh = np.einsum('ij,ik,kj->j', w, b @ EtE @ b.T, w)
    return sxh / np.sqrt(np.maximum(sxx * shh, np.finfo(float).tiny))


@dataclass(frozen=True)
class SpatialComponents:
    """Result of :func:`spatial_components`.

    Parameters
    ----------
    alpha, beta
        Selected diagonal loading and ridge parameter.
    n_components
        Number ``L`` of significant components retained.
    filters
        Spatial filters ``W``, shape ``(M, L)``, in the resolution subspace.
    patterns
        Spatial patterns (rows of ``W^-1``), shape ``(L, M)``.
    weights
        Component regression weights ``W.T @ B``, shape ``(L, n_coefficients)``,
        in the design's basis coefficients.
    source_maps
        Magnitude of the source-space patterns ``|U_Gamma @ P|`` per source,
        shape ``(n_sources, L)``.
    correlations
        Mean held-out correlation of every component at the selected
        hyperparameters, shape ``(M,)`` (the first ``L`` are significant).
    scores
        Weighted mean correlation (Eq. 30) over the ``alphas`` by ``betas`` grid.
    alphas, betas
        The searched grids.
    projection, u_gamma
        Subspace projection of the sensor data and the source-space basis.
    """

    alpha: float
    beta: float
    n_components: int
    filters: FloatArray
    patterns: FloatArray
    weights: FloatArray
    source_maps: FloatArray
    correlations: FloatArray
    scores: FloatArray
    alphas: FloatArray
    betas: FloatArray
    projection: FloatArray
    u_gamma: FloatArray

    def __repr__(self) -> str:
        return (f'<SpatialComponents: L={self.n_components}, alpha={self.alpha:.3g}, beta={self.beta:.3g}, '
                f'r={np.round(self.correlations[:self.n_components], 3).tolist()}>')

    def trfs(self, design) -> list[FloatArray]:
        """Component TRFs per predictor, each ``(L, n_stim, n_lags)``."""
        out = []
        start = 0
        for basis, stim_len in zip(design.basis, design.stim_lens):
            stop = start + basis.shape[1] * stim_len
            x = self.weights[:, start:stop].reshape(self.n_components, stim_len, -1)
            out.append(x @ basis.T)
            start = stop
        return out


def _chunk_products(data, projection, chunk_len):
    """Per-chunk cross-products ``bE (M, K)``, ``EtE``, ``bb`` of the projected data."""
    bE, EtE, bb = [], [], []
    for b, e in zip(data.responses, data.covariates):
        x = projection.T @ b  # (M, t)
        for start in range(0, x.shape[1] - chunk_len + 1, chunk_len):
            xs, es = x[:, start:start + chunk_len], e[start:start + chunk_len]
            bE.append(xs @ es)
            EtE.append(es.T @ es)
            bb.append(xs @ xs.T)
    if len(bE) < 2:
        raise ValueError(f"{chunk_len=} samples leaves fewer than 2 chunks of data; use shorter chunks")
    return np.array(bE), np.array(EtE), np.array(bb)


def spatial_components(
        data: RegressionData,
        forward: ForwardModel,
        *,
        snr: float = 3.0,
        dspm: bool = True,
        subspace: float = 0.999,
        alphas: FloatArray | None = None,
        betas: FloatArray | None = None,
        n_boot: int = 15,
        chunk: float = 4.0,
        holdout: float = 0.25,
        p_threshold: float = 1e-4,
        seed: int = 0,
        tstep: float | None = None,
) -> SpatialComponents:
    """Spatial components of whitened ``data``, with bootstrap hyperparameter selection.

    Parameters
    ----------
    data
        Whitened dataset (see :meth:`RegressionData.whiten`).
    forward
        The forward model ``data`` was whitened with.
    snr, dspm
        Regularization and noise normalization of the inverse operator the
        resolution matrix is built from (the paper uses dSPM).
    subspace
        Fraction of the resolution-matrix singular-value spectrum to retain.
    alphas
        Diagonal-loading grid (default: 10 values from ``10 ** -4`` to
        ``10 ** -0.5``, as in the paper).
    betas
        Ridge grid (default: 20 values spanning four decades around the design
        scale, as in :class:`MNRidge`; the paper's fixed grid depends on its own
        normalization).
    n_boot
        Bootstrap repetitions (paper: 15).
    chunk, holdout
        Seconds per bootstrap chunk (paper: 4 s) and fraction of chunks held out
        in each repetition. The paper holds out 150 chunks of a long recording;
        here the fraction is used instead.
    p_threshold
        Components are retained, in order, while their held-out correlation is
        significantly above zero across repetitions at this level (paper: 1e-4).
    seed
        Seed of the bootstrap random draws.
    tstep
        Sample spacing in seconds (default ``data.design.tstep``).
    """
    rng = np.random.RandomState(seed)
    if tstep is None:
        tstep = data.design.tstep
    u_gamma, projection = resolution_subspace(forward, snr, dspm, subspace)
    m = projection.shape[1]
    chunk_len = max(int(round(chunk / tstep)), 1)
    BE, ETE, BB = _chunk_products(data, projection, chunk_len)
    n_chunks = len(BE)

    mean_ete = ETE.mean(0)
    if alphas is None:
        alphas = np.logspace(-4, -0.5, 10)
    if betas is None:
        betas = np.sqrt(np.trace(mean_ete) / len(mean_ete)) * np.logspace(-2, 2, 20)
    alphas, betas = np.asarray(alphas, float), np.asarray(betas, float)

    n_hold = min(max(int(round(holdout * n_chunks)), 1), n_chunks - 1)
    r = np.zeros((n_boot, len(alphas), len(betas), m))
    for i in range(n_boot):
        test = np.zeros(n_chunks, bool)
        test[rng.choice(n_chunks, n_hold, replace=False)] = True
        bE, eTe, bb = BE[~test].mean(0), ETE[~test].mean(0), BB[~test].mean(0)
        bE_t, eTe_t, bb_t = BE[test].mean(0), ETE[test].mean(0), BB[test].mean(0)
        for j, b in enumerate(_ridge_path(bE, eTe, betas)):
            for a, alpha in enumerate(alphas):
                w = _spatial_filters(bb, bE, eTe, b, alpha)
                r[i, a, j] = _component_correlations(w, bb_t, bE_t, eTe_t, b)

    mean_r = r.mean(0)  # (alpha, beta, component)
    weight = np.abs(mean_r.sum(axis=(0, 1)))  # Eq. 30: high-performing components dominate
    scores = (mean_r * weight).sum(-1)
    a_best, b_best = np.unravel_index(np.argmax(scores), scores.shape)
    alpha, beta = float(alphas[a_best]), float(betas[b_best])

    # significance of each component's held-out correlation (Eq. 31)
    rr = r[:, a_best, b_best]  # (n_boot, M)
    n_components = 0
    for i in range(m):
        if n_boot < 2:
            significant = rr[:, i].mean() > 0
        else:
            significant = stats.ttest_1samp(rr[:, i], 0., alternative='greater').pvalue < p_threshold
        if not significant:
            break
        n_components += 1
    n_components = max(n_components, 1)

    # final fit on all data
    bE, eTe, bb = BE.mean(0), ETE.mean(0), BB.mean(0)
    b = _ridge_path(bE, eTe, [beta])[0]
    w = _spatial_filters(bb, bE, eTe, b, alpha)
    patterns = np.linalg.inv(w)[:n_components]  # (L, M)
    dc = forward.dc
    maps = np.linalg.norm((u_gamma @ patterns.T).reshape(-1, dc, n_components), axis=1)
    return SpatialComponents(
        alpha=alpha, beta=beta, n_components=n_components,
        filters=w[:, :n_components], patterns=patterns,
        weights=w[:, :n_components].T @ b, source_maps=maps,
        correlations=rr.mean(0), scores=scores, alphas=alphas, betas=betas,
        projection=projection, u_gamma=u_gamma,
    )
