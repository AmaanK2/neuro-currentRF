"""Minimum-norm + ridge NCRF solver.

This module implements the two-stage TRF estimator of
:cite:`donhauserTwoDistinctNeural2020` as a :class:`~ncrf.Solver`, so that it can
be compared against the joint :class:`~ncrf.ChampLasso` estimator on the same
data, folds and metrics.

The two algorithms solve genuinely different problems. ChampLasso estimates
source distribution and TRFs jointly, with a sparsity-inducing group penalty.
Here the two steps are separate and neither is sparse:

1. A linear inverse operator (minimum norm, optionally dSPM-normalized) is
   derived from the forward model alone -- it does not depend on the stimulus.
2. The TRF coefficients are obtained by ridge regression, in closed form.

Because both steps are linear and act on different axes of the data (the
inverse operator on sensors, the regression on time), applying them in either
order gives the same result::

    K @ (b @ E) @ inv(E.T @ E + beta ** 2 * I)  ==  ((K @ b) @ E) @ inv(...)

so the implementation regresses first, on the cached cross-products, and
projects the much smaller result into source space afterwards.
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, replace
from numbers import Real
from typing import TYPE_CHECKING
from collections.abc import Sequence

from eelbrain import fmtxt
import numpy as np
from scipy import linalg

from .._crossvalidation import crossvalidate
from .._repr import _theta_repr
from .._typing import BetaArg, FloatArray
from .base import Solver, SolverFit

if TYPE_CHECKING:
    from .._crossvalidation import CrossValidation, CVResult
    from .._data import RegressionData
    from .._forward import ForwardModel
    from .._model import NCRFEstimator

#: Criteria :class:`MNRidge` can select ``beta`` by, mapping to the held-out
#: score to optimize and whether that score is minimized.
CRITERIA = {
    'l2': ('l2_error', True),
    'explained-variance': ('explained_variance', False),
}


def _is_number(value: object) -> bool:
    """Whether ``value`` is a real number; ``bool`` is rejected."""
    return isinstance(value, Real) and not isinstance(value, bool)


#: Inverse operators by forward model (see :func:`_inverse_operator`); entries
#: die with their forward model.
_operator_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _inverse_operator(
        forward: ForwardModel,
        snr: float,
        dspm: bool,
) -> FloatArray:
    """Linear inverse operator for the whitened lead field, cached per forward model.

    The operator is a pure function of ``(forward, snr, dspm)`` and is only ever
    read, so the candidates of a ``beta`` grid and the folds a cross-validation
    worker scores all share one instance.

    Parameters
    ----------
    forward
        Forward model supplying the whitened lead field and the depth weighting.
    snr
        Assumed amplitude signal-to-noise ratio, setting the regularization
        ``lambda ** 2 = (s[0] / snr) ** 2`` relative to the largest singular
        value of the depth-weighted lead field. Expressing it relative to ``s[0]``
        keeps ``snr`` dimensionless, which matters because the whitened lead
        field carries an arbitrary overall scale
        (:attr:`ForwardModel.lead_field_scaling`).
    dspm
        Noise-normalize the operator. The data are whitened, so each source's
        noise sensitivity is just the norm of its own row(s).

    Returns
    -------
    numpy.ndarray
        Operator mapping whitened sensor data to source space, shape
        ``(n_sources * dc, n_sensors)``.
    """
    if snr <= 0:
        raise ValueError(f"{snr=}: must be > 0")
    # The initializer already holds the depth weights and the SVD of the
    # depth-weighted lead field, which is exactly what the operator needs.
    initializer = forward.mne_initializer
    w, u, s, vh = initializer.w, initializer.u, initializer.s, initializer.vh
    lambda2 = (s[0] / snr) ** 2
    # With source covariance R = diag(w ** 2) and whitened data (noise = I),
    #     K = R L.T (L R L.T + lambda2 I)^-1 = w * vh.T @ diag(s / (s ** 2 + lambda2)) @ u.T
    k = (w[:, None] * vh.T) * (s / (s ** 2 + lambda2))[None, :] @ u.T
    if not dspm:
        return k
    # dSPM: divide each source by the standard deviation of its own noise
    # projection. Free-orientation sources are normalized as a block, so that
    # the ratio between their orientation components is preserved.
    dc = forward.dc
    noise = np.sqrt((k ** 2).sum(axis=1).reshape(-1, dc).sum(axis=1) / dc)
    return k / np.repeat(noise, dc)[:, None]


def _pooled_cross_products(data: RegressionData) -> tuple[FloatArray, FloatArray]:
    """Average ``b @ E`` and ``E.T @ E`` over segments.

    Pooling the segments into one regression would sum the two cross-products,
    which both grow with the number of segments; ``beta`` would then have to be
    rescaled whenever segments are added or removed. Averaging instead keeps the
    data term per segment, so ``beta`` means the same thing for any number of
    segments -- the same convention the rest of the package follows when it
    averages scores over segments.
    """
    bE = np.mean(data.bE, axis=0)
    EtE = np.mean(data.EtE, axis=0)
    return bE, EtE


def _ridge(bE: FloatArray, EtE: FloatArray, beta: float) -> FloatArray:
    """Sensor-space ridge TRFs, shape ``(n_sensors, n_coefficients)``.

    Solves ``B (E.T E + beta ** 2 I) = b E``, the normal equations of
    ``min_B ||b - B E.T|| ** 2 + beta ** 2 ||B|| ** 2``. Equation 17 of
    :cite:`donhauserTwoDistinctNeural2020` states the same solution through the
    singular values of ``E`` (``d = s / (s ** 2 + beta ** 2)``); going through
    the normal equations instead reuses the cross-products the dataset already
    caches, which is what makes a whole ``beta`` grid cheap.

    ``E.T @ E`` is symmetric positive semidefinite, so it is inverted through its
    eigendecomposition: that stays well behaved at ``beta = 0``, where a
    rank-deficient design would make a plain linear solve singular.
    """
    eigenvalues, Q = linalg.eigh(EtE)
    denominator = eigenvalues + beta ** 2
    # A non-positive denominator is a direction that neither the design nor beta
    # constrains; dropping it makes the unregularized solution the pseudo-inverse
    threshold = np.finfo(float).eps * max(denominator.max(), 1.0) * len(EtE)
    keep = denominator > threshold
    Q = Q[:, keep]
    return (bE @ Q) / denominator[keep] @ Q.T


@dataclass(frozen=True, repr=False)
class MNRidgeFit(SolverFit):
    """Fitted state produced by :class:`MNRidge`.

    Parameters
    ----------
    beta
        The ridge parameter the fit was obtained with.
    """

    beta: float

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_theta_repr(self.theta)}, beta={self.beta:g}>'

    def score(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> dict[str, float]:
        """Ridge penalty on this fit, and the objective it adds to on ``data``.

        The data term is the unweighted squared error, which is also what the
        ``l2_error`` model metric reports; ``ridge_objective`` is repeated here
        so that the quantity the solver actually minimizes can be read off
        directly.
        """
        error = 0.0
        for meg, covariates in data:
            residual = meg - forward.whitened_lead_field @ self.theta @ covariates.T
            error += 0.5 * (residual ** 2).sum()
        penalty = 0.5 * self.beta ** 2 * (self.theta ** 2).sum()
        return {'ridge_penalty': penalty, 'ridge_objective': error / len(data) + penalty}


def select_by_criterion(cv_results: Sequence[CVResult], criterion: str = 'l2') -> MNRidge:
    """Pick the best solver from cross-validation results by the given criterion.

    Parameters
    ----------
    cv_results
        Results to choose from.
    criterion
        Criterion for best fit. Possible values:

        - ``'l2'``: the smallest held-out squared error (default)
        - ``'explained-variance'``: the largest held-out explained variance,
          which is the closest analogue of the held-out correlation that
          :cite:`donhauserTwoDistinctNeural2020` selects by
    """
    if criterion not in CRITERIA:
        raise ValueError(f'{criterion=}')
    key, minimize = CRITERIA[criterion]
    pick = min if minimize else max
    return pick(cv_results, key=lambda result: result.scores[key]).solver


@dataclass(frozen=True)
class MNRidge(Solver):
    """Minimum-norm/ridge NCRF solver :cite:`donhauserTwoDistinctNeural2020`.

    Source localization and TRF estimation are separate: a linear inverse
    operator derived from the forward model alone projects the data into source
    space, and the TRFs are the closed-form ridge solution. Unlike
    :class:`ChampLasso` this is a single non-iterative solve per ``beta``, and
    the estimate is dense rather than sparse.

    Parameters
    ----------
    beta
        Ridge parameter. A number fits one model, a sequence selects among an
        explicit grid with cross-validation, and ``'auto'`` derives and
        cross-validates a grid from the data (default).
    snr
        Assumed amplitude signal-to-noise ratio, regularizing the inverse
        operator. Note that this regularizes the *localization*, whereas
        ``beta`` regularizes the *TRFs*; the two are independent, and only
        ``beta`` is cross-validated.
    dspm
        Noise-normalize the inverse operator (default). This reproduces the
        dSPM operator of :cite:`donhauserTwoDistinctNeural2020`, but it rescales
        each source by its own noise sensitivity, so the resulting ``theta`` is
        in units of a statistic rather than of source current. Set to ``False``
        for a plain minimum-norm estimate whose amplitudes are comparable with
        :class:`ChampLasso`.
    criterion
        Held-out score to select ``beta`` by; see :func:`select_by_criterion`.
    n_beta
        Number of grid points ``beta='auto'`` derives.
    """

    beta: BetaArg = 'auto'
    snr: float = 3.0
    dspm: bool = True
    criterion: str = 'l2'
    n_beta: int = 20

    def __post_init__(self) -> None:
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion={self.criterion!r}: expected one of {sorted(CRITERIA)}")
        if self.n_beta < 1:
            raise ValueError(f"n_beta={self.n_beta}: must be at least 1")

    def search(
            self,
            estimator: NCRFEstimator,
            data: RegressionData,
            cv: CrossValidation,
    ) -> tuple[MNRidge, list[CVResult]]:
        """Resolve ``beta``, and cross-validate unless it is a fixed number."""
        candidates = self.candidates(estimator.forward, data)
        if _is_number(self.beta):
            return candidates[0], []
        cv_results = crossvalidate(estimator, data, candidates, cv)
        extension = self._extend_grid(cv_results)
        if extension:
            cv_results.extend(crossvalidate(estimator, data, extension, cv))
        return select_by_criterion(cv_results, self.criterion), cv_results

    def _extend_grid(self, cv_results: Sequence[CVResult]) -> tuple[MNRidge, ...]:
        """Return one additional decade when the winner is on a grid boundary."""
        logger = logging.getLogger(__name__)
        best = select_by_criterion(cv_results, self.criterion)
        betas = [result.solver.beta for result in cv_results]
        if best.beta == min(betas):
            if best.beta == 0.0:
                return ()  # cannot extend below the unregularized fit
            new_betas = np.logspace(np.log10(best.beta) - 1, np.log10(best.beta), 4)[:-1]
            direction = 'left'
        elif best.beta == max(betas):
            new_betas = np.logspace(np.log10(best.beta), np.log10(best.beta) + 1, 4)[1:]
            direction = 'right'
        else:
            return ()
        logger.info(f'Best beta is {best.beta}: extending range of beta towards the {direction}')
        return tuple(replace(best, beta=float(beta)) for beta in new_betas)

    def cv_table(self, cv_results: Sequence[CVResult]) -> fmtxt.Table:
        """Summarize cross-validation scores by ``beta``.

        Call this on the solver that was selected by :meth:`search`; the table
        marks it among the candidates it was chosen from, and warns when it sits
        at the bottom of the grid.

        Parameters
        ----------
        cv_results
            The results the selection was made from.
        """
        results = sorted(cv_results, key=lambda result: result.solver.beta)
        best = {criterion: select_by_criterion(cv_results, criterion).beta for criterion in CRITERIA}

        table = fmtxt.Table('lllll')
        table.cells('beta', 'l2-error', 'explained variance', 'ridge objective', 'ES metric')
        table.midrule()
        fmt = '%.5f'
        for result in results:
            text = fmtxt.stat(result.solver.beta, fmt=fmt)
            if result.solver == self:
                text += '*'
            table.cell(text)
            table.cell(fmtxt.stat(result.scores['l2_error'], fmt, 1 if result.solver.beta == best['l2'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['explained_variance'], fmt, 1 if result.solver.beta == best['explained-variance'] else 0, 1))
            table.cell(fmtxt.stat(result.scores['ridge_objective'], fmt=fmt))
            table.cell(fmtxt.stat(result.scores['estimation_stability'], fmt=fmt))
        if self.beta == min(result.solver.beta for result in results):
            table.caption("Warnings: Best beta is smallest beta")
        return table

    def candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[MNRidge, ...]:
        """Resolve ``beta`` into fixed solver configurations."""
        beta = self.beta
        if isinstance(beta, str):
            if beta != 'auto':
                raise ValueError(f"{beta=}: expected a number, a sequence of numbers, or 'auto'")
            return self.auto_candidates(forward, data)
        if _is_number(beta):
            values = (beta,)
        else:
            try:
                values = tuple(beta)
            except TypeError:
                raise TypeError(f"{beta=}: expected a number, a sequence of numbers, or 'auto'") from None
            if not values:
                raise ValueError(f"{beta=}: grid must contain at least one value")
            if not all(_is_number(value) for value in values):
                raise TypeError(f"{beta=}: all grid values must be numbers")
        if any(value < 0 for value in values):
            raise ValueError(f"{beta=}: beta must be non-negative")
        return tuple(replace(self, beta=float(value)) for value in values)

    def auto_candidates(
            self,
            forward: ForwardModel,
            data: RegressionData,
    ) -> tuple[MNRidge, ...]:
        """Derive a ``beta`` grid spanning four decades around the design scale.

        :cite:`donhauserTwoDistinctNeural2020` searches a fixed grid of 20 values
        between ``10 ** 0.5`` and ``10 ** 3.5``, which is only meaningful for
        their particular normalization. ``beta`` is commensurate with the
        singular values of the design matrix, so the grid is anchored here on
        the root mean eigenvalue of ``E.T @ E`` instead, making it invariant to
        how the covariates were scaled.
        """
        _, EtE = _pooled_cross_products(data)
        scale = np.sqrt(np.trace(EtE) / len(EtE))
        betas = scale * np.logspace(-2, 2, self.n_beta)
        return tuple(replace(self, beta=float(beta)) for beta in betas)

    def solve(
            self,
            forward: ForwardModel,
            data: RegressionData,
            *,
            verbose: bool = False,
    ) -> MNRidgeFit:
        """Estimate NCRF weights for one prepared, whitened dataset."""
        if not _is_number(self.beta):
            raise ValueError("MNRidge.solve() requires a fixed numeric beta; use NCRFEstimator.fit() to resolve a grid or beta='auto'")
        logger = logging.getLogger(__name__)
        if verbose:
            logger.info(f'MNRidge: solving with beta={self.beta:g}, snr={self.snr:g}, dspm={self.dspm}')
        k = _operator(forward, self.snr, self.dspm)
        bE, EtE = _pooled_cross_products(data)
        theta = k @ _ridge(bE, EtE, self.beta)
        return MNRidgeFit(theta=theta, beta=float(self.beta))


def _operator(forward: ForwardModel, snr: float, dspm: bool) -> FloatArray:
    """Look up or build the cached inverse operator for ``forward``."""
    key = (snr, dspm)
    cached = _operator_cache.get(forward)
    if cached is None:
        cached = _operator_cache[forward] = {}
    if key not in cached:
        cached[key] = _inverse_operator(forward, snr, dspm)
    return cached[key]
