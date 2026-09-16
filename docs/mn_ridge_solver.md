# The `MNRidge` solver

`MNRidge` implements the TRF estimator of Donhauser & Baillet (2020) as an
`ncrf.Solver`, so that it can be run against `ChampLasso` on the same data,
the same folds and the same metrics.

> Donhauser, P. W., & Baillet, S. (2020). Two Distinct Neural Timescales for
> Predictive Speech Processing. *Neuron*, 105(2), 385-393.e9.
> <https://doi.org/10.1016/j.neuron.2019.10.019>
> (open access: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6981026/>)

## Usage

```python
from ncrf import MNRidge, NCRFEstimator

estimator = NCRFEstimator.from_lead_field(lead_field, noise_covariance)
result = estimator.fit(data, MNRidge(beta='auto'))

print(result.solver.beta)                       # the selected ridge parameter
print(result.solver.cv_table(result.cv_results))  # the grid it was selected from
```

A fixed `beta` skips cross-validation entirely:

```python
result = estimator.fit(data, MNRidge(beta=1.0))
```

| parameter | meaning |
| --- | --- |
| `beta` | ridge parameter: a number, an explicit grid, or `'auto'` (default) |
| `snr` | assumed amplitude SNR, regularizing the *inverse operator* (not cross-validated) |
| `dspm` | noise-normalize the inverse operator (default `False`, see below) |
| `criterion` | held-out score to select `beta` by: `'l2'` (default) or `'explained-variance'` |
| `n_beta` | number of grid points `beta='auto'` derives |

## How it differs from `ChampLasso`

The two solvers do not solve the same optimization problem, which is the point
of comparing them.

| | `ChampLasso` | `MNRidge` |
| --- | --- | --- |
| localization and TRFs | estimated **jointly** | **two separate stages** |
| penalty | group lasso (sparse) | ridge / ℓ2 (dense) |
| source variances | learned from data (Champagne) | fixed depth prior |
| optimization | alternating, iterative | closed form |
| cost per fit | 30 outer × (10 + 100) inner iterations | one eigendecomposition |

## The algorithm

Stage 1 — a linear inverse operator, built from the forward model alone. It
does not depend on the stimulus, so it is computed once and cached per forward
model. With whitened data (noise covariance = identity) and the depth prior
`R = diag(w²)` that `MNEInitializer` already supplies:

```
K = R Lᵀ (L R Lᵀ + λ² I)⁻¹
λ² = (s[0] / snr)²
```

`λ²` is expressed relative to the largest singular value of the depth-weighted
lead field, because the whitened lead field carries an arbitrary overall scale
(`ForwardModel.lead_field_scaling`); this keeps `snr` dimensionless.

With `dspm=True` each source is then divided by the norm of its own rows — its
noise sensitivity, since the data are whitened. Free-orientation sources are
normalized per source block, so the ratios between orientation components
survive.

**`dspm` defaults to `False`, and should stay off for anything quantitative.**
dSPM rescales every source, which breaks the generative model the rest of the
package assumes — `theta` no longer satisfies `meg == L @ theta @ Eᵀ`, so
predictions, `explained_variance`, `l2_error`, and therefore cross-validated
selection of `beta` are all invalid. Measured on the test dataset:

| | explained variance |
| --- | --- |
| `MNRidge(beta=1.0, dspm=False)` | **+0.0130** |
| `MNRidge(beta=1.0, dspm=True)` | **−0.7895** |

The paper can use dSPM because it evaluates coherence on spatially filtered
component signals and never reconstructs sensor data. Here it is only useful
for inspecting a fitted `theta`.

Stage 2 — ridge regression. Equations 16-18 of the paper state the solution
through the SVD of the design matrix, `d_j = s_j / (s_j² + β²)`. `MNRidge`
solves the equivalent normal equations instead,

```
B (EᵀE + β² I) = b E
```

via an eigendecomposition of `EᵀE` (which stays well behaved at `β = 0`, where
a rank-deficient design would make a plain solve singular). This reuses the
`bE` and `EtE` cross-products `RegressionData` already caches, which is what
makes a whole `beta` grid nearly free. `test_ridge_matches_the_svd_formula`
asserts the two forms agree.

Both stages are linear and act on different axes of the data, so their order
does not matter:

```
K @ (b E) @ inv(EᵀE + β² I)  ==  ((K b) E) @ inv(EᵀE + β² I)
```

The implementation regresses first and projects the (much smaller) result into
source space afterwards.

## Deliberate departures from the paper

These are judgment calls, not transcription — they are the things to review
first.

1. **Segments are averaged, not summed.** Pooling segments into one regression
   would sum both cross-products, so `beta` would have to be rescaled whenever
   segments were added or removed. Averaging keeps the data term per segment,
   matching how the rest of the package averages scores over segments.

2. **The `'auto'` grid is derived from the data.** The paper searches a fixed
   grid of 20 values between `10^0.5` and `10^3.5`, which is only meaningful
   under its own normalization. `beta` is commensurate with the singular values
   of the design, so the grid is anchored on `sqrt(trace(EᵀE) / n_coefficients)`
   and is therefore invariant to covariate scaling.

3. **`snr` is not cross-validated.** Localization regularization (`snr`) and TRF
   regularization (`beta`) are independent; only `beta` is searched. The paper's
   `α` regularizes its spatial-filter step, which is not implemented here.

4. **Model selection uses held-out ℓ2 error by default.** The paper selects by
   held-out correlation; `criterion='explained-variance'` is the closer
   analogue, and `'l2'` is the ridge objective's own data term.

## Not implemented

Equations 21-26 — the generalized-eigenvalue spatial filters (`C₁w = λC₂w`),
the diagonal loading `α`, and the CCA rotation aligning components across
participants — are **group-level analysis** of already-fitted TRFs, not part of
estimating them. They have no counterpart in the `Solver` contract, which maps
one dataset to one set of source-space coefficients.

If the comparison later needs them, they belong in analysis code operating on
fitted `NCRF` models, not inside `solve()`.

## First results

On the packaged test dataset (5 s of MEG, TRF to 0.2 s, 9966 sources, 39
coefficients):

| | time | training explained variance | sources above 1% of peak |
| --- | --- | --- | --- |
| `MNRidge(beta='auto')`, 8 candidates, 3 folds | 17.7 s | +0.0047 | 100.0% |
| `ChampLasso(mu=0.0019444)`, 3 iterations | 5.2 s | +0.0064 | 17.5% |

The dense-versus-sparse contrast is the expected structural difference. The
explained variances are too small, and the dataset far too short, for the
comparison to mean anything yet — this only establishes that the two solvers
run through the same pipeline and produce comparable output.

Cross-validation selected `beta = 9.78` from a grid spanning 0.05 to 506,
comfortably inside the range rather than at a boundary. Held-out explained
variance was negative for every smaller `beta`, i.e. the unregularized end
overfits, as it should.

## Open questions

- Should the depth prior be configurable? It is currently inherited from
  `ForwardModel.mne_initializer` (depth weighting on, exponent 0.8).
- Is `snr=3.0` a sensible default for this data, and should it be searched
  jointly with `beta`?

## Tests

`ncrf/tests/test_mn_ridge.py` (31 tests, no data download required). Beyond the
usual argument handling, the ones that carry real weight:

- `test_ridge_matches_the_svd_formula` — agreement with equations 16-18
- `test_inverse_operator_matches_the_minimum_norm_formula` — agreement with the
  explicit `R Lᵀ (L R Lᵀ + λ²I)⁻¹`
- `test_solve_recovers_a_noiseless_trf` — exact recovery when overdetermined and
  noise-free
- `test_solve_pools_segments` / `test_solve_is_linear_in_the_data` — the
  invariants behind departure 1
- `test_auto_candidates_scales_with_the_design` — the invariance behind
  departure 2

```bash
pytest ncrf/tests/test_mn_ridge.py
```
