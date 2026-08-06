import numpy as np

from SearchLibrium import ZeroInflatedOrderedProbit


def test_structural_zero_mixture_has_valid_probabilities():
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0, 1, 2])
    model = ZeroInflatedOrderedProbit(_jax=False)
    model.setup(X, y, n_categories=3)
    model.params = np.array([0.4, 0.6, -0.3, 0.2, 0.8, -0.1])
    model.coeff_est = model.params.copy()

    probabilities = model.predict_proba()

    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)


def test_fit_recovers_finite_ordered_model():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(300, 2))
    seed_model = ZeroInflatedOrderedProbit(_jax=False)
    seed_model.setup(X, np.zeros(len(X), dtype=int), n_categories=4)
    true_params = np.array([
        0.25, 0.7, -0.35,
        -0.15, 0.4, 0.2,
        0.5, -0.25, -0.1,
    ])
    probabilities, _, _, _ = seed_model._probabilities_from_params(
        true_params,
        seed_model._participation_design,
        seed_model._activity_design,
    )
    y = np.array([rng.choice(4, p=row) for row in probabilities])

    model = ZeroInflatedOrderedProbit(_jax=False)
    model.setup(X, y, n_categories=4)
    result = model.fit(maxiter=250)
    predicted = model.predict_proba()

    assert result.success
    assert np.isfinite(model.loglik)
    assert np.all(np.diff(model.thresholds_) > 0.0)
    np.testing.assert_allclose(predicted.sum(axis=1), 1.0)
    assert model.predict_expected_frequency().shape == (len(X),)