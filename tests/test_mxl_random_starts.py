import numpy as np
import pandas as pd
import pytest

from SearchLibrium.MixedLogit import _mxl_random_start, _mxl_variance_start
from SearchLibrium.search import Parameters, Search


def test_lognormal_start_preserves_utility_space_moments():
    location, scale, distribution = _mxl_random_start(
        2.0, "ln", relative_sd=0.25, sd_floor=1e-12)
    mean = np.exp(location + 0.5 * scale ** 2)
    variance = (np.exp(scale ** 2) - 1.0) * np.exp(2.0 * location + scale ** 2)
    assert distribution == "ln"
    assert mean == pytest.approx(2.0)
    assert np.sqrt(variance) == pytest.approx(0.5)


def test_sign_constrained_start_follows_mnl_sign():
    location, scale, distribution = _mxl_random_start(-2.0, "ln")
    assert distribution == "nln"
    assert -np.exp(location + 0.5 * scale ** 2) == pytest.approx(-2.0)
    assert scale > 0.0

    location, scale, distribution = _mxl_random_start(2.0, "nln")
    assert distribution == "ln"
    assert np.exp(location + 0.5 * scale ** 2) == pytest.approx(2.0)
    assert scale > 0.0


def test_correlated_start_has_zero_off_diagonal_cholesky_terms():
    means, variance_start, distributions = _mxl_variance_start(
        [1.0, -2.0, 0.5], ["n", "n", "n"], correlation_length=2,
        relative_sd=0.1)
    assert means.tolist() == [1.0, -2.0, 0.5]
    assert distributions == ["n", "n", "n"]
    assert variance_start[1] == 0.0
    assert variance_start[0] > 0.0
    assert variance_start[2] > 0.0
    assert variance_start[3] > 0.0


def test_search_forwards_random_start_options(monkeypatch):
    import SearchLibrium.search as search_module

    captured = {}

    class FakeMixedLogit:
        def __init__(self, **kwargs):
            pass

        def setup(self, **kwargs):
            captured.update(kwargs)

        def fit(self):
            pass

    monkeypatch.setattr(search_module, "MixedLogit", FakeMixedLogit)
    params = Parameters(
        [('bic', -1)], pd.DataFrame({'x': [0.0, 1.0]}), ['x'],
        choice_set=[0, 1], choices=[1, 0], models=['mixed_logit'],
        random_sd_start=0.25, random_sd_floor=0.02,
        align_random_distributions=False, auto_as_is=False)

    Search(params).fit_mxl(
        X=np.array([[1.0], [2.0]]), y=np.array([1, 0]), varnames=['x'],
        alts=np.array([0, 1]), isvars=[], transvars=[], ids=np.array([0, 1]),
        panels=None, randvars={'x': 'n'}, corvars=[], fit_intercept=False,
        init_coeff=None, n_draws=8, weights=None, avail=None, base_alt=None,
        maxiter=1, ftol=1e-6, gtol=1e-6, save_fitted_params=False)

    assert captured['random_sd_start'] == 0.25
    assert captured['random_sd_floor'] == 0.02
    assert captured['align_random_distributions'] is False
    assert captured['n_draws'] == 8
