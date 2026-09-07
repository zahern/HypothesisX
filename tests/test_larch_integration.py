"""Tests for the Larch/ABM integration (regularization, jax_utils, skims,
accessibility, abm presets, registry wiring).

Larch- and openmatrix-dependent paths are skipped when those packages are
absent; everything here runs on the base dependencies (numpy/pandas/pytest).
"""
import io

import numpy as np
import pandas as pd
import pytest


def _needs_larch():
    return pytest.mark.skipif(
        __import__("importlib").util.find_spec("larch") is None,
        reason="larch not installed")


# ---------------------------------------------------------------------------
# regularization (no larch needed)
# ---------------------------------------------------------------------------

def test_penalty_math():
    from SearchLibrium.regularization import elasticnet_penalty_and_grad
    pvals = np.array([1.0, -2.0, 0.0])
    pen, grad = elasticnet_penalty_and_grad(pvals, None, alpha=0.5, l1_ratio=0.5)
    # l1 = 3, l2 = 5 -> 0.5*(0.5*3 + 0.5*5) = 2.0
    assert pen == pytest.approx(2.0)
    # grad = 0.5*(0.5*sign + 0.5*2*dev) = [0.75, -1.25, 0.0]
    np.testing.assert_allclose(grad, [0.75, -1.25, 0.0])


def test_penalty_holdfast_and_reference():
    from SearchLibrium.regularization import elasticnet_penalty_and_grad
    pvals = np.array([1.0, 1.0])
    pen, grad = elasticnet_penalty_and_grad(
        pvals, np.array([True, False]), alpha=1.0, l1_ratio=0.0,
        reference=np.array([1.0, 0.0]))
    # param 0: dev 0 (also held fast); param 1: dev 1 -> penalty 1.0
    assert pen == pytest.approx(1.0)
    np.testing.assert_allclose(grad, [0.0, 2.0])


def test_objective_noop_when_alpha_zero():
    from SearchLibrium.regularization import elasticnet_objective

    class Dummy:
        def __init__(self):
            self.logloss = lambda x=None: 1.0
            self.d_logloss = lambda x=None: np.zeros(2)
            self.pnames = ["a", "b"]
            self.pvals = np.zeros(2)

    m = Dummy()
    with elasticnet_objective(m, alpha=0.0) as out:
        assert out is m
    assert not hasattr(m, "_sl_reg_alpha")


def test_nl_priors_recipe():
    from SearchLibrium.regularization import nl_identification_priors

    class Dummy:
        pnames = ["Mu_car", "ASC_bus", "b_time"]

    ref, w = nl_identification_priors(Dummy())
    assert ref["Mu_car"] == pytest.approx(1.0)
    assert w["Mu_car"] == pytest.approx(10.0)
    assert w["ASC_bus"] == pytest.approx(5.0)
    assert ref["b_time"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# jax_utils (no jax needed for size/guard logic)
# ---------------------------------------------------------------------------

def test_is_large_model():
    from SearchLibrium.jax_utils import is_large_model
    assert is_large_model(n_cases=1000, n_alts=371)
    assert not is_large_model(n_cases=100, n_alts=4)
    assert is_large_model(n_cells=1_000_000)


def test_estimate_row_guard_flags_sampling():
    from SearchLibrium.jax_utils import estimate_row_guard
    g = estimate_row_guard(50_000, 371, sample_size=20)
    assert g["full_rows"] == 50_000 * 371
    assert g["sampled_rows"] == 50_000 * 20
    assert g["use_sampling"]
    assert g["oom_risk"]


# ---------------------------------------------------------------------------
# skims (DataFrame-level; no openmatrix needed)
# ---------------------------------------------------------------------------

def test_add_derived_los():
    from SearchLibrium.skims import add_derived_los
    df = pd.DataFrame({
        "OTAZ": [1, 1, 2], "DTAZ": [1, 2, 1],
        "SOVTOLL_TIME__AM": [10.0, 20.0, 30.0],
        "SOVTOLL_TIME__PM": [12.0, 22.0, 32.0],
        "DIST": [1.0, 2.0, 3.0],
        "DISTWALK": [1.1, 2.2, 3.3], "DISTBIKE": [1.2, 2.1, 3.0],
    })
    out = add_derived_los(df)
    np.testing.assert_allclose(out["tt_avg"], [11.0, 21.0, 31.0])
    np.testing.assert_allclose(out["log_dist"], np.log1p([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out["cost_proxy"], [0.12, 0.24, 0.36])
    assert "distwalk" in out.columns and "tt_am" in out.columns


def test_merge_skim_component():
    from SearchLibrium.skims import merge_skim_component
    skims = pd.DataFrame({"OTAZ": [1, 2], "DTAZ": [2, 1], "DIST": [5.0, 6.0]})
    extra = pd.DataFrame({"OTAZ": [1], "DTAZ": [2], "crash_rate": [0.7]})
    out = merge_skim_component(skims, extra, ["crash_rate"])
    assert out.loc[0, "crash_rate"] == pytest.approx(0.7)
    assert out.loc[1, "crash_rate"] == pytest.approx(0.0)


def test_prepare_skims_passthrough():
    from SearchLibrium.skims import prepare_skims_from_omx
    df = pd.DataFrame({"OTAZ": [1], "DTAZ": [1]})
    assert prepare_skims_from_omx(df) is df


# ---------------------------------------------------------------------------
# accessibility
# ---------------------------------------------------------------------------

def test_gravity_accessibility():
    from SearchLibrium.accessibility import gravity_accessibility
    skims = pd.DataFrame({
        "OTAZ": [1, 1, 2, 2], "DTAZ": [1, 2, 1, 2],
        "tt": [10.0, 20.0, 20.0, 5.0],
    })
    landuse = pd.DataFrame({"jobs": [100.0, 200.0]}, index=[1, 2])
    acc = gravity_accessibility(skims, landuse, {"jobs": 1.0}, "tt",
                                decay_coeff=0.0)
    assert acc.loc[1] == pytest.approx(300.0)
    assert acc.loc[2] == pytest.approx(300.0)


def test_logsum_accessibility():
    from SearchLibrium.accessibility import logsum_accessibility
    ls = pd.DataFrame({"z1": [0.0, 1.0], "z2": [0.0, 1.0]}, index=[1, 2])
    acc = logsum_accessibility(ls)
    assert acc.loc[1] == pytest.approx(np.log(2.0))
    assert acc.loc[2] == pytest.approx(1.0 + np.log(2.0))


# ---------------------------------------------------------------------------
# abm presets
# ---------------------------------------------------------------------------

def test_stage_configs():
    from SearchLibrium.abm import available_stages, stage_search_config
    assert set(available_stages()) >= {"frequency", "destination", "mode", "parking"}
    cfg = stage_search_config("mode")
    assert "larch_nested" in cfg["models"]
    assert cfg["needs_nests"]
    dest = stage_search_config("destination", n_zones=371)
    assert dest.get("needs_sampling")
    with pytest.raises(ValueError):
        stage_search_config("nope")


# ---------------------------------------------------------------------------
# registry + Parameters wiring (no larch needed)
# ---------------------------------------------------------------------------

def test_registry_lists_larch_models():
    from SearchLibrium.search import ModelRegistry
    models = ModelRegistry().get_models()
    assert {"larch_mnl", "larch_nested", "larch_mixed"} <= set(models)


def test_parameters_accepts_larch_opts_quietly(capsys):
    from SearchLibrium.search import Parameters
    df = pd.DataFrame({"asc_a": [1, 0, 1, 0], "x1": [0.5, 0.2, 0.3, 0.8]})
    p = Parameters(
        criterions=[("bic", -1)], df=df, varnames=["asc_a", "x1"],
        asvarnames=["asc_a", "x1"], isvarnames=[], choice_set=[0, 1],
        choices=np.array([1, 0, 1, 0], dtype="int32"),
        alt_var=np.array([1, 0, 1, 0], dtype="int32"),
        choice_id=np.array([0, 1, 2, 3], dtype="int32"),
        ind_id=np.array([0, 1, 2, 3], dtype="int32"),
        models=["multinomial"], auto_as_is=False, larch_opts={"reg_alpha": 0.01})
    assert p.larch_opts == {"reg_alpha": 0.01}
    out = capsys.readouterr().out
    assert "Unexpected keyword" not in out


@_needs_larch()
def test_larch_mnl_fits_synthetic():
    import larch  # noqa: F401
    from SearchLibrium.larch_models import LarchMNL
    rng = np.random.default_rng(3)
    n, alts = 300, [0, 1, 2]
    frames = []
    for i in range(n):
        x = rng.normal(size=len(alts))
        u = x + rng.gumbel(size=len(alts))
        ch = int(np.argmax(u))
        for j, a in enumerate(alts):
            frames.append((i, a, x[j], 1 if j == ch else 0))
    df = pd.DataFrame(frames, columns=["case", "alt", "x", "ch"])
    m = LarchMNL()
    m.setup(X=df[["x"]].values, y=df["ch"].values, varnames=["x"],
            alts=df["alt"].values, ids=df["case"].values, base_alt=0)
    m.fit()
    assert np.isfinite(m.loglik)
    assert len(m.coeff_est) >= 1


@_needs_larch()
def test_larch_compat_noop_flags():
    from SearchLibrium.larch_compat import apply_larch_patches
    assert apply_larch_patches() in (True, False)
