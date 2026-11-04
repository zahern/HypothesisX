"""Tests for the Larch/ABM integration (regularization, jax_utils, skims,
accessibility, abm presets, registry wiring).

Larch- and openmatrix-dependent paths are skipped when those packages are
absent; everything here runs on the base dependencies (numpy/pandas/pytest).
"""
import io

import numpy as np
import pandas as pd
import pytest


_LARCH_WORKS = None


def _larch_works():
    """True when larch is installed AND can fit a trivial MNL.

    larch can be importable yet non-functional when its sharrow dataflow
    stack is missing/incompatible: every larch flow then fails with
    "Setup failed for variable ..." (even larch's own bundled examples).
    Probe once so the larch-dependent tests skip cleanly instead of failing
    on an environment problem unrelated to SearchLibrium.
    """
    global _LARCH_WORKS
    if _LARCH_WORKS is None:
        _LARCH_WORKS = False
        try:
            from SearchLibrium.jax_utils import ensure_jax_environment
            ensure_jax_environment()
            import larch as lx
            from larch import P, X as LX

            rng = np.random.default_rng(0)
            rows = []
            for i in range(12):
                x = rng.normal(size=2)
                ch = int(np.argmax(x + rng.gumbel(size=2)))
                for j, a in enumerate((0, 1)):
                    rows.append((i, a, x[j], 1 if j == ch else 0))
            frame = pd.DataFrame(rows, columns=["case", "alt", "x", "ch"])
            frame = frame.set_index(["case", "alt"])
            ds = lx.Dataset.construct.from_idca(frame)
            m = lx.Model(ds)
            m.choice_ca_var = "ch"
            m.utility_ca = P("B_x") * LX("x")
            m.maximize_loglike()
            _LARCH_WORKS = bool(np.isfinite(m.loglike()))
        except Exception:
            _LARCH_WORKS = False
    return _LARCH_WORKS


def _needs_larch():
    if __import__("importlib").util.find_spec("larch") is None:
        return pytest.mark.skipif(True, reason="larch not installed")
    return pytest.mark.skipif(
        not _larch_works(),
        reason="larch installed but its sharrow dataflow stack cannot fit a "
               "trivial MNL in this environment")


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
def test_larch_mixed_correlated_normal_fits():
    from SearchLibrium.larch_models import LarchMixedLogit

    rng = np.random.default_rng(11)
    n, alts = 30, [0, 1, 2]
    X = rng.normal(size=(n * len(alts), 2))
    ids = np.repeat(np.arange(n), len(alts))
    alt_codes = np.tile(alts, n)
    y = np.zeros(n * len(alts))
    y[np.arange(n) * len(alts) + rng.integers(len(alts), size=n)] = 1
    model = LarchMixedLogit().setup(
        X=X, y=y, varnames=["x", "z"], alts=alt_codes, ids=ids,
        randvars={"x": "n", "z": "n"}, corvars=["x", "z"],
        n_draws=8, draw_method="halton", draw_seed=3, maxiter=2,
    )
    model.fit()
    names = [str(name) for name in model.coeff_names]
    assert {"chol.x.x", "chol.z.x", "chol.z.z"} <= set(names)
    assert np.isfinite(model.loglik)


@_needs_larch()
def test_larch_latent_class_membership_flow_recovers_classes():
    """Regression for the Larch/sharrow membership flow.

    Previously LarchLatentClass handed larch a DataTree (ds.dc.as_tree())
    instead of the Dataset+groupid it needs, so the membership model was left
    with ``graph is None`` and every fit died with
    "'NoneType' object has no attribute 'node_slot_arrays'".  With the fix the
    model fits, recovers two distinct class slopes (no class collapse), and the
    coefficient frame is extracted by name (not positionally).
    """
    from SearchLibrium.larch_models import LarchLatentClass

    rng = np.random.default_rng(1)
    n, J = 600, 3
    z = rng.normal(size=n)
    cls = (0.8 + 1.6 * z + rng.normal(scale=0.4, size=n) > 0).astype(int)
    x = rng.normal(size=(n, J))
    beta = np.array([-1.8, 1.2])            # opposite-signed class tastes
    u = x * beta[cls][:, None]
    pr = np.exp(u)
    pr /= pr.sum(1, keepdims=True)
    ch = np.array([rng.choice(J, p=pr[i]) for i in range(n)])
    chw = np.zeros((n, J))
    chw[np.arange(n), ch] = 1
    ids = np.repeat(np.arange(n), J)
    alts = np.tile([1, 2, 3], n)
    X = np.column_stack([x.ravel(), np.repeat(z, J)])
    y = chw.ravel()

    mdl = LarchLatentClass()
    mdl.setup(X=X, y=y, varnames=["x", "z"], alts=alts, ids=ids,
              num_classes=2, class_params_spec=[["x"], ["x"]],
              member_params_spec=["z"], fit_intercept=False, maxiter=400)
    mdl.fit()

    assert hasattr(mdl, "_larch_model")        # fit did not fall into _fail
    assert np.isfinite(mdl.loglik)
    names = [str(c) for c in mdl.coeff_names]
    assert names == ["LC0_B_x", "LC1_B_x", "LCM1_intercept", "LCM1_z"]
    betas = dict(zip(names, np.asarray(mdl.betas, dtype=float)))
    # classes must not collapse to the same taste coefficient
    assert abs(betas["LC0_B_x"] - betas["LC1_B_x"]) > 0.5
    # and the two class slopes should straddle zero (opposite tastes recovered)
    assert betas["LC0_B_x"] * betas["LC1_B_x"] < 0


@_needs_larch()
def test_larch_compat_noop_flags():
    from SearchLibrium.larch_compat import apply_larch_patches
    assert apply_larch_patches() in (True, False)


# ---------------------------------------------------------------------------
# mixed logit: distributions, panels, isvars, engine (larch + jax needed)
# ---------------------------------------------------------------------------

def _panel_mixed_data(n_people=120, n_tasks=4, seed=7):
    """Panel data: random ``time`` coefficient (person-level), fixed ``x``,
    and an individual-specific ``inc`` with alternative-specific effects."""
    rng = np.random.default_rng(seed)
    rows, case = [], 0
    for p in range(n_people):
        b_time = -1.0 + 0.8 * rng.normal()
        inc = rng.normal()
        for _ in range(n_tasks):
            x = rng.normal(size=3)
            t = rng.uniform(0, 2, size=3)
            v = 0.5 * x + b_time * t + np.array([0.0, 0.5 * inc, -0.4 * inc])
            ch = int(np.argmax(v + rng.gumbel(size=3)))
            for j in range(3):
                rows.append((p, case, j, x[j], t[j], inc, int(j == ch)))
            case += 1
    return pd.DataFrame(rows, columns=["pid", "cid", "alt", "x", "time", "inc", "ch"])


def _fit_mixed(df, **kw):
    from SearchLibrium.larch_models import LarchMixedLogit
    args = dict(X=df[["x", "time", "inc"]].to_numpy(), y=df["ch"].to_numpy(),
                varnames=["x", "time", "inc"], isvars=["inc"],
                alts=df["alt"].to_numpy(), ids=df["cid"].to_numpy(),
                panels=df["pid"].to_numpy(), n_draws=50,
                draw_method="halton", maxiter=200)
    args.update(kw)
    model = LarchMixedLogit().setup(**args)
    model.fit()
    return model


def _needs_jax():
    return pytest.mark.skipif(
        __import__("importlib").util.find_spec("jax") is None,
        reason="Larch mixtures are only evaluated by the JAX engine")


@_needs_larch()
@_needs_jax()
@pytest.mark.parametrize("dist", ["n", "ln", "nln", "tn", "u", "t"])
def test_larch_mixed_every_search_distribution_is_estimated(dist):
    """Every distribution code the search can propose must reach Larch as a
    real mixture on ``B_<var>`` (mean) + ``sigma_<var>`` (spread).

    Regression: ``u`` used ``lx.mixtures.Uniform`` (which randomises its *low*
    parameter, never ``B_time``), ``t`` failed to fit and ``tn`` silently fell
    back to Normal.
    """
    df = _panel_mixed_data()
    model = _fit_mixed(df, randvars={"time": dist})
    names = [str(n) for n in model.coeff_names]
    assert hasattr(model, "_larch_model"), "fit fell into _fail"
    assert np.isfinite(model.loglik)
    assert {"B_time", "sigma_time"} <= set(names)
    assert not any(n.startswith(("low_", "high_")) for n in names)
    mix = model._larch_model.mixtures[0]
    assert mix.to_dict()["dist"] == dist
    # the spread must actually be estimated (it moves off its 0.1 start) with
    # a finite standard error -- including 'tn' on this negative coefficient
    if dist in {"n", "u", "t", "tn"}:
        sigma = abs(dict(zip(names, model.coeff_est))["sigma_time"])
        assert sigma > 0.2
        se = dict(zip(names, model.stderr))
        assert np.isfinite(se["sigma_time"]) and np.isfinite(se["B_time"])


@_needs_larch()
@_needs_jax()
def test_larch_mixed_tn_truncates_on_the_coefficients_sign():
    """'tn' on a negative coefficient (time) must truncate at zero from above
    (min(., 0)), not pin the draws at zero with max(., 0); the sign chosen in
    training is carried into a warm-start refit (the MAE path)."""
    df = _panel_mixed_data()
    normal = _fit_mixed(df, randvars={"time": "n"})
    tn = _fit_mixed(df, randvars={"time": "tn", "x": "tn"})
    assert tn.tn_signs == {"B_time": -1.0, "B_x": 1.0}
    signs = {m.target_: m.sign_ for m in tn._larch_model.mixtures}
    assert signs == {"B_time": -1.0, "B_x": 1.0}
    assert dict(zip(tn.coeff_names, tn.coeff_est))["B_time"] < 0
    # no worse than a few log-points from the untruncated normal fit
    assert tn._larch_model.loglike() > normal._larch_model.loglike() - 5.0

    refit = _fit_mixed(df, randvars={"time": "tn", "x": "tn"},
                       init_coeff=tn.coeff_est, maxiter=0,
                       tn_signs=tn.tn_signs)
    assert refit.tn_signs == tn.tn_signs
    np.testing.assert_allclose(refit._larch_model.loglike(),
                               tn._larch_model.loglike(), rtol=1e-4)


@_needs_larch()
@_needs_jax()
def test_larch_mixed_panel_isvars_and_simulated_shares():
    df = _panel_mixed_data()
    model = _fit_mixed(df, randvars={"time": "n"})
    names = set(str(n) for n in model.coeff_names)
    # isvars get alternative-specific coefficients (identified), not B_inc
    assert {"B_inc_1", "B_inc_2"} <= names and "B_inc" not in names
    assert model.n_panels == df["pid"].nunique()
    assert model._larch_model._draws.shape == (model.n_panels, 50, 1)
    probs = model.simulated_probability()
    assert probs.shape == (df["cid"].nunique(), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(model.pred_prob, model.obs_prob, atol=0.03)
    se = dict(zip(model.coeff_names, model.stderr))
    assert np.isfinite(se["B_inc_1"]) and np.isfinite(se["sigma_time"])


@_needs_larch()
@_needs_jax()
def test_larch_mixed_random_isvar_per_alternative_key():
    df = _panel_mixed_data()
    model = _fit_mixed(df, randvars={"time": "n", "inc.1": "n"})
    names = set(str(n) for n in model.coeff_names)
    assert {"sigma_time", "sigma_inc_1"} <= names
    assert "sigma_inc_2" not in names


@_needs_larch()
@_needs_jax()
def test_larch_mixed_never_falls_back_to_numba():
    """The numba engine ignores ``mixtures`` (it would fit an MNL whose sigma
    never moves), so a numba request must still be estimated on JAX."""
    df = _panel_mixed_data()
    model = _fit_mixed(df, randvars={"time": "n"}, compute_engine="numba")
    assert model._larch_model.compute_engine == "jax"
    sigma = abs(dict(zip(model.coeff_names, model.coeff_est))["sigma_time"])
    assert sigma > 0.2


@_needs_larch()
@_needs_jax()
def test_larch_mixed_correlated_block_uses_its_own_draw_columns():
    """A correlated block mixed with an independent random parameter: the
    block's members must read the block's draw columns, not column 0..k-1
    regardless of position."""
    df = _panel_mixed_data()
    df["z"] = np.random.default_rng(0).normal(size=len(df))
    model = _fit_mixed(
        df, X=df[["x", "time", "z", "inc"]].to_numpy(),
        varnames=["x", "time", "z", "inc"],
        randvars={"x": "u", "time": "n", "z": "n"}, corvars=["time", "z"])
    mixes = model._larch_model.mixtures
    assert [getattr(m, "draw_cols", None) for m in mixes[:2]] == [[0, 1], [0, 1]]
    assert mixes[2].to_dict()["target"] == "B_x"
    names = set(str(n) for n in model.coeff_names)
    assert {"chol.time.time", "chol.z.time", "chol.z.z", "sigma_x"} <= names
    assert np.isfinite(model.loglik)


# ---------------------------------------------------------------------------
# search perturbations keep the random-parameter spec consistent
# ---------------------------------------------------------------------------

def _search_for_perturbations(models=("larch_mnl", "larch_mixed")):
    from SearchLibrium.search import Parameters, Search
    df = pd.DataFrame({"x": [0.5, 0.2, 0.3, 0.8], "y": [1.0, 0.0, 2.0, 1.0]})
    p = Parameters(
        criterions=[("bic", -1)], df=df, varnames=["x", "y"],
        asvarnames=["x", "y"], isvarnames=[], choice_set=[0, 1],
        choices=np.array([1, 0, 1, 0], dtype="int32"),
        alt_var=np.array([1, 0, 1, 0], dtype="int32"),
        choice_id=np.array([0, 0, 1, 1], dtype="int32"),
        ind_id=np.array([0, 0, 1, 1], dtype="int32"),
        models=list(models), auto_as_is=False, allow_corvars=True,
        distr=["n", "u", "f"])
    return Search(p)


def test_change_distribution_fixed_drops_randvar_and_realigns_model():
    s = _search_for_perturbations()
    sol = {"asvars": ["x", "y"], "isvars": [], "randvars": {"x": "n", "y": "n"},
           "corvars": ["x", "y"], "model_n": "larch_mixed"}
    s.change_distribution("x", "u", sol)
    assert sol["randvars"] == {"x": "u", "y": "n"}
    assert sol["corvars"] == ["y"]
    s.change_distribution("x", "f", sol)
    s.change_distribution("y", "f", sol)
    assert sol["randvars"] == {}
    assert sol["model_n"] == "larch_mnl"


def test_agds_sync_random_spec_after_operators():
    from SearchLibrium.sparseea_agds import SparseEAAGDS
    s = _search_for_perturbations()
    # a child that picked up a random parameter by crossover but kept a
    # fixed-coefficient model, plus a corvar that is no longer Normal
    child = {"asvars": ["x", "y"], "isvars": [], "randvars": {"x": "u", "y": "n"},
             "corvars": ["x", "y"], "model_n": "larch_mnl"}
    SparseEAAGDS._sync_random_spec(s, child)
    assert child["model_n"] == "larch_mixed"
    assert child["corvars"] == []
