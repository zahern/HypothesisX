"""Tests for latent-class smart initialisation and Hessian hardening.

Covers ``LatentClassMixedLogit`` smart starts (pooled MNL + random panel
partitions), descending-share sorting (label-switch fix, incl. membership
gamma re-referencing), collapse detection, deterministic seeding, the
scale-aware finite-difference Hessian, and the OPG standard-error fallback.
Runs on the numpy paths (``_jax=False``) so no GPU/JAX build is needed.
"""
import numpy as np
import pytest

from SearchLibrium.latent_class import LatentClassMixedLogit


def _synth(seed=0, n_panels=90, tasks=3):
    """Two well-separated classes, J=2 alts, K=2 vars (x + asc1)."""
    rng = np.random.default_rng(seed)
    betas = [np.array([1.5, -1.0]), np.array([-1.2, 1.2])]
    X_rows, y_rows, ids, alts, panels = [], [], [], [], []
    tid = 0
    for p in range(n_panels):
        c = p % 2
        for _t in range(tasks):
            x = rng.normal(size=2)
            Xa = np.array([[x[0], 0.0], [x[1], 1.0]])
            u = Xa @ betas[c] + rng.gumbel(size=2)
            ch = int(np.argmax(u))
            for j in range(2):
                X_rows.append(Xa[j])
                y_rows.append(1 if j == ch else 0)
                ids.append(tid)
                alts.append(j)
                panels.append(p)
            tid += 1
    return (np.asarray(X_rows), np.asarray(y_rows), np.asarray(ids),
            np.asarray(alts), np.asarray(panels))


def _model(**kw):
    X, y, ids, alts, panels = _synth()
    m = LatentClassMixedLogit(n_classes=2, maxiter=20, class_maxiter=30,
                              random_state=0, _jax=False, **kw)
    m.setup(X=X, y=y, varnames=["x", "asc1"], ids=ids, alts=alts, panels=panels)
    return m


# ---------------------------------------------------------------------------
# smart_init
# ---------------------------------------------------------------------------

def test_smart_init_fits_sorts_and_tracks_starts():
    m = _model(smart_init=True, n_init=2)
    m.fit()
    assert np.isfinite(m.loglik)
    # shares sorted descending (label-switch fix)
    assert m.class_probs[0] >= m.class_probs[1]
    assert m.coeff_names[0].startswith("class_1_")
    # start diagnostics recorded
    assert len(m.starts_) == 2
    assert all(s["kind"] == "smart" for s in m.starts_)
    assert m.smart_init_used_
    # EM monotonicity from each smart start
    for s in m.starts_:
        assert s["final_loglik"] >= s["start_loglik"] - 1e-6
    # collapse bookkeeping present
    assert isinstance(m.collapsed_, list)
    assert np.isfinite(m.min_class_share_)


def test_smart_init_deterministic():
    a = _model(smart_init=True, n_init=2)
    a.fit()
    b = _model(smart_init=True, n_init=2)
    b.fit()
    np.testing.assert_allclose(a.coeff_est, b.coeff_est)
    np.testing.assert_allclose(a.class_probs, b.class_probs)


def test_random_stream_unchanged_when_smart_off():
    # Legacy N(0, 0.05) path must not consume extra rng draws.
    a = _model(smart_init=False, n_init=1)
    a.fit()
    b = _model(smart_init=False, n_init=1)
    b.fit()
    np.testing.assert_allclose(a.coeff_est, b.coeff_est)


def test_sort_permutes_betas_posterior_and_gammas():
    m = _model()
    rng = np.random.default_rng(0)
    b0 = np.array([1.0, 2.0])
    b1 = np.array([3.0, 4.0])
    m.class_betas = [b0.copy(), b1.copy()]
    m.class_probs = np.array([0.3, 0.7])
    post = rng.random((m.n_panels, 2))
    post /= post.sum(axis=1, keepdims=True)
    m.posterior = post.copy()
    # fake 1-D membership equation to exercise gamma re-referencing
    m._has_membership = True
    m.K_membership = 1
    m.X_membership = rng.normal(size=(m.n_panels, 1))
    m.class_gammas = np.array([[0.5]])
    priors_before = m._compute_membership_priors(m.class_gammas)
    perm = m._sort_classes_in_place()
    np.testing.assert_array_equal(perm, [1, 0])
    np.testing.assert_allclose(m.class_probs, [0.7, 0.3])
    np.testing.assert_allclose(m.class_betas[0], b1)
    np.testing.assert_allclose(m.posterior, post[:, [1, 0]])
    # membership model identical under the new reference class
    priors_after = m._compute_membership_priors(m.class_gammas)
    np.testing.assert_allclose(priors_after, priors_before[:, [1, 0]], atol=1e-10)


def test_collapse_warning_is_raisable():
    m = _model(min_share=0.95)
    with pytest.warns(UserWarning, match="collapsed"):
        m.fit()


def test_fit_direct_accepts_smart_init():
    m = _model()
    m.fit_direct(smart_init=True, maxiter=50)
    assert np.isfinite(m.loglik)
    assert m.smart_init_used_
    assert m.class_probs[0] >= m.class_probs[1]


# ---------------------------------------------------------------------------
# Hessian hardening
# ---------------------------------------------------------------------------

def test_numerical_hessian_finite_and_scaled():
    m = _model(smart_init=True, n_init=1)
    m.fit()
    C = m.n_classes
    phi = np.log(np.clip(m.class_probs[:C - 1], 1e-300, None)) \
        - np.log(np.clip(m.class_probs[-1], 1e-300, None))
    params = np.concatenate([phi] + [np.asarray(b).ravel() for b in m.class_betas])
    H = m._numerical_hessian(params)
    assert np.all(np.isfinite(H))
    assert H.shape == (len(params), len(params))


def test_opg_fallback_on_collapsed_shares():
    m = _model(smart_init=True, n_init=1)
    m.fit()
    # Force a collapsed share, then recompute SEs: the gate must engage.
    m.class_probs = np.array([0.999, 0.001])
    with pytest.warns(UserWarning, match="[Oo][Pp][Gg]"):
        stats = m.compute_standard_errors()
    assert stats["se_method"].startswith("opg fallback")
    assert np.all(np.isfinite(stats["se"]))
    assert "se_reliable" in stats and "collapsed" in stats


def test_se_finite_on_identified_fit():
    m = _model(smart_init=True, n_init=2)
    m.fit()
    assert np.all(np.isfinite(m.stderr))
    assert np.all(np.isfinite(m.pvalues))


def test_parameters_forwards_lc_flags(capsys):
    from SearchLibrium.search import Parameters
    import pandas as pd
    df = pd.DataFrame({"asc_a": [1, 0, 1, 0], "x1": [0.5, 0.2, 0.3, 0.8]})
    p = Parameters(
        criterions=[("bic", -1)], df=df, varnames=["asc_a", "x1"],
        asvarnames=["asc_a", "x1"], isvarnames=[], choice_set=[0, 1],
        choices=np.array([1, 0, 1, 0], dtype="int32"),
        alt_var=np.array([1, 0, 1, 0], dtype="int32"),
        choice_id=np.array([0, 1, 2, 3], dtype="int32"),
        ind_id=np.array([0, 1, 2, 3], dtype="int32"),
        models=["multinomial"], auto_as_is=False,
        lc_smart_init=True, lc_n_init=3, lc_min_share=0.1, lc_sort_classes=True)
    assert (p.lc_smart_init, p.lc_n_init, p.lc_min_share, p.lc_sort_classes) == (True, 3, 0.1, True)
    assert "Unexpected keyword" not in capsys.readouterr().out
