"""Numerical-equivalence gate for the masked (compile-once) JAX negloglik.

The masked path (_jax_mxl_negloglik_masked) pads the design to a fixed (KF, KR)
and zeroes inactive columns with masks, so ONE JIT compilation serves every
spec. It must be numerically identical to the per-shape path
(_jax_mxl_negloglik) it replaces — in both the log-likelihood value and its
gradient — for the supported (uncorrelated, no-heterogeneity, no-Box-Cox) case.

We build several small mixed-logit designs, evaluate both functions at random
beta points with the SAME draws, and (crucially) with the masked path PADDED to
larger (KF, KR) than the spec needs, so the masking/padding is actually exercised.
"""
import os
os.environ.setdefault("SL_QUIET", "1")
os.environ.setdefault("JAX_ENABLE_X64", "True")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from SearchLibrium.MixedLogit import MixedLogit

_CODE = {'n': 0, 'ln': 1, 'nln': 2, 'tn': 3, 'u': 4}


def _orig(b, X, y, pi, draws, fxidx, rvidx, Kf, Kr, rvdist):
    return MixedLogit._jax_mxl_negloglik(
        b, X, y, pi, draws, jnp.asarray(fxidx, bool), jnp.asarray(rvidx, bool),
        Kf, Kr, 0, Kr, list(rvdist), 0)


def _masked(bpad, X, y, pi, draws_pad, mf, mr, codes, KF, KR):
    return MixedLogit._jax_mxl_negloglik_masked(
        bpad, X, y, pi, draws_pad, jnp.asarray(mf), jnp.asarray(mr),
        jnp.asarray(codes), KF, KR)


def _synth(N=60, P=2, J=4, Kf=3, Kr=2, R=40, seed=0):
    rng = np.random.default_rng(seed)
    K = Kf + Kr
    X = rng.normal(size=(N, P, J, K))
    # one chosen alt per (n, p)
    y = np.zeros((N, P, J))
    for n in range(N):
        for p in range(P):
            y[n, p, rng.integers(J)] = 1.0
    pi = np.ones((N, P))
    draws = rng.normal(size=(N, Kr, R))
    # design order is [fixed cols, random cols]
    fxidx = np.array([True] * Kf + [False] * Kr)
    rvidx = np.array([False] * Kf + [True] * Kr)
    return (jnp.asarray(X), jnp.asarray(y), jnp.asarray(pi), jnp.asarray(draws),
            fxidx, rvidx, Kf, Kr)


def _pad(b, X, draws, fxidx, rvidx, Kf, Kr, rvdist, KF, KR):
    Xn = np.asarray(X); dn = np.asarray(draws)
    N, P, J, _ = Xn.shape; R = dn.shape[2]
    X_pad = np.zeros((N, P, J, KF + KR))
    X_pad[:, :, :, :Kf] = Xn[:, :, :, fxidx]
    X_pad[:, :, :, KF:KF + Kr] = Xn[:, :, :, rvidx]
    draws_pad = np.zeros((N, KR, R)); draws_pad[:, :Kr, :] = dn[:, :Kr, :]
    mf = np.zeros(KF); mf[:Kf] = 1.0
    mr = np.zeros(KR); mr[:Kr] = 1.0
    codes = np.zeros(KR); codes[:Kr] = [_CODE[d] for d in rvdist]
    bn = np.asarray(b)
    bpad = np.zeros(KF + 2 * KR)
    bpad[:Kf] = bn[:Kf]
    bpad[KF:KF + Kr] = bn[Kf:Kf + Kr]
    bpad[KF + KR:KF + KR + Kr] = bn[Kf + Kr:Kf + 2 * Kr]
    active = list(range(Kf)) + list(range(KF, KF + Kr)) + list(range(KF + KR, KF + KR + Kr))
    return (jnp.asarray(X_pad), jnp.asarray(draws_pad), mf, mr, codes,
            jnp.asarray(bpad), active)


@pytest.mark.parametrize("rvdist", [['n', 'n'], ['n', 'ln'], ['nln', 'tn'], ['u', 'n']])
def test_value_and_grad_match(rvdist):
    Kf, Kr = 3, 2
    X, y, pi, draws, fxidx, rvidx, Kf, Kr = _synth(Kf=Kf, Kr=Kr)
    rng = np.random.default_rng(1)
    KF, KR = Kf + 3, Kr + 2          # force padding so masking is exercised
    vg_o = jax.value_and_grad(lambda b: _orig(b, X, y, pi, draws, fxidx, rvidx, Kf, Kr, rvdist))
    for t in range(4):
        b = jnp.asarray(rng.normal(size=Kf + 2 * Kr) * 0.5)
        v_o, g_o = vg_o(b)
        X_pad, draws_pad, mf, mr, codes, bpad, active = _pad(
            b, X, draws, fxidx, rvidx, Kf, Kr, rvdist, KF, KR)
        vg_m = jax.value_and_grad(lambda bb: _masked(bb, X_pad, y, pi, draws_pad, mf, mr, codes, KF, KR))
        v_m, g_m = vg_m(bpad)
        assert np.isfinite(float(v_o)) and np.isfinite(float(v_m))
        assert abs(float(v_o) - float(v_m)) < 1e-8, (rvdist, float(v_o), float(v_m))
        g_m_active = np.asarray(g_m)[active]
        assert np.allclose(np.asarray(g_o), g_m_active, atol=1e-7), (
            rvdist, np.asarray(g_o), g_m_active)


def test_no_padding_is_identity():
    # KF==Kf, KR==Kr : masked path with no padding must equal the original.
    Kf, Kr = 4, 3
    X, y, pi, draws, fxidx, rvidx, Kf, Kr = _synth(Kf=Kf, Kr=Kr, seed=3)
    rvdist = ['n', 'ln', 'n']
    b = jnp.asarray(np.random.default_rng(2).normal(size=Kf + 2 * Kr) * 0.4)
    v_o = float(_orig(b, X, y, pi, draws, fxidx, rvidx, Kf, Kr, rvdist))
    X_pad, draws_pad, mf, mr, codes, bpad, _ = _pad(
        b, X, draws, fxidx, rvidx, Kf, Kr, rvdist, Kf, Kr)
    v_m = float(_masked(bpad, X_pad, y, pi, draws_pad, mf, mr, codes, Kf, Kr))
    assert abs(v_o - v_m) < 1e-9, (v_o, v_m)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
