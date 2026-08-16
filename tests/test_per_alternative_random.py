"""Regression tests for PER-ALTERNATIVE random coefficients on individual-
specific variables.

An individual-specific variable expands into one column per non-base
alternative (e.g. ``alone.bike``, ``alone.micro``). Historically a random
coefficient keyed by the base name (``{'alone': 'n'}``) forced a random SD on
EVERY alternative at once, spraying non-significant ``sd.<var>.<alt>`` moments.
Random keys may now be scoped to a single alternative via ``"var.alt"``:

    randvars={'alone.bike': 'n'}   ->  only sd.alone.bike is estimated

The base-name key keeps its legacy "random on every alternative" meaning.
"""
import os
os.environ.setdefault("SL_QUIET", "1")
import numpy as np
import pandas as pd
import pytest

from SearchLibrium import MixedLogit, Parameters

ALTS = ["walk", "bike", "micro"]


def _synth(n=400, seed=7):
    rng = np.random.default_rng(seed)
    alone = rng.integers(0, 2, n).astype(float)
    male = rng.integers(0, 2, n).astype(float)
    u_bike = -0.5 + 0.8 * alone - 0.3 * male + rng.gumbel(size=n)
    u_micro = -0.8 - 1.0 * alone + 0.5 * male + rng.gumbel(size=n)
    u_walk = rng.gumbel(size=n)
    ch = np.argmax(np.column_stack([u_walk, u_bike, u_micro]), axis=1)
    rows = []
    for i in range(n):
        for j, a in enumerate(ALTS):
            rows.append({"obs_id": i + 1, "alt": a, "choice": int(ch[i] == j),
                         "alone": alone[i], "male": male[i]})
    return pd.DataFrame(rows)


def _fit(randvars):
    df = _synth()
    varnames = ["alone", "male"]
    m = MixedLogit()
    m.setup(X=df[varnames].values.astype(np.float64),
            y=df["choice"].values.astype(np.int32),
            varnames=varnames, alts=df["alt"].values.astype(object),
            isvars=varnames, randvars=randvars,
            panels=df["obs_id"].values.astype(np.int32),
            base_alt="walk", n_draws=200, halton=True, mnl_init=True,
            reg_penalty=0.05, maxiter=1000, ftol=1e-6, gtol=1e-6)
    m.fit()
    return [str(n) for n in m.coeff_names], m


def test_base_name_key_is_random_on_all_alternatives():
    names, m = _fit({"alone": "n"})
    sd = sorted(n for n in names if n.startswith("sd."))
    assert sd == ["sd.alone.bike", "sd.alone.micro"]
    assert bool(getattr(m, "converged", True))


def test_per_alternative_key_bike_only():
    names, m = _fit({"alone.bike": "n"})
    sd = [n for n in names if n.startswith("sd.")]
    assert sd == ["sd.alone.bike"]
    assert "alone.micro" in names          # still present as a FIXED coefficient
    assert "sd.alone.micro" not in names


def test_per_alternative_key_micro_only():
    names, _ = _fit({"alone.micro": "n"})
    sd = [n for n in names if n.startswith("sd.")]
    assert sd == ["sd.alone.micro"]
    assert "sd.alone.bike" not in names


def test_search_pool_is_per_alternative_for_isvars():
    df = _synth()
    varnames = ["alone", "male"]
    p = Parameters(
        criterions=[("bic", -1), ("nsig", -1)], df=df, varnames=varnames,
        isvarnames=varnames, asvarnames=[], choice_set=ALTS,
        choices=df["choice"].values, alt_var=df["alt"].values,
        choice_id=df["obs_id"].values, ind_id=df["obs_id"].values,
        base_alt="walk", models=["multinomial", "mixed_logit"], distr=["n", "f"],
        allow_random=True, allow_random_isvars=True, n_draws=100, p_val=0.05,
        all_sig=False)
    # Per-alternative keys for every isvar x non-base alternative, and NO bare
    # base-name isvar entries.
    assert set(p.avail_rvars) == {
        "alone.bike", "alone.micro", "male.bike", "male.micro"}


def test_asvar_only_pool_unchanged():
    # Backward compatibility: with allow_random_isvars=False the random pool is
    # exactly the asvars (no per-alternative expansion).
    df = _synth()
    p = Parameters(
        criterions=[("bic", -1), ("nsig", -1)], df=df, varnames=["male"],
        isvarnames=[], asvarnames=["male"], choice_set=ALTS,
        choices=df["choice"].values, alt_var=df["alt"].values,
        choice_id=df["obs_id"].values, ind_id=df["obs_id"].values,
        base_alt="walk", models=["multinomial", "mixed_logit"], distr=["n", "f"],
        allow_random=True, allow_random_isvars=False, n_draws=100, p_val=0.05,
        all_sig=False)
    assert p.avail_rvars == ["male"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
