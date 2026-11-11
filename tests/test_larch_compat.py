"""Regression tests for the promoted larch compatibility patches.

Covers the two LCCM fixes tracked in GitHub issue #1:

1. graph-None guards on ``NumbaModel._rebuild_fixed_arrays`` /
   ``_rebuild_work_arrays``;
2. ``LatentClass.__init__`` membership datatree wiring for ``DataTree``
   inputs, including the cross-copy marker contract with the tour
   pipeline's ``larch_patches.py`` (stand-in: a second copy of this module).
"""
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _larch_importable():
    return importlib.util.find_spec("larch") is not None


pytestmark = pytest.mark.skipif(
    not _larch_importable(), reason="larch not installed")


def _apply():
    from SearchLibrium.larch_compat import apply_larch_patches
    assert apply_larch_patches() is True


def _toy_datatree(n=80, alts=(1, 2), seed=0):
    """Return an idca Dataset wrapped in a single-node DataTree."""
    import larch as lx

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = rng.normal(size=len(alts))
        chosen = int(np.argmax(x + rng.gumbel(size=len(alts))))
        for j, alt in enumerate(alts):
            rows.append((i, alt, x[j], 1 if j == chosen else 0))
    frame = pd.DataFrame(
        rows, columns=["case", "alt", "x", "ch"]
    ).set_index(["case", "alt"])
    ds = lx.Dataset.construct.from_idca(frame)
    return lx.DataTree(main=ds), ds


def test_larch_latent_class_is_public():
    """The first-class LCCM wrapper is exported at package level."""
    from SearchLibrium import LarchLatentClass
    from SearchLibrium.larch_models import LarchLatentClass as Direct

    assert LarchLatentClass is Direct


def test_graph_none_rebuild_guards_installed():
    _apply()
    from larch.model.numbamodel import NumbaModel

    for name in ("_rebuild_fixed_arrays", "_rebuild_work_arrays"):
        assert getattr(getattr(NumbaModel, name), "_graph_none_guard", False)
    # A graphless model must survive a rebuild call instead of raising
    # "'NoneType' object has no attribute 'node_slot_arrays'".
    import larch as lx

    model = lx.Model()
    assert getattr(model, "graph", None) is None
    model._rebuild_fixed_arrays()
    model._rebuild_work_arrays()


def test_latent_class_membership_wired_for_datatree_input():
    """Issue #1 acceptance test (a) + (b)."""
    _apply()
    import larch as lx
    from larch import P

    tree, ds = _toy_datatree()
    class_models = {}
    for k, name in ((1, "c1"), (2, "c2")):
        cm = lx.Model(ds)
        cm.choice_ca_var = "ch"
        cm.utility_co[2] = P(name)
        class_models[k] = cm
    membership = lx.Model()
    membership.utility_co[2] = P("g")

    latent = lx.LatentClass(
        membership, class_models, datatree=tree)

    assert latent.class_membership_model.datatree is not None
    ll0 = float(latent.loglike())
    assert np.isfinite(ll0)

    # Parameters must actually move the likelihood (no frozen/datatreeless
    # membership path).
    latent.pvals = np.asarray(latent.pvals, dtype=float) + 0.3
    ll1 = float(latent.loglike())
    assert np.isfinite(ll1)
    assert abs(ll1 - ll0) > 1e-6


def test_apply_is_idempotent_for_latent_class_init():
    _apply()
    import larch as lx

    init_before = lx.LatentClass.__init__
    _apply()
    assert lx.LatentClass.__init__ is init_before


def test_marker_contract_with_second_patch_copy():
    """Issue #1 acceptance test (c): a second copy must not double-wrap.

    The tour pipeline's ``larch_patches.py`` shares the
    ``_graph_none_guard`` / ``_qut_membership_wiring`` markers.  A second
    copy of this module is a faithful stand-in: load it under the
    SearchLibrium package so its relative imports resolve, apply it, then
    apply the package copy in the same fresh interpreter.
    """
    repo = Path(__file__).resolve().parents[1]
    src = str(repo / "src")
    code = textwrap.dedent(
        """
        import importlib.util
        import larch as lx
        from SearchLibrium import larch_compat as slc

        spec = importlib.util.spec_from_file_location(
            "SearchLibrium._larch_compat_second_copy", slc.__file__)
        second = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(second)
        assert second.apply_larch_patches() is True

        init_before = lx.LatentClass.__init__
        assert slc.apply_larch_patches() is True
        print("REVERSE_OK" if lx.LatentClass.__init__ is init_before
              else "REVERSE_BAD")
        """
    )
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, timeout=300)
    assert "REVERSE_OK" in proc.stdout, (
        f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}")
