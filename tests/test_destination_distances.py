"""Tests for SearchLibrium.destination_distances (larch-style case x alt distances)."""

import numpy as np
import pandas as pd

from SearchLibrium.destination_distances import (
    build_case_alt_distances,
    haversine_km,
    load_skim_matrices,
    pick_skim_matrices,
    summarize_distance_sources,
)


def _toy_system():
    cases = pd.DataFrame({
        "case_id": [0, 1],
        "orig_zone": ["A", "B"],
        "orig_lat": [-27.5, -27.6],
        "orig_lon": [153.0, 153.1],
    })
    alts = pd.DataFrame({
        "alt_id": [1, 2, 3],
        "dest_zone": ["A", "B", "C"],
        "dest_lat": [-27.5, -27.6, -27.7],
        "dest_lon": [153.0, 153.1, 153.2],
        "distance_offset_km": [0.0, 0.0, 15.0],
    })
    return cases, alts


def test_haversine_known_distance():
    # Brisbane CBD -> Gold Coast approx 66 km crow-fly.
    d = float(haversine_km(-27.4698, 153.0251, -28.0167, 153.4000))
    assert 60.0 < d < 80.0


def test_pick_skim_matrices_walk():
    names = ["WH_NetworkDist_Walk", "WH_NetworkTime_Walk",
             "WH_NetworkDist_Bike", "WH_NetworkTime_Bike"]
    dist, time = pick_skim_matrices(names, "walk")
    assert dist == "WH_NetworkDist_Walk"
    assert time == "WH_NetworkTime_Walk"
    dist, _ = pick_skim_matrices(names, "bike")
    assert dist == "WH_NetworkDist_Bike"


def test_build_pure_haversine():
    cases, alts = _toy_system()
    long = build_case_alt_distances(cases, alts)
    assert len(long) == 2 * 3
    assert set(long["dist_source"].unique()) == {"haversine"}
    # Intra-case: distance to own zone ~ floor-free haversine (~0 km).
    own = long[(long["case_id"] == 0) & (long["alt_id"] == 1)].iloc[0]
    assert own["dist_km"] < 0.01
    # Pooled-style offset respected (alt 3 carries +15 km).
    off = long[(long["case_id"] == 0) & (long["alt_id"] == 3)].iloc[0]
    base = float(haversine_km(-27.5, 153.0, -27.7, 153.2))
    assert off["dist_km"] == base + 15.0
    assert np.isfinite(long["ln_dist"]).all()


def test_build_skim_with_fallback():
    cases, alts = _toy_system()
    zone_ids = ["A", "B", "C"]
    # Skim covers A<->B only; C unreachable (NaN) -> haversine fallback.
    sd = np.array([[0.5, 4.0, np.nan],
                   [4.0, 0.5, np.nan],
                   [np.nan, np.nan, np.nan]])
    long = build_case_alt_distances(
        cases, alts, skim_zone_ids=zone_ids, skim_dist_mat=sd,
        min_zone_km=0.6,
    )
    got = {(r["case_id"], r["alt_id"]): (r["dist_km"], r["dist_source"])
           for _, r in long.iterrows()}
    # Skim used where available, with intra-zone floor (0.5 -> 0.6).
    assert got[(0, 1)] == (0.6, "skim")
    assert got[(0, 2)][0] == 4.0 and got[(0, 2)][1] == "skim"
    # Missing skim pair -> haversine (+ offset for alt 3).
    assert got[(0, 3)][1] == "haversine"
    assert got[(1, 3)][1] == "haversine"
    summary = summarize_distance_sources(long)
    assert summary["n_pairs"] == 6
    assert 0.0 < summary["skim_share"] < 1.0


def test_load_skim_matrices_roundtrip(tmp_path):
    h5py = __import__("pytest").importorskip("h5py")
    import h5py as _h5
    p = tmp_path / "toy.omx"
    zones = ["A", "B"]
    with _h5.File(str(p), "w") as f:
        f.attrs["OMX_VERSION"] = "0.2"
        g = f.create_group("data")
        g.create_dataset("WH_NetworkDist_Walk",
                         data=np.array([[0.5, 4.0], [4.0, 0.5]], dtype=np.float32))
        g.create_dataset("WH_NetworkTime_Walk",
                         data=np.array([[1.0, 8.0], [8.0, 1.0]], dtype=np.float32))
        look = f.create_group("lookup")
        look.create_dataset("SUBURB", data=np.array(zones, dtype="S64"))
    zids, dist, time, dname, tname = load_skim_matrices(str(p), style="walk")
    assert zids == ["A", "B"]
    assert dname == "WH_NetworkDist_Walk"
    assert dist.shape == (2, 2)
    assert time is not None


def test_load_skim_matrices_missing_is_none():
    assert load_skim_matrices("", style="walk") == (None, None, None, None, None)
    assert load_skim_matrices("/nonexistent/path.omx") == (None, None, None, None, None)
