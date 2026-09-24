"""Case-by-alternative destination distances, larch-idca style.

This module builds *proper* destination distances *respective to each
destination* for every case (trip), the way ``larch`` consumes them in
``idca`` (individual-choice-alternative) format:

* when both the case origin zone and the destination zone exist in a network
  skim matrix (OMX), the zone-to-zone skim distance is used;
* otherwise the distance falls back to haversine computed from the
  *case-specific* origin coordinates to the destination coordinates;
* an intra-zone floor keeps skim self/short pairs from collapsing to zero;
* a per-destination offset (e.g. for a pooled-external fallback alternative)
  is added after the floor, exactly as in ``larch`` utility preprocessing.

The result is a long ``(case_id, alt_id)`` frame whose columns plug straight
into :class:`SearchLibrium.multinomial_logit.MultinomialLogit`,
:class:`SearchLibrium.MixedLogit.MixedLogit` and
:class:`SearchLibrium.mh_choice_estimator.ChoiceSetFrame` as
alternative-specific (``asvar``) attributes — e.g. ``ln_dist`` — instead of a
single destination-constant proxy.

Only ``numpy``/``pandas`` are required; ``h5py`` is needed solely to read OMX
skim files (the light-weight ``OMX_VERSION`` layout written by the SEQ
walk/bike skim builder, ``/data/<matrix>`` + ``/lookup/<zone>``).
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Haversine (vectorised)
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (scalar- or array-like inputs)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Skim-matrix selection + loading
# ---------------------------------------------------------------------------

def pick_skim_matrices(available: list[str], style: str = "walk") -> tuple[str | None, str | None]:
    """Choose ``(dist, time)`` matrix names for ``style`` (``walk``/``bike``/``min``).

    Preference order mirrors the OD pipeline: style-specific network-distance
    matrices first (``WH_NetworkDist_Walk`` / ``WH_NetworkDist_Bike``), then any
    matrix carrying the style token, then any distance-like matrix, then the
    first available matrix as a last resort.
    """
    style = str(style or "walk").strip().lower()
    names = list(available or [])
    lower = {n: n.lower() for n in names}

    def _find(pred):
        for n in names:
            if pred(lower[n], n):
                return n
        return None

    dist_key, time_key = None, None
    if style in ("walk", "bike"):
        tok = style
        dist_key = _find(lambda nl, n: tok in nl and ("networkdist" in nl or "dist" in nl))
        time_key = _find(lambda nl, n: tok in nl and ("networktime" in nl or "time" in nl))
    elif style == "min":
        dist_key = _find(lambda nl, n: "dist" in nl and ("network" in nl or "wh_" in nl))
        time_key = _find(lambda nl, n: "time" in nl and ("network" in nl or "wh_" in nl))
    if dist_key is None:
        dist_key = _find(lambda nl, n: "dist" in nl) or (names[0] if names else None)
    return dist_key, time_key


def _decode_zone_ids(raw) -> list[str]:
    out = []
    for z in np.asarray(raw).ravel():
        if isinstance(z, (bytes, np.bytes_)):
            z = z.decode()
        out.append(str(z).strip().upper())
    return out


def load_skim_matrices(omx_path: str = "", style: str = "walk"):
    """Read ``(zone_ids, dist_mat, time_mat, dist_name, time_name)`` from OMX.

    Returns ``(None, None, None, None, None)`` when no path is given, the file
    is missing, or ``h5py`` is unavailable — callers then use pure haversine.
    """
    if not str(omx_path or ""):
        return None, None, None, None, None
    from pathlib import Path
    path = Path(omx_path)
    if not path.exists():
        logger.warning("destination_distances: OMX not found: %s — haversine fallback", path)
        return None, None, None, None, None
    try:
        import h5py
    except ImportError:
        logger.warning("destination_distances: h5py unavailable — haversine fallback")
        return None, None, None, None, None
    with h5py.File(str(path), "r") as f:
        if "data" not in f:
            logger.warning("destination_distances: no /data group in %s", path.name)
            return None, None, None, None, None
        names = list(f["data"].keys())
        zone_ids = None
        if "lookup" in f:
            for cand in ("SUBURB", "zone_id", "ZONE_ID", "SA2_CODE"):
                if cand in f["lookup"]:
                    try:
                        zone_ids = _decode_zone_ids(f["lookup"][cand][...])
                        break
                    except Exception:
                        zone_ids = None
        mats = {n: np.array(f["data"][n], dtype=np.float32) for n in names}
    if not mats:
        return None, None, None, None, None
    if zone_ids is not None and len(zone_ids) != next(iter(mats.values())).shape[0]:
        logger.warning("destination_distances: zone mapping length != matrix size — index fallback")
        zone_ids = None
    dist_name, time_name = pick_skim_matrices(names, style)
    dist_mat = mats.get(dist_name) if dist_name else None
    time_mat = mats.get(time_name) if time_name else None
    logger.info("destination_distances: %d zones from %s (dist=%s time=%s style=%s)",
                next(iter(mats.values())).shape[0], path.name, dist_name, time_name, style)
    return zone_ids, dist_mat, time_mat, dist_name, time_name


# ---------------------------------------------------------------------------
# Core builder: case x alternative distances
# ---------------------------------------------------------------------------

def build_case_alt_distances(
    cases: pd.DataFrame,
    alternatives: pd.DataFrame,
    skim_zone_ids: list | None = None,
    skim_dist_mat=None,
    skim_time_mat=None,
    *,
    style: str = "walk",
    min_zone_km: float = 0.6,
    max_km: float = 120.0,
    log_offset: float = 0.1,
    case_id_col: str = "case_id",
    orig_zone_col: str = "orig_zone",
    orig_lat_col: str = "orig_lat",
    orig_lon_col: str = "orig_lon",
    alt_id_col: str = "alt_id",
    dest_zone_col: str = "dest_zone",
    dest_lat_col: str = "dest_lat",
    dest_lon_col: str = "dest_lon",
    offset_col: str | None = "distance_offset_km",
) -> pd.DataFrame:
    """Build larch-style per-(case, alternative) distances.

    Parameters
    ----------
    cases : DataFrame with one row per case (``case_id``, origin zone label,
        origin coordinates). Origin coordinates are case-specific (route
        waypoint / zone centroid / trip coordinate — whichever the caller
        resolved), so haversine fallback is respective to the case.
    alternatives : DataFrame with one row per destination (``alt_id``,
        destination zone label, destination coordinates, optional per-dest
        ``distance_offset_km`` for pooled-external fallbacks).
    skim_zone_ids / skim_dist_mat / skim_time_mat : as returned by
        :func:`load_skim_matrices` (all ``None`` = pure haversine).
    min_zone_km : intra-zone floor applied to *skim* distances only.
    max_km : skim pairs outside ``(0, max_km)`` are treated as unusable and
        fall back to haversine.
    log_offset : ``ln_dist = log(dist_km + log_offset)``.

    Returns
    -------
    Long DataFrame ``[case_id, alt_id, dist_km, skim_dist_km, skim_time_min,
    dist_source, ln_dist]`` where ``dist_source`` is ``'skim'`` or
    ``'haversine'``. Distances are respective to the destination: every
    ``(case, alt)`` pair carries its own value, exactly what ``larch``
    ``idca`` consumers expect.
    """
    req_cases = [case_id_col, orig_zone_col, orig_lat_col, orig_lon_col]
    req_alts = [alt_id_col, dest_zone_col, dest_lat_col, dest_lon_col]
    missing = [c for c in req_cases if c not in cases.columns]
    if missing:
        raise KeyError(f"build_case_alt_distances: cases missing columns {missing}")
    missing = [c for c in req_alts if c not in alternatives.columns]
    if missing:
        raise KeyError(f"build_case_alt_distances: alternatives missing columns {missing}")

    c = cases[req_cases].copy()
    a = alternatives[req_alts + ([offset_col] if offset_col and offset_col in alternatives.columns else [])].copy()
    c["_oz"] = c[orig_zone_col].astype(str).str.strip().str.upper()
    a["_dz"] = a[dest_zone_col].astype(str).str.strip().str.upper()
    if offset_col and offset_col in a.columns:
        a["_off"] = pd.to_numeric(a[offset_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        a["_off"] = 0.0

    case_ids = c[case_id_col].to_numpy()
    alt_ids = a[alt_id_col].to_numpy()
    n_c, n_a = len(c), len(a)

    o_lat = pd.to_numeric(c[orig_lat_col], errors="coerce").to_numpy(dtype=float)
    o_lon = pd.to_numeric(c[orig_lon_col], errors="coerce").to_numpy(dtype=float)
    d_lat = pd.to_numeric(a[dest_lat_col], errors="coerce").to_numpy(dtype=float)
    d_lon = pd.to_numeric(a[dest_lon_col], errors="coerce").to_numpy(dtype=float)
    offsets = a["_off"].to_numpy(dtype=float)

    # Skim index lookup (zone label -> matrix row).
    skim_lookup: dict[str, int] = {}
    if skim_zone_ids is not None and skim_dist_mat is not None:
        skim_lookup = {z: i for i, z in enumerate(skim_zone_ids)}
    o_idx = np.array([skim_lookup.get(z, -1) for z in c["_oz"].to_numpy()], dtype=int)
    d_idx = np.array([skim_lookup.get(z, -1) for z in a["_dz"].to_numpy()], dtype=int)

    skim_d = np.full((n_c, n_a), np.nan, dtype=float)
    skim_t = np.full((n_c, n_a), np.nan, dtype=float)
    if skim_dist_mat is not None and (o_idx >= 0).any() and (d_idx >= 0).any():
        sd = np.asarray(skim_dist_mat, dtype=float)
        st = np.asarray(skim_time_mat, dtype=float) if skim_time_mat is not None else None
        ok_c = np.where(o_idx >= 0)[0]
        ok_a = np.where(d_idx >= 0)[0]
        sub = sd[np.ix_(o_idx[ok_c], d_idx[ok_a])]
        usable = (sub > 0.0) & (sub < float(max_km))
        block = np.where(usable, sub, np.nan)
        skim_d[np.ix_(ok_c, ok_a)] = block
        if st is not None and st.shape == sd.shape:
            sub_t = st[np.ix_(o_idx[ok_c], d_idx[ok_a])]
            skim_t[np.ix_(ok_c, ok_a)] = np.where(usable, sub_t, np.nan)

    # Haversine fallback from case-specific origin coords (broadcast).
    with np.errstate(invalid="ignore"):
        hav = haversine_km(
            o_lat[:, None], o_lon[:, None],
            d_lat[None, :], d_lon[None, :],
        )
    hav = np.asarray(hav, dtype=float)

    use_skim = np.isfinite(skim_d)
    raw = np.where(use_skim, np.maximum(skim_d, float(min_zone_km)), hav)
    dist = raw + offsets[None, :]
    ln_dist = np.log(np.maximum(dist, 1e-9) + float(log_offset))

    out = pd.DataFrame({
        case_id_col: np.repeat(case_ids, n_a),
        alt_id_col: np.tile(alt_ids, n_c),
        "dist_km": dist.ravel(),
        "skim_dist_km": skim_d.ravel(),
        "skim_time_min": skim_t.ravel(),
        "dist_source": np.where(use_skim.ravel(), "skim", "haversine"),
        "ln_dist": ln_dist.ravel(),
    })
    return out


def summarize_distance_sources(long_df: pd.DataFrame, source_col: str = "dist_source") -> dict:
    """Share of ``(case, alt)`` pairs served by skim vs haversine."""
    if long_df is None or len(long_df) == 0:
        return {"n_pairs": 0, "skim_share": 0.0}
    n = len(long_df)
    s = float((long_df[source_col] == "skim").mean()) if source_col in long_df.columns else 0.0
    return {"n_pairs": int(n), "skim_share": s}


__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "pick_skim_matrices",
    "load_skim_matrices",
    "build_case_alt_distances",
    "summarize_distance_sources",
]
