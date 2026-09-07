"""Zone skim (OD matrix) handling for SearchLibrium.

Promoted from the SEQ activity-based pipeline (``read_skims.py`` in
``Z:/test_runs_tours/code/``) so destination-choice, accessibility and
logsum workflows share one maintained OMX reader instead of each stage
script carrying its own copy.

Capabilities
------------
* :func:`omx_to_dataframe` — OMX file → flat ``OTAZ`` / ``DTAZ`` long
  DataFrame, chunked for memory. Auto-detects lightweight h5py-written OMX
  files (``OMX_VERSION`` attr + ``/lookup``) and falls back to the
  ``openmatrix`` package otherwise. Both ``h5py`` and ``openmatrix`` are
  optional dependencies (``pip install 'SearchLibrium[skims]'``).
* :func:`read_omx_skims` — reader plus derived level-of-service columns
  (``tt_avg``, ``log_dist``/``dist``, ``distwalk``/``distbike``, ``crash``,
  ``cost_proxy``, ``tt_am``/``tt_pm``) and an optional zone-count guard.
* :func:`prepare_skims_from_omx` — accept a path, an open ``omx.File`` or an
  existing DataFrame and always return a ``{name: DataFrame}``-style dict.
* :func:`merge_skim_component` — generic OD-keyed merge helper (crash/ECF,
  safety, ...).

Unlike the pipeline original, nothing here hard-codes a zone system: pass
``expected_n_zones`` to keep the loud mismatch guard, or leave it ``None``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from typing import Dict, List, Optional
except Exception:  # pragma: no cover
    pass


def _build_default_matrices():
    """LOS matrix universe: auto/transit submodes × periods + legacy names.

    Missing names are silently skipped by :func:`omx_to_dataframe`, so this
    list is safe against files that only carry a subset.
    """
    periods = ['AM', 'MD', 'PM', 'EA', 'EV']
    auto = [f'{mode}_{metric}__{p}'
            for mode in ('SOVTOLL', 'HOV2TOLL', 'HOV3TOLL')
            for metric in ('TIME', 'DIST', 'TOLL')
            for p in periods]
    transit = [f'{mode}_{metric}__{p}'
               for mode in ('DRV_LOC_WLK', 'WLK_LOC_DRV', 'WLK_LOC_WLK')
               for metric in ('TOTIVT', 'FAR', 'WAIT', 'BOARDS')
               for p in periods]
    legacy_alternates = [
        'CRASH', 'SOV_TIME__AM', 'SOV_TIME__PM', 'SOVTIME_AM', 'SOVTIME_PM',
        'AUTO_TIME', 'WALK_DIST', 'BIKE_DIST',
    ]
    return ['DIST', 'DISTWALK', 'DISTBIKE'] + auto + transit + legacy_alternates


DEFAULT_MATRICES = _build_default_matrices()


def list_omx_matrices(omx_path):
    """Print and return the matrix names stored in an OMX file."""
    try:
        import openmatrix as omx
    except ImportError as e:
        raise ImportError("pip install openmatrix") from e
    f = omx.open_file(omx_path, 'r')
    names = f.list_matrices()
    mappings = f.list_mappings()
    try:
        shape_str = str(f.shape())
    except Exception:
        shape_str = 'unknown'
    matrix_info = {}
    for n in names:
        try:
            matrix_info[n] = getattr(f[n], 'shape', '?')
        except Exception:
            matrix_info[n] = '?'
    f.close()
    print(f"OMX file : {os.path.basename(omx_path)}")
    print(f"  Shape  : {shape_str}")
    print(f"  Zone mappings : {mappings}")
    print(f"  Matrices ({len(names)}):")
    for n in names:
        print(f"    {n}  {matrix_info.get(n, '')}")
    return names


def _is_h5py_omx(omx_path) -> bool:
    """Detect lightweight h5py-written OMX files (``OMX_VERSION`` attr)."""
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(omx_path, 'r') as f:
            return 'OMX_VERSION' in f.attrs
    except Exception:
        return False


def _omx_to_dataframe_h5py(omx_path, matrices=None, zone_index=None, chunksize=500):
    """h5py-based reader for lightweight OMX files (see :func:`_is_h5py_omx`)."""
    import h5py

    with h5py.File(omx_path, 'r') as f:
        available = list(f['data'].keys())
        logger.info("OMX (h5py): %d matrices available", len(available))
        if matrices is None:
            matrices = [m for m in DEFAULT_MATRICES if m in available]
            if not matrices:
                matrices = list(available)
                logger.warning("None of the default matrix names found. "
                               "Reading all %d matrices.", len(matrices))
        else:
            missing = [m for m in matrices if m not in available]
            if missing:
                logger.warning("Requested matrices not found in OMX: %s", missing)
            matrices = [m for m in matrices if m in available]
        if not matrices:
            raise ValueError(f"No readable matrices found in {omx_path}. "
                             f"Available: {available}")

        lookup_names = list(f['lookup'].keys()) if 'lookup' in f else []
        lookup_name = zone_index if zone_index in lookup_names else (
            lookup_names[0] if lookup_names else None)

        if lookup_name:
            raw = f['lookup'][lookup_name][...]
            zone_ids = np.array([int(z.decode() if isinstance(z, bytes) else z)
                                 for z in raw], dtype=int)
            row_order = np.arange(len(zone_ids), dtype=int)
        else:
            n = f['data'][matrices[0]].shape[0]
            zone_ids = np.arange(1, n + 1, dtype=int)
            row_order = np.arange(n, dtype=int)
            logger.warning("No usable zone mapping — using 1-based indices 1..%d.", n)

        n_zones = len(zone_ids)
        arrays = {}
        for name in matrices:
            arr = np.array(f['data'][name], dtype=np.float32)
            if arr.shape != (n_zones, n_zones):
                logger.warning("Matrix '%s' shape %s != (%d,%d) — skipping",
                               name, arr.shape, n_zones, n_zones)
                continue
            arrays[name] = arr

    if not arrays:
        raise ValueError("No matrices could be read from the OMX file.")
    return _matrices_to_long(arrays, zone_ids, row_order, chunksize)


def _matrices_to_long(arrays, zone_ids, row_order, chunksize=500):
    """Stack ``(n, n)`` arrays into a flat OTAZ/DTAZ DataFrame (chunked)."""
    n_zones = len(zone_ids)
    chunks = []
    read_matrices = list(arrays.keys())
    for chunk_start in range(0, n_zones, chunksize):
        chunk_end = min(chunk_start + chunksize, n_zones)
        chunk_zids = zone_ids[chunk_start:chunk_end]
        chunk_rows = row_order[chunk_start:chunk_end]
        n_chunk = len(chunk_zids)
        chunk_dict = {'OTAZ': np.repeat(chunk_zids, n_zones),
                      'DTAZ': np.tile(zone_ids, n_chunk)}
        for name in read_matrices:
            chunk_dict[name] = arrays[name][np.ix_(chunk_rows, row_order)].ravel()
        chunks.append(pd.DataFrame(chunk_dict))
    df = pd.concat(chunks, ignore_index=True)
    df['OTAZ'] = df['OTAZ'].astype(int)
    df['DTAZ'] = df['DTAZ'].astype(int)
    return df


def omx_to_dataframe(omx_path, matrices=None, zone_index=None, chunksize=500):
    """Convert an OMX file to a flat OTAZ/DTAZ DataFrame.

    Parameters
    ----------
    omx_path : path to ``.omx`` file
    matrices : list of matrix names to read; None = default universe (or all
        matrices when none of the defaults are present)
    zone_index : OMX zone mapping name; None = auto-detect
    chunksize : origin zones per chunk (bounds peak memory)
    """
    if _is_h5py_omx(omx_path):
        return _omx_to_dataframe_h5py(omx_path, matrices=matrices,
                                      zone_index=zone_index, chunksize=chunksize)
    try:
        import openmatrix as omx
    except ImportError as e:
        raise ImportError("pip install openmatrix") from e

    f = omx.open_file(omx_path, 'r')
    available = f.list_matrices()
    logger.info("OMX: %d matrices available", len(available))
    if matrices is None:
        matrices = [m for m in DEFAULT_MATRICES if m in available]
        if not matrices:
            matrices = list(available)
            logger.warning("None of the default matrix names found. "
                           "Reading all %d matrices.", len(matrices))
    else:
        missing = [m for m in matrices if m not in available]
        if missing:
            logger.warning("Requested matrices not found in OMX: %s", missing)
        matrices = [m for m in matrices if m in available]
    if not matrices:
        f.close()
        raise ValueError(f"No readable matrices found in {omx_path}. "
                         f"Available: {available}")

    mappings = f.list_mappings()
    if zone_index is None:
        zone_index = mappings[0] if mappings else None
    zone_ids, row_order = None, None
    if zone_index and zone_index in mappings:
        try:
            raw = f.mapping(zone_index)
            if isinstance(raw, dict):
                pairs = sorted(raw.items(), key=lambda kv: kv[1])
                zone_ids = np.array([int(k) for k, _v in pairs], dtype=int)
                row_order = np.array([int(v) for _k, v in pairs], dtype=int)
        except Exception as e:  # noqa: BLE001
            logger.warning("Zone mapping read failed (%s); using 1-based indices.", e)
    arrays = {}
    for name in matrices:
        try:
            arr = np.array(f[name], dtype=np.float32)
            arrays[name] = arr
        except Exception as e:  # noqa: BLE001
            logger.warning("Matrix '%s' unreadable (%s) — skipping.", name, e)
    if zone_ids is None:
        n = next(iter(arrays.values())).shape[0]
        zone_ids = np.arange(1, n + 1, dtype=int)
        row_order = np.arange(n, dtype=int)
    f.close()
    if not arrays:
        raise ValueError("No matrices could be read from the OMX file.")
    return _matrices_to_long(arrays, zone_ids, row_order, chunksize)


def add_derived_los(df, fuel_cost_per_km=0.12):
    """Add derived level-of-service columns (idempotent).

    Adds ``tt_avg``, ``log_dist``/``dist``, ``distwalk``/``distbike``,
    ``crash``, ``cost_proxy``, ``tt_am``/``tt_pm`` from whatever base
    columns are present. Missing inputs yield neutral defaults.
    """
    df = df.copy()
    am_col = next((c for c in ['SOVTOLL_TIME__AM', 'SOV_TIME__AM',
                               'SOVTIME_AM', 'AUTO_TIME'] if c in df.columns), None)
    pm_col = next((c for c in ['SOVTOLL_TIME__PM', 'SOV_TIME__PM',
                               'SOVTIME_PM'] if c in df.columns), None)
    if am_col and pm_col:
        df['tt_avg'] = (df[am_col] + df[pm_col]) / 2.0
    elif am_col:
        df['tt_avg'] = df[am_col]
    else:
        df['tt_avg'] = 0.0
    dist_col = next((c for c in ['DIST', 'distance_km', 'dist'] if c in df.columns), None)
    if dist_col:
        df['log_dist'] = np.log1p(df[dist_col].astype(float))
        df['dist'] = df[dist_col]
    else:
        df['log_dist'] = 0.0
        df['dist'] = 0.0
    for src, dst in [('DISTWALK', 'distwalk'), ('WALK_DIST', 'distwalk'),
                     ('DISTBIKE', 'distbike'), ('BIKE_DIST', 'distbike')]:
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if 'distwalk' not in df.columns:
        df['distwalk'] = df.get('dist', 0.0)
    if 'distbike' not in df.columns:
        df['distbike'] = df.get('dist', 0.0)
    df['crash'] = df['CRASH'] if 'CRASH' in df.columns else 0.0
    df['cost_proxy'] = df['dist'] * float(fuel_cost_per_km)
    if am_col and 'tt_am' not in df.columns:
        df['tt_am'] = df[am_col]
    if pm_col and 'tt_pm' not in df.columns:
        df['tt_pm'] = df[pm_col]
    return df


def overlay_walk_bike(df, overlay_path=None, walk_col='DISTWALK', bike_col='DISTBIKE'):
    """Replace placeholder walk/bike distances with OSM-routed ones.

    No-op when ``overlay_path`` is None or missing (self-activates once the
    OSM walk/bike builder has run). The overlay file must be an OMX or a
    DataFrame with OTAZ/DTAZ plus walk/bike distance columns.
    """
    if overlay_path is None:
        return df
    try:
        import os as _os
        if isinstance(overlay_path, str) and not _os.path.exists(overlay_path):
            return df
        over = (overlay_path if isinstance(overlay_path, pd.DataFrame)
                else omx_to_dataframe(overlay_path, matrices=[walk_col, bike_col]))
        keep = [c for c in ['OTAZ', 'DTAZ', walk_col, bike_col] if c in over.columns]
        over = over[keep].drop_duplicates(['OTAZ', 'DTAZ'])
        df = df.merge(over, on=['OTAZ', 'DTAZ'], how='left', suffixes=('', '_osm'))
        for base in (walk_col, bike_col):
            osm = base + '_osm'
            if osm in df.columns:
                df[base] = df[osm].fillna(df[base] if base in df.columns else 0.0)
                df = df.drop(columns=[osm])
        logger.info("overlay_walk_bike: applied OSM walk/bike from %s", overlay_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("overlay_walk_bike skipped (%s)", e)
    return df


def read_omx_skims(omx_path, matrices=None, zone_index=None, add_derived=True,
                   expected_n_zones=None, overlay_path=None):
    """Read an OMX skim file into a modelling-ready DataFrame.

    Parameters
    ----------
    expected_n_zones : int or None. When set, raise loudly if the skim's
        zone count differs (the single choke point that stops a stray
        wrong-sized OMX from silently misaligning OTAZ/DTAZ downstream).
    overlay_path : optional OSM walk/bike overlay (see :func:`overlay_walk_bike`).
    """
    df = omx_to_dataframe(omx_path, matrices=matrices, zone_index=zone_index)
    if expected_n_zones is not None:
        try:
            seen = max(int(df['DTAZ'].nunique()), int(df['OTAZ'].nunique()))
        except Exception:
            seen = 0
        if seen != int(expected_n_zones):
            raise ValueError(
                f"read_omx_skims: '{os.path.basename(str(omx_path))}' has {seen} zones, "
                f"expected {expected_n_zones}.")
    df = overlay_walk_bike(df, overlay_path)
    if add_derived:
        df = add_derived_los(df)
    return df


def prepare_skims_from_omx(skims_input, matrices=None, zone_index=None):
    """Accept a path, an open ``omx.File`` or a DataFrame; return a DataFrame."""
    if isinstance(skims_input, pd.DataFrame):
        return skims_input
    if isinstance(skims_input, (str, os.PathLike)):
        return omx_to_dataframe(str(skims_input), matrices=matrices, zone_index=zone_index)
    # Assume an open openmatrix file handle.
    try:
        names = skims_input.list_matrices()
        if matrices is None:
            matrices = [m for m in DEFAULT_MATRICES if m in names] or list(names)
        arrays, n = {}, None
        for name in matrices:
            try:
                arr = np.array(skims_input[name], dtype=np.float32)
                arrays[name] = arr
                n = arr.shape[0]
            except Exception as e:  # noqa: BLE001
                logger.warning("Matrix '%s' unreadable (%s) — skipping.", name, e)
        zone_ids = np.arange(1, n + 1, dtype=int)
        return _matrices_to_long(arrays, zone_ids, np.arange(n, dtype=int))
    except Exception as e:
        raise ValueError(f"prepare_skims_from_omx: unsupported input {type(skims_input)}: {e}")


def merge_skim_component(skims, extra, value_cols, on=('OTAZ', 'DTAZ'), fill=0.0):
    """Left-merge an OD-keyed component (crash rates, safety, ECF, ...) onto skims."""
    extra = extra.copy()
    keep = [c for c in list(on) + list(value_cols) if c in extra.columns]
    extra = extra[keep].drop_duplicates(list(on))
    out = skims.merge(extra, on=list(on), how='left')
    for c in value_cols:
        if c in out.columns:
            out[c] = out[c].fillna(fill)
    return out


__all__ = [
    "DEFAULT_MATRICES",
    "list_omx_matrices",
    "omx_to_dataframe",
    "add_derived_los",
    "overlay_walk_bike",
    "read_omx_skims",
    "prepare_skims_from_omx",
    "merge_skim_component",
]
