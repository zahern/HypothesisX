"""Accessibility measures for SearchLibrium.

Promoted from the SEQ activity-based pipeline (``compute_accessibility.py``
+ ``logsum_accessibility.py`` in ``Z:/test_runs_tours/code/``), generalized
so any zone system, skim column set and land-use table works — the pipeline's
SEQ-specific column names live in :data:`SEQ_ACCESSIBILITY_SPEC` as a ready
starting point, not as hard requirements.

Two families
------------
* **Gravity accessibility** — ``acc[o] = Σ_d Attr[d] · f(tt[o,d])`` with
  exponential (``exp(-c·tt)``) or power (``tt^-c``) decay and an optional
  travel-time cap. Attractors are weighted sums of land-use columns.
* **Logsum accessibility** — ``acc[o] = log Σ_z exp(LS[o,z])`` over a matrix
  of mode-choice logsums, i.e. the expected maximum utility of reaching any
  destination (the stage-5 ``compute_logsum_accessibility`` pattern).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ready-made spec in the pipeline's conventions (SEQ land-use + skim names).
# Copy, trim and re-point at your own columns; see compute_accessibility().
SEQ_ACCESSIBILITY_SPEC = {
    'auto_work_acc': {
        'attractor_cols': {'TOTEMP': 1.0, 'FPSEMPN': 0.15, 'HEREMPN': 0.10},
        'skim_col': 'SOVTOLL_TIME__AM', 'decay_type': 'exp',
        'decay_coeff': 0.05, 'cap_tt': 120.0,
    },
    'auto_edu_acc': {
        'attractor_cols': {'Primary': 1.0, 'Secondary': 1.0, 'Education': 50.0},
        'skim_col': 'SOVTOLL_TIME__AM', 'decay_type': 'exp',
        'decay_coeff': 0.06, 'cap_tt': 90.0,
    },
    'auto_shop_acc': {
        'attractor_cols': {'Shopping': 10.0, 'retail': 0.001, 'RETEMPN': 0.5},
        'skim_col': 'SOVTOLL_TIME__AM', 'decay_type': 'exp',
        'decay_coeff': 0.08, 'cap_tt': 60.0,
    },
}


def _decay(tt, decay_type='exp', decay_coeff=0.05, cap_tt=None):
    tt = np.asarray(tt, dtype=float)
    if cap_tt is not None:
        tt = np.where(tt > float(cap_tt), np.inf, tt)
    if (decay_type or 'exp') == 'power':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(tt > 0, np.power(np.maximum(tt, 1e-6), -float(decay_coeff)), 0.0)
    return np.exp(-float(decay_coeff) * tt)


def gravity_accessibility(skims, landuse, attractor_cols, skim_col,
                          origin_col='OTAZ', dest_col='DTAZ',
                          zone_col=None, decay_type='exp', decay_coeff=0.05,
                          cap_tt=None):
    """Gravity accessibility of each origin zone.

    Parameters
    ----------
    skims : DataFrame with ``origin_col`` / ``dest_col`` / ``skim_col``
    landuse : DataFrame with one row per zone and attractor columns
    attractor_cols : dict ``{column: weight}``; missing columns are skipped
        with a warning
    zone_col : land-use zone-id column; None = use the land-use index
    """
    skim_col = str(skim_col)
    if skim_col not in skims.columns:
        raise ValueError(f"gravity_accessibility: skim column '{skim_col}' not in skims")
    zone_key = landuse.index if zone_col is None else landuse[zone_col]
    attract = np.zeros(len(landuse), dtype=float)
    for col, w in dict(attractor_cols or {}).items():
        if col in landuse.columns:
            attract += float(w) * np.asarray(landuse[col], dtype=float)
        else:
            logger.warning("gravity_accessibility: attractor column '%s' missing — skipped", col)
    lut = pd.Series(attract, index=np.asarray(zone_key)).to_dict()
    dest_attr = skims[dest_col].map(lut).fillna(0.0).to_numpy(dtype=float)
    imp = _decay(skims[skim_col].to_numpy(dtype=float), decay_type, decay_coeff, cap_tt)
    skims = skims.copy()
    skims['_acc_w'] = dest_attr * imp
    acc = skims.groupby(origin_col)['_acc_w'].sum()
    return acc


def compute_accessibility(skims, landuse, spec, origin_col='OTAZ',
                          dest_col='DTAZ', zone_col=None):
    """Compute every measure in a ``{name: measure-spec}`` dict.

    Each measure spec accepts ``attractor_cols`` (``{col: weight}``),
    ``skim_col``, ``decay_type`` (``'exp'``/``'power'``), ``decay_coeff``
    and ``cap_tt``. Returns a DataFrame indexed by origin zone.
    """
    out = {}
    for name, cfg in dict(spec or {}).items():
        cfg = dict(cfg)
        try:
            out[name] = gravity_accessibility(
                skims, landuse,
                attractor_cols=cfg.get('attractor_cols', {}),
                skim_col=cfg.get('skim_col'),
                origin_col=origin_col, dest_col=dest_col, zone_col=zone_col,
                decay_type=cfg.get('decay_type', 'exp'),
                decay_coeff=cfg.get('decay_coeff', 0.05),
                cap_tt=cfg.get('cap_tt'))
        except Exception as e:  # noqa: BLE001
            logger.warning("compute_accessibility: measure '%s' failed (%s)", name, e)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def logsum_accessibility(mode_logsums):
    """Logsum accessibility from mode-choice logsums.

    Parameters
    ----------
    mode_logsums : DataFrame (origins × destinations) of combined mode
        logsums ``LS[o,z]``, or a dict ``{mode: DataFrame}`` which is first
        combined as ``log Σ_modes exp(LS_mode)``.

    Returns
    -------
    Series indexed by origin: ``acc[o] = log Σ_z exp(LS[o,z])``.
    """
    if isinstance(mode_logsums, dict):
        ref = next(iter(mode_logsums.values()))
        stacked = None
        for df in mode_logsums.values():
            e = np.exp(np.asarray(df, dtype=float))
            stacked = e if stacked is None else stacked + e
        vals = np.log(np.maximum(stacked, 1e-300))
        with np.errstate(over='ignore'):
            acc = np.log(np.exp(vals - vals.max(axis=1, keepdims=True)).sum(axis=1)) + vals.max(axis=1)
        return pd.Series(acc, index=ref.index)
    df = mode_logsums
    vals = np.asarray(df, dtype=float)
    with np.errstate(over='ignore'):
        acc = np.log(np.exp(vals - vals.max(axis=1, keepdims=True)).sum(axis=1)) + vals.max(axis=1)
    return pd.Series(acc, index=df.index)


__all__ = [
    "SEQ_ACCESSIBILITY_SPEC",
    "gravity_accessibility",
    "compute_accessibility",
    "logsum_accessibility",
]
