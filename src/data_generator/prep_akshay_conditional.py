"""Expand the Akshay MaaS panel into the *conditional* preference->purchase layout.

The raw file ``data/akshay_long_true.csv`` stores, per person (``indID``) and per
scenario (``CHID``), two rows -- one for each hypothetical MaaS scheme (``alt2``
in {1, 2}).  Two things are observed for every scenario:

  * ``CHOICE`` / ``pref``  -- which of the two schemes the respondent preferred.
  * ``purchase``           -- whether they would actually buy the preferred
                              scheme, or opt out.

Vij et al. (2020) model this with a *conditional* structure: a preference logit
over {A, B} sharing tastes ``beta_s`` with a purchase logit that pits the
*preferred* scheme (utility ``x'beta_s``) against an opt-out constant
``alpha_s``.  Because the class log-likelihood is additive across the two parts,
we can reproduce it exactly in a standard latent-class choice model by turning
every scenario into **two J=2 choice tasks**:

  1. Preference task : rows {scheme A, scheme B}, chosen = preferred scheme.
  2. Purchase task   : rows {preferred scheme, opt-out}, chosen = buy / opt-out.
     The opt-out row is all-zero on the attributes except ``asc_optout`` = 1,
     which the model estimates per class -> the class-specific alpha_s.

The class-membership equation stays at the person level (``indID``), so the
companion panel patch to ``LatentClassMixedLogit`` is what ties the eight tasks
of a person to a single class draw.

Run::

    python -m data_generator.prep_akshay_conditional \
        --in data/akshay_long_true.csv --out data/akshay_conditional.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# 19 scheme attributes that enter the class-specific utility x'beta_s.
CHOICE_ATTRS = [
    "Cost", "TktInt", "BkInt", "RTInf", "Pers",
    "LocalPTPayG", "LDPTPayG", "TaxiPayG", "CarRentalPayG",
    "CarsharePayG", "RidesharePayG", "BikesharePayG",
    "LocalPTUnl", "LDPTUnl", "TaxiUnl", "CarRentalUnl",
    "CarshareUnl", "RideshareUnl", "BikeshareUnl",
]

# Person-level covariates for the class-membership MNL (z_n).
MEMBERSHIP_VARS = [
    "InnerCity", "InnerRegional", "Under30", "Over65", "College",
    "FullTime", "PartTime", "Male", "Children", "NDI", "Income",
]

PANEL_COL = "indID"
SCENARIO_COL = "CHID"
ALT_COL = "alt2"
PREF_CHOICE_COL = "CHOICE"      # 1 on the row of the preferred scheme
PURCHASE_COL = "purchase"       # constant within scenario


def build_conditional_long(
    df: pd.DataFrame,
    purchase_yes_value: int = 1,
) -> pd.DataFrame:
    """Return the expanded preference+purchase long frame.

    Parameters
    ----------
    df : DataFrame
        Raw long data (two rows per scenario).
    purchase_yes_value : int
        Value of ``purchase`` that means "would buy the preferred scheme".
        Defaults to 1 (so 2 == opt out).
    """
    keep = [PANEL_COL, SCENARIO_COL, ALT_COL, PREF_CHOICE_COL, PURCHASE_COL]
    keep += CHOICE_ATTRS + MEMBERSHIP_VARS
    df = df[keep].copy()

    out_rows = []
    task_counter = 0

    # Deterministic order: person, scenario.
    for (ind, chid), g in df.groupby([PANEL_COL, SCENARIO_COL], sort=True):
        g = g.sort_values(ALT_COL)
        pref_row = g[g[PREF_CHOICE_COL] == 1]
        if len(pref_row) != 1:
            raise ValueError(
                f"scenario indID={ind} CHID={chid} does not have exactly one "
                f"preferred scheme (CHOICE==1)."
            )
        pref_row = pref_row.iloc[0]
        bought = int(g[PURCHASE_COL].iloc[0]) == purchase_yes_value

        # ---- Task 1: preference (A vs B) -------------------------------------
        task_counter += 1
        pref_task_id = task_counter
        for _, r in g.iterrows():
            rec = {
                "task_id": pref_task_id,
                "indID": ind,
                "CHID": chid,
                "task_type": "pref",
                "alt": int(r[ALT_COL]),
                "choice": int(r[PREF_CHOICE_COL]),
                "asc_optout": 0,
            }
            for c in CHOICE_ATTRS:
                rec[c] = r[c]
            for c in MEMBERSHIP_VARS:
                rec[c] = r[c]
            out_rows.append(rec)

        # ---- Task 2: purchase (preferred scheme vs opt-out) ------------------
        task_counter += 1
        purch_task_id = task_counter

        # alt 1 == preferred scheme (its real attributes), chosen if bought.
        rec = {
            "task_id": purch_task_id,
            "indID": ind,
            "CHID": chid,
            "task_type": "purch",
            "alt": 1,
            "choice": 1 if bought else 0,
            "asc_optout": 0,
        }
        for c in CHOICE_ATTRS:
            rec[c] = pref_row[c]
        for c in MEMBERSHIP_VARS:
            rec[c] = pref_row[c]
        out_rows.append(rec)

        # alt 2 == opt-out: zero attributes, asc_optout = 1, chosen if not bought.
        rec = {
            "task_id": purch_task_id,
            "indID": ind,
            "CHID": chid,
            "task_type": "purch",
            "alt": 2,
            "choice": 0 if bought else 1,
            "asc_optout": 1,
        }
        for c in CHOICE_ATTRS:
            rec[c] = 0.0
        for c in MEMBERSHIP_VARS:
            rec[c] = pref_row[c]  # person covariates carried on every row
        out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    cols = (
        ["task_id", "indID", "CHID", "task_type", "alt", "choice", "asc_optout"]
        + CHOICE_ATTRS + MEMBERSHIP_VARS
    )
    return out[cols]


def _validate(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    n_scen = raw[SCENARIO_COL].nunique()
    n_person = raw[PANEL_COL].nunique()

    assert len(out) == n_scen * 4, (len(out), n_scen * 4)
    assert out["task_id"].nunique() == n_scen * 2
    assert (out.groupby("task_id").size() == 2).all(), "every task must be J=2"
    assert out.groupby("task_id")["choice"].sum().eq(1).all(), \
        "every task must have exactly one chosen alt"
    # membership constant within person
    for c in MEMBERSHIP_VARS:
        assert (out.groupby("indID")[c].nunique() == 1).all(), c
    print(f"[prep] OK  persons={n_person}  scenarios={n_scen}  "
          f"tasks={out['task_id'].nunique()}  rows={len(out)}")
    # purchase rate sanity
    purch = out[(out.task_type == 'purch') & (out.asc_optout == 0)]
    print(f"[prep] purchase (buy) rate = {purch['choice'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/akshay_long_true.csv")
    ap.add_argument("--out", dest="out", default="data/akshay_conditional.csv")
    ap.add_argument("--purchase-yes", type=int, default=1,
                    help="value of 'purchase' meaning 'would buy' (default 1)")
    args = ap.parse_args()

    raw = pd.read_csv(args.inp)
    out = build_conditional_long(raw, purchase_yes_value=args.purchase_yes)
    _validate(raw, out)
    out.to_csv(args.out, index=False)
    print(f"[prep] wrote {args.out}")


if __name__ == "__main__":
    main()
