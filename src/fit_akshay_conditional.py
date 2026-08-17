"""Fit the conditional panel LCCM (Vij et al. 2020 structure) on the Akshay data.

Expects the expanded file produced by
``data_generator/prep_akshay_conditional.py``.

    python fit_akshay_conditional.py            # 2 classes
    python fit_akshay_conditional.py --classes 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SearchLibrium"))
from latent_class import LatentClassMixedLogit  # noqa: E402

CHOICE_ATTRS = [
    "Cost", "TktInt", "BkInt", "RTInf", "Pers",
    "LocalPTPayG", "LDPTPayG", "TaxiPayG", "CarRentalPayG",
    "CarsharePayG", "RidesharePayG", "BikesharePayG",
    "LocalPTUnl", "LDPTUnl", "TaxiUnl", "CarRentalUnl",
    "CarshareUnl", "RideshareUnl", "BikeshareUnl",
]
ASC = ["asc_optout"]
MEMBERSHIP_VARS = [
    "InnerCity", "InnerRegional", "Under30", "Over65", "College",
    "FullTime", "PartTime", "Male", "Children", "NDI", "Income",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/akshay_conditional.csv")
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--l2", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    # utility variables = 19 scheme attributes + opt-out constant;
    # membership variables carried in X but excluded from utility by name.
    util_vars = CHOICE_ATTRS + ASC
    varnames = util_vars + MEMBERSHIP_VARS

    X = df[varnames].to_numpy(dtype=float)
    y = df["choice"].to_numpy(dtype=float)
    ids = df["task_id"].to_numpy()
    alts = df["alt"].to_numpy()
    panels = df["indID"].to_numpy()

    print(f"rows={len(df)}  tasks={df['task_id'].nunique()}  "
          f"persons={df['indID'].nunique()}  K_util={len(util_vars)}  "
          f"K_memb={len(MEMBERSHIP_VARS)}")

    model = LatentClassMixedLogit(
        n_classes=args.classes,
        random_state=1,
        maxiter=200,
        n_init=1,
        l2_penalty=args.l2,
    )
    model.setup(
        X, y, varnames, ids, alts,
        membership_vars=MEMBERSHIP_VARS,
        panels=panels,
    )
    model.fit(em_method="squarem")
    model.get_loglik_null()
    model.summarise_lc(compute_se=True)


if __name__ == "__main__":
    main()
