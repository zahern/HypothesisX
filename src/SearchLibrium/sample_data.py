"""Bundled example discrete-choice datasets.

Mirrors the loader pattern used by the sibling ``metacountregressor`` package
(``metacountregressor/sample_data.py``): try local on-disk candidates first
(works for editable/source installs), then fall back to
``importlib.resources`` (works for installed wheels). Every loader returns
the dataset completely unmodified from source — see each docstring for the
exact ``Parameters(...)`` kwargs the dataset expects, since the three sets
use different column-naming conventions.

These CSVs were fetched from the package author's own reference-data repo
(https://github.com/zahern/HypothesisX/tree/main/data) — the same three URLs
``main.py``'s ``prepare_dataset()``/``preview_dataset()`` fetch at call time —
now vendored locally so they work offline (e.g. on HPC compute nodes with no
internet egress) and so a remote outage can't break example code.
"""
from __future__ import annotations

from importlib import resources
from io import BytesIO
from pathlib import Path

import pandas as pd

_DATA_FILES = {
    "electricity": "electricity.csv",
    "travel_mode": "TravelMode.csv",
    "swiss_metro": "Swissmetro_final.csv",
}


def _load_bytes(filename: str) -> bytes:
    local_path = Path(__file__).resolve().parent / "data" / filename
    if local_path.exists():
        return local_path.read_bytes()
    try:
        resource = resources.files("SearchLibrium").joinpath("data", filename)
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        raise FileNotFoundError(
            f"Could not locate packaged dataset {filename!r}. "
            f"Expected SearchLibrium/data/{filename} to be installed."
        )


def load_electricity_data() -> pd.DataFrame:
    """Stated-preference electricity supplier choice (Train & Hensher).

    Long format, 4 alternatives/observation. Columns: ``choice`` (bool),
    ``id`` (individual), ``alt`` (1-4), ``chid`` (choice occasion —
    use this as ``choice_id``, NOT ``id``, since each individual answers
    multiple choice scenarios), ``pf``/``cl``/``loc``/``wk``/``tod``/``seas``
    (attributes).

    Example
    -------
    >>> from SearchLibrium import Parameters, call_search, load_electricity_data
    >>> df = load_electricity_data()
    >>> varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    >>> params = Parameters(
    ...     criterions=[("bic", -1)], df=df, varnames=varnames,
    ...     asvarnames=varnames, isvarnames=[],
    ...     choice_set=sorted(df["alt"].unique().tolist()),
    ...     choices=df["choice"].astype(int).values,
    ...     alt_var=df["alt"].values, choice_id=df["chid"].values,
    ...     ind_id=df["id"].values, base_alt=sorted(df["alt"].unique())[0],
    ...     models=["multinomial"],
    ... )
    >>> best = call_search(params, algorithm="sa")
    """
    return pd.read_csv(BytesIO(_load_bytes(_DATA_FILES["electricity"])))


def load_travel_mode_data() -> pd.DataFrame:
    """Mode choice: air / train / bus / car (Greene & Hensher).

    Long format, 4 alternatives/observation. Columns: ``individual``
    (doubles as the choice-set id — each individual has exactly one choice
    occasion), ``mode`` (the alternative: air/train/bus/car — use as
    ``alt_var``), ``choice`` ("yes"/"no" string), ``wait``/``vcost``/
    ``travel``/``gcost``/``income``/``size`` (attributes).

    Example
    -------
    >>> from SearchLibrium import Parameters, call_search, load_travel_mode_data
    >>> df = load_travel_mode_data()
    >>> varnames = ["wait", "vcost", "travel", "gcost"]
    >>> params = Parameters(
    ...     criterions=[("bic", -1)], df=df, varnames=varnames,
    ...     asvarnames=varnames, isvarnames=[],
    ...     choice_set=sorted(df["mode"].unique().tolist()),
    ...     choices=(df["choice"] == "yes").astype(int).values,
    ...     alt_var=df["mode"].values, choice_id=df["individual"].values,
    ...     base_alt="car", models=["multinomial"],
    ... )
    >>> best = call_search(params, algorithm="sa")
    """
    return pd.read_csv(BytesIO(_load_bytes(_DATA_FILES["travel_mode"])), index_col=0)


def load_swiss_metro_data() -> pd.DataFrame:
    """Swiss Metro stated-preference study (car / train / SM).

    Long format, 3 alternatives/observation, panel structure. Columns:
    ``custom_id`` (choice-set id), ``alt`` (CAR/TRAIN/SM), ``CHOICE`` (bool),
    ``ID`` (individual — for panel/random-parameter models), ``TIME``/
    ``COST``/``HEADWAY``/``SEATS`` (attributes), plus SP-survey demographics
    (``PURPOSE``, ``AGE``, ``INCOME``, ...).

    This is the dataset used in the package README's Quick Start.

    Example
    -------
    >>> from SearchLibrium import Parameters, call_search, load_swiss_metro_data
    >>> df = load_swiss_metro_data()
    >>> varnames = ["TIME", "COST", "HEADWAY", "SEATS"]
    >>> params = Parameters(
    ...     criterions=[("bic", -1)], df=df, varnames=varnames,
    ...     asvarnames=varnames, isvarnames=[],
    ...     choice_set=sorted(df["alt"].unique().tolist()),
    ...     choices=df["CHOICE"].astype(int).values,
    ...     alt_var=df["alt"].values, choice_id=df["custom_id"].values,
    ...     ind_id=df["ID"].values, base_alt="SM",
    ...     models=["multinomial", "mixed_logit"], allow_random=True,
    ... )
    >>> best = call_search(params, algorithm="sa")
    """
    return pd.read_csv(BytesIO(_load_bytes(_DATA_FILES["swiss_metro"])))


_LOADERS = {
    "electricity": load_electricity_data,
    "travel_mode": load_travel_mode_data,
    "swiss_metro": load_swiss_metro_data,
}


def preview_datasets() -> None:
    """Print the head of each bundled example dataset.

    Reliable local equivalent of ``SearchLibrium.main.preview_dataset()``,
    which fetches the same three datasets over the network instead.
    """
    for name, loader in _LOADERS.items():
        df = loader()
        print(f"\n{'=' * 60}\n{name}  (shape={df.shape})\n{'=' * 60}")
        print(df.head())
