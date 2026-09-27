import numpy as np
import pandas as pd

from SearchLibrium.mh_choice_estimator import (
    ChoiceSetFrame,
    estimate_dest_choice,
)


def test_public_destination_estimator_accepts_sparse_choice_frame():
    rows = []
    for case_id, chosen in enumerate([1, 2, 1, 2]):
        for alt_id in [1, 2, 3]:
            rows.append({
                "case_id": case_id,
                "alt_id": alt_id,
                "chosen": alt_id == chosen,
                "distance": float(abs(alt_id - chosen) + 1),
            })
    frame = ChoiceSetFrame.from_long(pd.DataFrame(rows), ["distance"])

    result = estimate_dest_choice(
        frame, method="gibbs", draws=4, burn_in=2, set_size=2,
        inner_steps=1, seed=11)

    assert result.method == "gibbs"
    assert result.coef.shape == (1,)
    assert np.isfinite(result.loglike)


def test_public_destination_estimator_rejects_mismatched_sparse_features():
    data = pd.DataFrame({
        "case_id": [0, 0, 1, 1],
        "alt_id": [1, 2, 1, 2],
        "chosen": [True, False, False, True],
        "distance": [1.0, 2.0, 2.0, 1.0],
    })
    frame = ChoiceSetFrame.from_long(data, ["distance"])

    try:
        estimate_dest_choice(frame, feature_cols=["other"])
    except ValueError as error:
        assert "feature_cols" in str(error)
    else:
        raise AssertionError("mismatched sparse feature names should fail")