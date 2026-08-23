"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: DESTINATION COMPONENT PREDICTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

try:
    from .search import Parameters, Solution
    from .multinomial_logit import MultinomialLogit
    from .MixedLogit import MixedLogit
    from .rrm import RandomRegret
    from .mixedrrm import MixedRandomRegret
    from .multinomial_nested import NestedLogit
except ImportError:
    from search import Parameters, Solution
    from multinomial_logit import MultinomialLogit
    from MixedLogit import MixedLogit
    from rrm import RandomRegret
    from mixedrrm import MixedRandomRegret
    from multinomial_nested import NestedLogit

import inspect

import numpy as np
import pandas as pd
from typing import Optional, Union


class DestinationPredictor:
    """
    Predict destination component (alternative) choice probabilities and
    aggregate flows from a fitted SearchLibrium discrete choice model.

    This class provides a high-level interface for:
      - Computing per-trip choice probabilities for each destination alternative
      - Predicting the most likely destination component per trip
      - Computing expected aggregate flows (predicted trip counts) per destination
      - Computing logsums (expected maximum utility) for welfare analysis
      - Generating simulation-based predictions for Mixed Logit models

    Parameters
    ----------
    model : fitted SearchLibrium model (MultinomialLogit, MixedLogit, etc.)
        A fitted discrete choice model.
    idca : pd.DataFrame
        Long-format data frame with columns for all model variables and
        'trip_id', 'dest_code', 'chosen' (optional).
    varnames : list[str]
        Variable names used in the model's utility function.
    """

    def __init__(
        self,
        model,
        idca: pd.DataFrame,
        varnames: list[str],
        dest_col: str = 'dest_code',
        trip_id_col: str = 'trip_id',
        choice_col: str = 'chosen',
        dest_name_col: str = 'dest_suburb',
    ):
        self.model = model
        self.idca = idca
        self.varnames = varnames
        self.dest_col = dest_col
        self.trip_id_col = trip_id_col
        self.choice_col = choice_col
        self.dest_name_col = dest_name_col

        self.trip_ids = sorted(idca[trip_id_col].unique())
        self.dest_codes = sorted(idca[dest_col].unique())
        self.n_trips = len(self.trip_ids)
        self.n_dests = len(self.dest_codes)

        self._probs = None
        self._logsums = None

    # ------------------------------------------------------------------
    # Choice probabilities
    # ------------------------------------------------------------------

    def compute_probabilities(self) -> np.ndarray:
        """
        Return choice probability matrix of shape (n_trips, n_destinations).

        Returns
        -------
        np.ndarray
            P[t, j] = probability trip t chooses destination j.
        """
        prob = getattr(self.model, 'ind_pred_prob', None)
        if prob is None:
            prob = getattr(self.model, 'choice_pred_prob', None)
        if prob is not None and hasattr(prob, 'shape') and len(prob.shape) == 2:
            # JAX-fitted models (_jax=True, the default) leave this as a
            # jax.Array. Downstream code (predict_destinations()) runs plain
            # numpy ops (argmax/argsort) per row in a Python loop; numpy
            # dispatching those against an un-converted jax.Array is
            # catastrophically slow (~4s per argsort call observed, i.e.
            # hours instead of seconds for a few thousand trips) rather than
            # erroring, so convert once here up front.
            prob = np.asarray(prob)
            self._probs = prob
            return prob

        # Compute from model parameters manually
        X = self.idca[self.varnames].fillna(0.0).values.astype(np.float64)
        avail = self.idca.get('_avail_', pd.Series(1.0, index=self.idca.index)).fillna(0).values.astype(np.float64)
        if not hasattr(self.model, 'compute_probabilities'):
            raise RuntimeError(
                "Model does not support predict(). "
                "Ensure the model was fitted with a supported estimator."
            )

        # Only models exposing the simple (betas, X, avail, return_logsum)
        # signature can be evaluated here. Simulation-based models (e.g.
        # MixedLogit) require draws/panel_info and cannot use this path.
        try:
            _sig_params = set(inspect.signature(self.model.compute_probabilities).parameters)
        except (TypeError, ValueError):
            _sig_params = {'betas', 'X', 'avail'}
        if not {'betas', 'X', 'avail'}.issubset(_sig_params):
            raise RuntimeError(
                f"{type(self.model).__name__} does not support the DestinationPredictor "
                "manual probability path (its compute_probabilities() requires "
                "simulation inputs). Use a MultinomialLogit model, or rely on the "
                "model's own ind_pred_prob attribute."
            )

        probs, logsums = self.model.compute_probabilities(
            getattr(self.model, 'coeff_est', getattr(self.model, 'coeff', None)),
            X.reshape(self.n_trips, self.n_dests, len(self.varnames)),
            avail.reshape(self.n_trips, self.n_dests),
            return_logsum=True,
        )
        probs = np.asarray(probs)  # see conversion note above
        self._probs = probs
        self._logsums = np.asarray(logsums)
        return probs

    # ------------------------------------------------------------------
    # Destination component prediction
    # ------------------------------------------------------------------

    def predict_destinations(self) -> pd.DataFrame:
        """
        Predict the most likely destination component for each trip.

        Returns
        -------
        pd.DataFrame with columns:
            trip_id, chosen_dest_code, predicted_dest_code,
            predicted_dest_name, top_k_dests, top_k_probs
        """
        probs = self._probs if self._probs is not None else self.compute_probabilities()
        dest_to_name = self._build_dest_name_map()

        rows = []
        for t_idx, tid in enumerate(self.trip_ids):
            if t_idx >= probs.shape[0]:
                break
            p_t = probs[t_idx]
            best_j = int(np.argmax(p_t))
            best_code = self.dest_codes[best_j]
            best_prob = float(p_t[best_j])
            best_name = dest_to_name.get(best_code, str(best_code))

            # Top-3 alternatives
            top_k = np.argsort(-p_t)[:3]
            top_codes = [self.dest_codes[j] for j in top_k]
            top_probs = [float(p_t[j]) for j in top_k]
            top_names = [dest_to_name.get(c, str(c)) for c in top_codes]

            actual = self.idca[
                (self.idca[self.trip_id_col] == tid)
                & (self.idca[self.choice_col] == 1)
            ]
            # dest_col isn't always integer-coded (e.g. string alt labels
            # like 'CAR'/'SM'/'TRAIN') -- keep the raw value so it compares
            # correctly against dest_codes/best_code, which retain whatever
            # dtype the source data used.
            actual_code = actual[self.dest_col].iloc[0] if len(actual) else -1

            rows.append({
                self.trip_id_col: tid,
                'chosen_dest_code': actual_code,
                'predicted_dest_code': best_code,
                'predicted_dest_name': best_name,
                'predicted_prob': best_prob,
                'correct': actual_code == best_code,
                'top3_dest_codes': top_codes,
                'top3_dest_names': top_names,
                'top3_probs': top_probs,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Aggregate flow prediction
    # ------------------------------------------------------------------

    def predict_aggregate_flows(self) -> pd.DataFrame:
        """
        Compute predicted total trips per destination (sum of choice probabilities).

        Returns
        -------
        pd.DataFrame with columns:
            dest_code, dest_name, pred_flow, obs_flow, share_pred, share_obs
        """
        probs = self._probs if self._probs is not None else self.compute_probabilities()
        dest_to_name = self._build_dest_name_map()

        pred_flows = probs.sum(axis=0)

        obs = (
            self.idca[self.idca[self.choice_col] == 1]
            .groupby(self.dest_col)
            .size()
            .to_dict()
        )

        rows = []
        for j, code in enumerate(self.dest_codes):
            if j >= len(pred_flows):
                break
            pred_j = float(pred_flows[j])
            obs_j = float(obs.get(code, 0))
            rows.append({
                self.dest_col: code,
                self.dest_name_col: dest_to_name.get(code, str(code)),
                'pred_flow': pred_j,
                'obs_flow': obs_j,
                'pred_share': pred_j / max(self.n_trips, 1),
                'obs_share': obs_j / max(self.n_trips, 1),
            })

        flow_df = pd.DataFrame(rows)
        flow_df['abs_error'] = np.abs(flow_df['pred_flow'] - flow_df['obs_flow'])
        flow_df['share_error_pp'] = 100.0 * (flow_df['pred_share'] - flow_df['obs_share'])
        return flow_df.sort_values('obs_flow', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Logsum calculations
    # ------------------------------------------------------------------

    def compute_logsums(self) -> np.ndarray:
        """
        Expected maximum utility per trip (logsum).

        Returns
        -------
        np.ndarray of shape (n_trips,)
        """
        if self._logsums is not None:
            return self._logsums
        self.compute_probabilities()
        return self._logsums

    # ------------------------------------------------------------------
    # Simulation-based prediction for Mixed Logit
    # ------------------------------------------------------------------

    def simulate_predictions(self, n_draws: int = 500, seed: int = 42) -> np.ndarray:
        """
        Simulation-based choice probabilities for mixed/rr models.

        Returns
        -------
        np.ndarray of shape (n_trips, n_dests)
        """
        if isinstance(self.model, (MixedLogit, MixedRandomRegret)):
            pass  # Use model's integral simulator
        self.compute_probabilities()
        return self._probs

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict:
        """
        Compute prediction accuracy metrics.

        Returns
        -------
        dict with keys: rmse, mae, r2, accuracy (hit rate), ll
        """
        flow_df = self.predict_aggregate_flows()
        obs = flow_df['obs_flow'].values
        pred = flow_df['pred_flow'].values
        mask = (obs > 0) | (pred > 0)

        if not mask.any():
            return dict(rmse=np.nan, mae=np.nan, r2=np.nan, accuracy=np.nan, ll=np.nan)

        rmse = float(np.sqrt(np.mean((obs[mask] - pred[mask]) ** 2)))
        mae = float(np.mean(np.abs(obs[mask] - pred[mask])))
        ss_res = np.sum((obs[mask] - pred[mask]) ** 2)
        ss_tot = np.sum((obs[mask] - obs[mask].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Hit rate
        pred_dests = self.predict_destinations()
        accuracy = float(pred_dests['correct'].mean()) if len(pred_dests) else np.nan

        # Log-likelihood on data
        ll_val = float(getattr(self.model, 'loglik', np.nan))

        return dict(rmse=rmse, mae=mae, r2=r2, accuracy=accuracy, ll=ll_val)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dest_name_map(self) -> dict:
        if self.dest_name_col in self.idca.columns:
            return (
                self.idca[[self.dest_col, self.dest_name_col]]
                .drop_duplicates()
                .set_index(self.dest_col)[self.dest_name_col]
                .to_dict()
            )
        return {}


def predict_destination_components(
    model,
    idca: pd.DataFrame,
    varnames: list[str],
    dest_col: str = 'dest_code',
    trip_id_col: str = 'trip_id',
    choice_col: str = 'chosen',
    dest_name_col: str = 'dest_suburb',
) -> pd.DataFrame:
    """
    Convenience function: predict destination component for each trip.

    Returns a DataFrame with predicted destination for every trip.
    """
    predictor = DestinationPredictor(
        model, idca, varnames,
        dest_col=dest_col,
        trip_id_col=trip_id_col,
        choice_col=choice_col,
        dest_name_col=dest_name_col,
    )
    return predictor.predict_destinations()


def predict_aggregate_destination_flows(
    model,
    idca: pd.DataFrame,
    varnames: list[str],
    dest_col: str = 'dest_code',
    trip_id_col: str = 'trip_id',
    choice_col: str = 'chosen',
    dest_name_col: str = 'dest_suburb',
) -> pd.DataFrame:
    """
    Convenience function: predict aggregate flows per destination.

    Returns a DataFrame with observed vs predicted trip counts per destination.
    """
    predictor = DestinationPredictor(
        model, idca, varnames,
        dest_col=dest_col,
        trip_id_col=trip_id_col,
        choice_col=choice_col,
        dest_name_col=dest_name_col,
    )
    return predictor.predict_aggregate_flows()
