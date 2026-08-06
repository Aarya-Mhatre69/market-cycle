"""
Walk-forward (expanding window) cross-validation and Split Conformal Calibration.

Prevents lookahead bias and computes exact non-conformity adjustments (Q_conf)
to lock out-of-sample empirical coverage at nominal targets.
"""

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    fold_idx:    int
    train_idx:   np.ndarray   # integer positional indices into the passed df
    val_idx:     np.ndarray
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp


def compute_conformal_adjustment(
    val_actual_high_ret: np.ndarray,
    val_actual_low_ret: np.ndarray,
    val_pred_upper_ret: np.ndarray,
    val_pred_lower_ret: np.ndarray,
    target_coverage: float = 0.68,
) -> float:
    """
    Compute Split Conformal Prediction Non-Conformity Adjustment Q_conf.

    Calculates the minimum symmetric expansion factor needed on validation residual
    errors to achieve exact target empirical coverage.
    """
    # Non-conformity score: maximum violation distance in log-return space
    upper_err = val_actual_high_ret - val_pred_upper_ret
    lower_err = val_pred_lower_ret - val_actual_low_ret
    non_conformity_scores = np.maximum(upper_err, lower_err)

    # Compute empirical quantile at target_coverage (1 - alpha)
    n = len(non_conformity_scores)
    if n == 0:
        return 0.0

    quantile_level = np.clip(target_coverage * (1.0 + 1.0 / n), 0.0, 1.0)
    q_conf = float(np.quantile(non_conformity_scores, quantile_level))

    logger.info(
        "Conformal Calibration computed | N_val=%d | Target Coverage=%.2f | Q_conf=%.5f",
        n, target_coverage, q_conf
    )
    return max(0.0, q_conf)


def walk_forward_folds(
    df: pd.DataFrame,
    n_splits:       int = 5,
    val_size:       int = 120,
    min_train_size: int = 250,
    gap:            int = 1,
) -> list[Fold]:
    """Generate expanding-window walk-forward folds."""
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    n_dates   = len(all_dates)

    required = min_train_size + gap + val_size
    if n_dates < required:
        raise ValueError(
            f"Not enough unique dates for walk-forward CV.\n"
            f"  Have : {n_dates} unique trading dates\n"
            f"  Need : {required}  (min_train={min_train_size} + gap={gap} + val_size={val_size})\n"
        )

    first_valid_val_end = min_train_size + gap + val_size - 1
    val_end_positions = np.linspace(
        first_valid_val_end,
        n_dates - 1,
        num=n_splits,
        dtype=int,
    )
    val_end_positions = np.unique(val_end_positions)
    date_int64    = all_dates.asi8
    df_date_int64 = pd.DatetimeIndex(df["date"]).asi8
    df_date_pos   = np.searchsorted(date_int64, df_date_int64)
    df_date_pos   = np.clip(df_date_pos, 0, n_dates - 1)

    folds: list[Fold] = []
    for k, val_end_pos in enumerate(val_end_positions):
        val_start_pos  = int(val_end_pos) - val_size + 1
        train_end_pos  = val_start_pos - gap - 1
        train_start_pos = 0

        if train_end_pos < min_train_size - 1:
            continue

        train_mask = (df_date_pos >= train_start_pos) & (df_date_pos <= train_end_pos)
        val_mask   = (df_date_pos >= val_start_pos)   & (df_date_pos <= int(val_end_pos))

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        folds.append(
            Fold(
                fold_idx    = k,
                train_idx   = train_idx,
                val_idx     = val_idx,
                train_start = all_dates[train_start_pos],
                train_end   = all_dates[train_end_pos],
                val_start   = all_dates[val_start_pos],
                val_end     = all_dates[int(val_end_pos)],
            )
        )

    if not folds:
        raise RuntimeError("No valid CV folds generated. Reduce min_train_size in config.")

    logger.info(
        "Walk-forward CV: %d folds | val_size=%d dates | gap=%d | date range %s → %s",
        len(folds), val_size, gap, folds[0].train_start.date(), folds[-1].val_end.date(),
    )
    return folds