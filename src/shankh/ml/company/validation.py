"""
Walk-forward (expanding window) cross-validation for time-series data.

Design decisions
----------------
- Each fold uses ALL data up to a cutoff date as training data
  (expanding window) so older patterns are not discarded.
- A configurable ``gap`` (in unique trading dates) is left between the
  last training day and the first validation day to prevent look-ahead
  leakage from multi-day features (e.g. a 50-day MA on the boundary row).
- Folds are date-based rather than row-based so multi-ticker frames
  (where the same date appears once per ticker) split consistently.
- Returns index arrays rather than data copies to stay memory-efficient.

Call contract
-------------
Pass the pre-test DataFrame directly.  Do NOT pass test_cutoff — the
caller is responsible for slicing the frame before calling this function.
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



def walk_forward_folds(
    df: pd.DataFrame,
    n_splits:       int = 5,
    val_size:       int = 120,
    min_train_size: int = 250,
    gap:            int = 1,
) -> list[Fold]:
    """
    Generate expanding-window walk-forward folds.

    The frame passed in should already be restricted to the pre-test
    period.  This function does not perform any date-based cutoff itself.

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame with a ``date`` column (datetime64).
        Rows may span multiple tickers; the same date appears once per ticker.
    n_splits : int
        Number of CV folds to generate.
    val_size : int
        Number of *unique trading dates* per validation window.
    min_train_size : int
        Minimum number of unique trading dates required before the first
        validation window begins.
    gap : int
        Number of unique trading dates to skip between the last training
        date and the first validation date.  Prevents leakage from any
        feature whose look-back window spans the fold boundary.

    Returns
    -------
    list[Fold]
        Folds ordered earliest-first.  Each fold's ``train_idx`` /
        ``val_idx`` are positional (iloc) indices into ``df``.
    """
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    n_dates   = len(all_dates)

    required = min_train_size + gap + val_size
    if n_dates < required:
        raise ValueError(
            f"Not enough unique dates for walk-forward CV.\n"
            f"  Have : {n_dates} unique trading dates\n"
            f"  Need : {required}  (min_train={min_train_size} + gap={gap} + val_size={val_size})\n"
            f"  Fix  : reduce n_splits, val_size, or min_train_size in config."
        )

    first_valid_val_end = min_train_size + gap + val_size - 1
    val_end_positions = np.linspace(
        first_valid_val_end,
        n_dates - 1,
        num=n_splits,
        dtype=int,
    )
    val_end_positions = np.unique(val_end_positions)
    date_int64    = all_dates.asi8                                  # shape (n_dates,)
    df_date_int64 = pd.DatetimeIndex(df["date"]).asi8               # shape (n_rows,)
    df_date_pos   = np.searchsorted(date_int64, df_date_int64)
    df_date_pos   = np.clip(df_date_pos, 0, n_dates - 1)

    folds: list[Fold] = []
    for k, val_end_pos in enumerate(val_end_positions):
        val_start_pos  = int(val_end_pos) - val_size + 1
        train_end_pos  = val_start_pos - gap - 1
        train_start_pos = 0

        if train_end_pos < min_train_size - 1:
            logger.debug("Fold %d skipped — only %d training dates available (need %d).", k, train_end_pos + 1, min_train_size)
            continue

        # Positional range comparisons — no type issues, O(n) not O(n log n)
        train_mask = (df_date_pos >= train_start_pos) & (df_date_pos <= train_end_pos)
        val_mask   = (df_date_pos >= val_start_pos)   & (df_date_pos <= int(val_end_pos))

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.debug("Fold %d skipped — empty train or val split.", k)
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
        raise RuntimeError(
            "No valid CV folds generated.\n"
            f"  unique dates in frame : {n_dates}\n"
            f"  required per fold     : {required}\n"
            "  Reduce min_train_size or val_size in config."
        )

    logger.info(
        "Walk-forward CV: %d folds | val_size=%d dates | gap=%d | "
        "date range %s → %s",
        len(folds), val_size, gap,
        folds[0].train_start.date(), folds[-1].val_end.date(),
    )
    for f in folds:
        logger.debug(
            "  Fold %d  train %s→%s (%d rows)  val %s→%s (%d rows)",
            f.fold_idx,
            f.train_start.date(), f.train_end.date(), len(f.train_idx),
            f.val_start.date(),   f.val_end.date(),   len(f.val_idx),
        )

    return folds
