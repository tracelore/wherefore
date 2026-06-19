"""
synthetic/corruptors/timezone_shift.py

Reference corruptor -- the import path
"wherefore.synthetic.corruptors.timezone_shift:apply" in
taxonomy/patterns/timezone_shift.yaml points HERE. This file's
function signature is therefore part of the taxonomy contract: every
other corruptor module should follow the same `apply()` shape so
ground_truth.py and the regenerate script can call any corruptor
uniformly via taxonomy.registry.resolve_import_path().

Contract (see CONTRIBUTING.md "Adding a new failure pattern"):
  - Takes a clean DataFrame, returns a CORRUPTED COPY plus the
    precise list of row indices that were modified.
  - Never mutates the input DataFrame in place -- ground_truth.py
    needs an intact, uncorrupted "source" to pair with the corrupted
    "target", and silent in-place mutation would destroy that.
  - affected_row_indices is REQUIRED, computed exactly at corruption
    time -- not inferred later by diffing -- because it's the
    eval-harness ground truth for cluster-level (not just
    pattern-level) scoring accuracy.
"""

from __future__ import annotations

import pandas as pd


def apply(
    df: pd.DataFrame,
    column: str,
    offset_hours: float = 5.0,
    affected_fraction: float = 0.3,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns (corrupted_df, affected_row_indices).

    Selects `affected_fraction` of rows at random (seeded) and adds
    `offset_hours` to their value in `column`, leaving all other rows
    untouched. This produces exactly the "constant_offset_subset"
    statistical signature timezone_shift.yaml's detection_hints
    describe: a SUBSET of rows sharing one constant time delta from
    their original value, while the rest of the column is unaffected.

    Raises if `column` isn't a datetime dtype -- silently no-op'ing on
    the wrong column type would produce a "corrupted" fixture that
    isn't actually corrupted, and a ground-truth label that lies.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise TypeError(
            f"timezone_shift.apply requires a datetime column, "
            f"got dtype {df[column].dtype} for column {column!r}"
        )

    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError(f"affected_fraction must be in (0, 1], got {affected_fraction}")

    corrupted = df.copy(deep=True)
    n_rows = len(corrupted)
    n_affected = max(1, round(n_rows * affected_fraction))

    rng = pd.Series(range(n_rows)).sample(
        n=n_affected, random_state=seed
    )
    affected_row_indices = sorted(rng.tolist())

    offset = pd.Timedelta(hours=offset_hours)
    corrupted.loc[affected_row_indices, column] = (
        corrupted.loc[affected_row_indices, column] + offset
    )

    return corrupted, affected_row_indices
