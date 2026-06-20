"""
synthetic/corruptors/float_precision.py

Corrupts a float column by rounding values through float32 precision
-- the classic "the target column was defined as REAL/FLOAT4 instead
of DOUBLE/FLOAT8" migration bug, or repeated arithmetic/serialization
through a lower-precision intermediate format. Confirmed by direct
testing: float32 rounding produces a tiny, non-zero delta whose
MAGNITUDE scales with the value itself (e.g. ~0.0003 lost on 128775.18,
~0.00004 lost on 7438.71) -- this is a genuinely different statistical
shape than any other corruptor in the taxonomy: not a constant offset,
not a prefix relationship, not a consistent value mapping. It's small,
value-proportional drift.

Follows the same apply() contract as the other corruptors (see
CONTRIBUTING.md): takes a clean DataFrame, returns a corrupted copy
plus the exact affected row indices, computed at corruption time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply(
    df: pd.DataFrame,
    column: str,
    affected_fraction: float = 0.5,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns (corrupted_df, affected_row_indices).

    Selects `affected_fraction` of rows at random (seeded) and rounds
    their value in `column` through float32 precision, leaving all
    other rows untouched. Null values are never selected -- there's
    no precision to lose on a value that isn't there.

    A row only counts as genuinely affected if the float32 round-trip
    actually CHANGED the value -- confirmed by direct testing that
    this isn't guaranteed for every float (some values happen to be
    exactly representable in float32, e.g. whole numbers or simple
    fractions like 0.5). Reporting an unaffected row as "affected"
    would make the ground truth lie about which rows the corruption
    actually touched, exactly the same principle as truncation.py's
    "already shorter than max_length" guard.

    Raises if `column` isn't a float dtype.
    """
    if not pd.api.types.is_float_dtype(df[column]):
        raise TypeError(
            f"float_precision.apply requires a float column, got dtype {df[column].dtype} for column {column!r}"
        )

    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError(f"affected_fraction must be in (0, 1], got {affected_fraction}")

    corrupted = df.copy(deep=True)
    non_null_indices = corrupted.index[corrupted[column].notna()].tolist()

    if not non_null_indices:
        return corrupted, []

    n_selected = max(1, round(len(non_null_indices) * affected_fraction))
    selected_indices = sorted(
        pd.Series(non_null_indices).sample(n=n_selected, random_state=seed).tolist()
    )

    affected_row_indices = []
    for idx in selected_indices:
        original_value = corrupted.at[idx, column]
        rounded_value = float(np.float32(original_value))
        if rounded_value != original_value:
            corrupted.at[idx, column] = rounded_value
            affected_row_indices.append(idx)

    return corrupted, affected_row_indices
