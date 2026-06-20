"""
synthetic/corruptors/null_type_coercion.py

Corrupts a nullable column by converting genuine nulls (None/NaN/NaT)
into a literal SENTINEL STRING -- the classic "the target system wrote
the word NULL instead of leaving the cell empty" migration bug. This
is a real, common failure: many ETL tools and legacy export formats
write nulls as the literal text "NULL", "N/A", or "None" rather than
a true empty/null value, and downstream systems that don't expect this
treat it as real data (e.g. a string column containing the text
"NULL" four bytes long, not an actual null).

Deliberately the INVERSE direction of what loaders.py guards against:
loaders.py preserves the distinction between a literal "NULL" string
and a true empty cell when READING a CSV. This corruptor WRITES that
exact confusion into a fixture on purpose, so the taxonomy has
something real to detect.

Follows the same apply() contract as the other corruptors (see
CONTRIBUTING.md): takes a clean DataFrame, returns a corrupted copy
plus the exact affected row indices, computed at corruption time.
"""

from __future__ import annotations

import pandas as pd


def apply(
    df: pd.DataFrame,
    column: str,
    sentinel: str = "NULL",
    affected_fraction: float = 0.6,
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Returns (corrupted_df, affected_row_indices).

    Finds rows where `column` is genuinely null (covers None, NaN, and
    NaT via pandas' isna(), so this works uniformly across numeric,
    string, and datetime columns) and converts `affected_fraction` of
    THOSE rows' null value to the literal string `sentinel`, leaving
    the rest as true nulls. Non-null rows are never touched.

    Only rows that were genuinely null can be "affected" -- there's no
    null to coerce in a row that already has real data, so
    affected_row_indices is always a subset of the column's null rows,
    never anything else.

    Raises if the column has no genuine nulls at all -- a corruption
    with nothing to corrupt would produce a ground-truth label that
    lies about what happened.
    """
    null_mask = df[column].isna()
    null_indices = df.index[null_mask].tolist()

    if not null_indices:
        raise ValueError(
            f"null_type_coercion.apply requires column {column!r} to have at least "
            "one genuinely null value to corrupt; found none."
        )

    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError(f"affected_fraction must be in (0, 1], got {affected_fraction}")

    corrupted = df.copy(deep=True)

    # Coercing a null to a string sentinel means the column can no
    # longer be purely numeric/datetime dtype -- it must become an
    # object/string column to hold both real values and the sentinel
    # text side by side, exactly as a real schema-mismatched migration
    # would produce (e.g. a NUMERIC column becoming VARCHAR on the
    # target because of one bad write).
    corrupted[column] = corrupted[column].astype(object)

    n_selected = max(1, round(len(null_indices) * affected_fraction))
    selected_indices = sorted(
        pd.Series(null_indices).sample(n=n_selected, random_state=seed).tolist()
    )

    for idx in selected_indices:
        corrupted.at[idx, column] = sentinel

    return corrupted, selected_indices
