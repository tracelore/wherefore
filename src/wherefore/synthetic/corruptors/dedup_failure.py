"""
synthetic/corruptors/dedup_failure.py

Corrupts a dataset by duplicating a sample of rows and giving each
duplicate a NEW key -- simulating a migration retry that re-inserts
already-migrated records without deduplicating. Confirmed by direct
testing that real auto-generated-key systems (the overwhelmingly
common case) assign a NEW id on insert, so a re-inserted duplicate row
has different key but IDENTICAL content to its original -- not the
same key duplicated, which most diffing tools (including datacompy)
already catch trivially via a duplicate-key check. The harder, more
realistic case is what this corruptor produces.

Unlike every other corruptor in this taxonomy, dedup_failure's signal
does NOT show up in DiffResult.mismatches at all -- it shows up
entirely as extra rows in target_only_rows (confirmed by direct
testing: concatenating duplicated rows with new keys produces ZERO
column-level mismatches). Detection therefore happens via
clustering.cluster_mismatches.detect_row_presence_patterns, a
SEPARATE function from the column-mismatch clustering path -- see
that module's docstring for the full architectural reasoning.
"""

from __future__ import annotations

import pandas as pd


def apply(
    df: pd.DataFrame,
    key_column: str,
    affected_fraction: float = 0.15,
    new_key_prefix: str = "DUPE",
    seed: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (corrupted_df, new_keys_assigned_to_duplicates).

    Selects `affected_fraction` of rows at random (seeded), duplicates
    them, and assigns each duplicate a NEW key (via new_key_prefix +
    an index) rather than reusing the original row's key -- the
    realistic auto-generated-key scenario this corruptor models.

    Unlike every other corruptor's apply() contract, this one returns
    NEW KEYS (strings), not row indices -- there's no meaningful
    "affected row index" in the original DataFrame's index space,
    since the corruption ADDS rows rather than modifying existing
    ones. The returned keys are exactly what ground_truth.py needs to
    record as "these specific target-side keys are the injected
    duplicates."

    Raises if `key_column` isn't present, or if affected_fraction
    would select zero rows.
    """
    if key_column not in df.columns:
        raise ValueError(f"key_column {key_column!r} not found in DataFrame columns")

    if not 0.0 < affected_fraction <= 1.0:
        raise ValueError(f"affected_fraction must be in (0, 1], got {affected_fraction}")

    n_to_duplicate = max(1, round(len(df) * affected_fraction))
    sampled = df.sample(n=n_to_duplicate, random_state=seed).copy()

    new_keys = [f"{new_key_prefix}-{i}" for i in range(n_to_duplicate)]
    sampled[key_column] = new_keys

    corrupted = pd.concat([df, sampled], ignore_index=True)
    return corrupted, new_keys
