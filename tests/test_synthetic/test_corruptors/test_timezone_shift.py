"""
Tests for synthetic/corruptors/timezone_shift.py -- the reference
corruptor. These tests double as a proof that the YAML pattern's
detection_hints description ("a subset of mismatched rows... differ
by the same constant time delta") accurately describes what this
corruptor actually produces -- see TAXONOMY_TODO.md on why patterns
are validated against real corruptor output, not designed in advance
of it.
"""

import pandas as pd
import pytest

from wherefore.synthetic.base_dataset import (
    FINANCIAL_ACCOUNTS,
    HEALTHCARE_PATIENTS,
    generate_dataset,
)
from wherefore.synthetic.corruptors.timezone_shift import apply


@pytest.fixture
def financial_source():
    return generate_dataset(FINANCIAL_ACCOUNTS, n_rows=50, seed=42)


def test_does_not_mutate_input_dataframe(financial_source):
    original = financial_source.copy(deep=True)
    apply(financial_source, column="opened_at", offset_hours=5.0, seed=1)
    pd.testing.assert_frame_equal(financial_source, original)


def test_affected_fraction_controls_row_count(financial_source):
    _, affected = apply(
        financial_source, column="opened_at", affected_fraction=0.3, seed=1
    )
    expected = round(len(financial_source) * 0.3)
    assert len(affected) == expected


def test_affected_rows_shift_by_exact_constant_offset(financial_source):
    """
    This IS the constant_offset_subset signature described in
    timezone_shift.yaml's detection_hints -- every affected row must
    differ from source by exactly the same delta.
    """
    target, affected = apply(
        financial_source, column="opened_at", offset_hours=5.0, affected_fraction=0.3, seed=1
    )
    deltas = (target.loc[affected, "opened_at"] - financial_source.loc[affected, "opened_at"])
    assert deltas.nunique() == 1
    assert deltas.iloc[0] == pd.Timedelta(hours=5.0)


def test_unaffected_rows_are_completely_untouched(financial_source):
    target, affected = apply(
        financial_source, column="opened_at", affected_fraction=0.3, seed=1
    )
    unaffected = [i for i in range(len(financial_source)) if i not in affected]
    assert (
        target.loc[unaffected, "opened_at"] == financial_source.loc[unaffected, "opened_at"]
    ).all()


def test_other_columns_are_completely_untouched(financial_source):
    target, _ = apply(financial_source, column="opened_at", seed=1)
    for col in financial_source.columns:
        if col == "opened_at":
            continue
        pd.testing.assert_series_equal(target[col], financial_source[col])


def test_rejects_non_datetime_column(financial_source):
    with pytest.raises(TypeError, match="requires a datetime column"):
        apply(financial_source, column="customer_name")


def test_rejects_invalid_affected_fraction(financial_source):
    with pytest.raises(ValueError, match="affected_fraction"):
        apply(financial_source, column="opened_at", affected_fraction=1.5)
    with pytest.raises(ValueError, match="affected_fraction"):
        apply(financial_source, column="opened_at", affected_fraction=0.0)


def test_deterministic_given_same_seed(financial_source):
    target_a, affected_a = apply(financial_source, column="opened_at", seed=99)
    target_b, affected_b = apply(financial_source, column="opened_at", seed=99)
    assert affected_a == affected_b
    pd.testing.assert_frame_equal(target_a, target_b)


def test_works_on_healthcare_domain_too():
    """
    Proves the corruptor is domain-agnostic, as required by the
    contract in CONTRIBUTING.md -- same function, different domain,
    different column name, different offset.
    """
    source = generate_dataset(HEALTHCARE_PATIENTS, n_rows=40, seed=42)
    target, affected = apply(
        source, column="encounter_date", offset_hours=9.0, affected_fraction=0.25, seed=3
    )
    deltas = target.loc[affected, "encounter_date"] - source.loc[affected, "encounter_date"]
    assert deltas.nunique() == 1
    assert deltas.iloc[0] == pd.Timedelta(hours=9.0)
