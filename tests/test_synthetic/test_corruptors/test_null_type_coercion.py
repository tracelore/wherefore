"""
Tests for synthetic/corruptors/null_type_coercion.py.
"""

import pandas as pd
import pytest

from wherefore.synthetic.base_dataset import FINANCIAL_ACCOUNTS, generate_dataset
from wherefore.synthetic.corruptors.null_type_coercion import apply


@pytest.fixture
def financial_source():
    return generate_dataset(FINANCIAL_ACCOUNTS, n_rows=50, seed=1)


def test_does_not_mutate_input_dataframe(financial_source):
    original = financial_source.copy(deep=True)
    apply(financial_source, column="last_transaction_at", sentinel="NULL", seed=1)
    pd.testing.assert_frame_equal(financial_source, original)


def test_only_genuinely_null_rows_are_eligible(financial_source):
    target, affected = apply(
        financial_source, column="last_transaction_at", sentinel="NULL", affected_fraction=1.0, seed=1
    )
    for idx in affected:
        assert pd.isna(financial_source.loc[idx, "last_transaction_at"])
        assert target.loc[idx, "last_transaction_at"] == "NULL"


def test_non_null_rows_never_affected(financial_source):
    target, affected = apply(
        financial_source, column="last_transaction_at", sentinel="NULL", affected_fraction=1.0, seed=1
    )
    non_null_indices = financial_source.index[financial_source["last_transaction_at"].notna()]
    assert not any(idx in affected for idx in non_null_indices)


def test_column_becomes_object_dtype_to_hold_mixed_values(financial_source):
    target, _ = apply(financial_source, column="last_transaction_at", sentinel="NULL", seed=1)
    assert target["last_transaction_at"].dtype == object


def test_raises_when_column_has_no_nulls():
    df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="at least one genuinely null value"):
        apply(df, column="val", sentinel="NULL")


def test_rejects_invalid_affected_fraction(financial_source):
    with pytest.raises(ValueError, match="affected_fraction"):
        apply(financial_source, column="last_transaction_at", affected_fraction=0.0)


def test_deterministic_given_same_seed(financial_source):
    target_a, affected_a = apply(financial_source, column="last_transaction_at", sentinel="NULL", seed=99)
    target_b, affected_b = apply(financial_source, column="last_transaction_at", sentinel="NULL", seed=99)
    assert affected_a == affected_b
    pd.testing.assert_frame_equal(target_a, target_b)


def test_custom_sentinel_string():
    df = pd.DataFrame({"id": [1, 2], "val": [1.5, None]})
    target, affected = apply(df, column="val", sentinel="N/A", affected_fraction=1.0, seed=1)
    assert target.loc[1, "val"] == "N/A"


def test_partial_affected_fraction_leaves_some_nulls_as_true_nulls():
    df = pd.DataFrame({"id": range(10), "val": [None] * 10})
    target, affected = apply(df, column="val", sentinel="NULL", affected_fraction=0.5, seed=1)
    assert len(affected) == 5
    remaining_null_count = target["val"].isna().sum()
    assert remaining_null_count == 5
