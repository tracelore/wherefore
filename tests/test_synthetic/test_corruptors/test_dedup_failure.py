"""
Tests for synthetic/corruptors/dedup_failure.py.
"""

import pandas as pd
import pytest

from wherefore.synthetic.base_dataset import FINANCIAL_ACCOUNTS, generate_dataset
from wherefore.synthetic.corruptors.dedup_failure import apply


@pytest.fixture
def financial_source():
    return generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)


def test_does_not_mutate_input_dataframe(financial_source):
    original = financial_source.copy(deep=True)
    apply(financial_source, key_column="account_id", seed=1)
    pd.testing.assert_frame_equal(financial_source, original)


def test_duplicated_rows_have_new_keys_not_reused_keys(financial_source):
    target, new_keys = apply(financial_source, key_column="account_id", affected_fraction=0.2, seed=1)
    original_keys = set(financial_source["account_id"])
    for k in new_keys:
        assert k not in original_keys


def test_duplicated_rows_have_identical_content_to_their_original():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "val": [10, 20, 30]})
    target, new_keys = apply(df, key_column="id", affected_fraction=1.0, seed=1)

    duplicated_rows = target[target["id"].isin(new_keys)]
    for _, dupe_row in duplicated_rows.iterrows():
        matches = df[(df["name"] == dupe_row["name"]) & (df["val"] == dupe_row["val"])]
        assert len(matches) == 1


def test_target_has_more_rows_than_source(financial_source):
    target, new_keys = apply(financial_source, key_column="account_id", affected_fraction=0.2, seed=1)
    assert len(target) == len(financial_source) + len(new_keys)


def test_rejects_missing_key_column(financial_source):
    with pytest.raises(ValueError, match="not found"):
        apply(financial_source, key_column="not_a_real_column")


def test_rejects_invalid_affected_fraction(financial_source):
    with pytest.raises(ValueError, match="affected_fraction"):
        apply(financial_source, key_column="account_id", affected_fraction=0.0)


def test_custom_new_key_prefix():
    df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    target, new_keys = apply(df, key_column="id", affected_fraction=1.0, new_key_prefix="RETRY", seed=1)
    assert all(k.startswith("RETRY-") for k in new_keys)


def test_deterministic_given_same_seed(financial_source):
    target_a, keys_a = apply(financial_source, key_column="account_id", seed=99)
    target_b, keys_b = apply(financial_source, key_column="account_id", seed=99)
    assert keys_a == keys_b
    pd.testing.assert_frame_equal(target_a, target_b)
