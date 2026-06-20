"""
Tests for synthetic/corruptors/float_precision.py.
"""

import numpy as np
import pandas as pd
import pytest

from wherefore.synthetic.base_dataset import FINANCIAL_ACCOUNTS, HEALTHCARE_PATIENTS, generate_dataset
from wherefore.synthetic.corruptors.float_precision import apply


@pytest.fixture
def financial_source():
    return generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)


def test_does_not_mutate_input_dataframe(financial_source):
    original = financial_source.copy(deep=True)
    apply(financial_source, column="balance", seed=1)
    pd.testing.assert_frame_equal(financial_source, original)


def test_affected_rows_are_exact_float32_round_trips(financial_source):
    target, affected = apply(financial_source, column="balance", affected_fraction=0.5, seed=1)
    assert len(affected) > 0
    for idx in affected:
        original = financial_source.loc[idx, "balance"]
        corrupted = target.loc[idx, "balance"]
        assert corrupted == float(np.float32(original))
        assert corrupted != original


def test_unaffected_rows_completely_untouched(financial_source):
    target, affected = apply(financial_source, column="balance", affected_fraction=0.5, seed=1)
    unaffected = [i for i in range(len(financial_source)) if i not in affected]
    assert (target.loc[unaffected, "balance"] == financial_source.loc[unaffected, "balance"]).all()


def test_other_columns_untouched(financial_source):
    target, _ = apply(financial_source, column="balance", seed=1)
    for col in financial_source.columns:
        if col == "balance":
            continue
        pd.testing.assert_series_equal(target[col], financial_source[col])


def test_null_values_never_selected():
    df = pd.DataFrame({"id": [1, 2, 3], "val": [1.5, None, 2.5]})
    target, affected = apply(df, column="val", affected_fraction=1.0, seed=1)
    assert 1 not in affected


def test_values_exactly_representable_in_float32_are_not_reported_as_affected():
    """
    Confirmed by direct testing: not every float loses precision when
    rounded through float32 (e.g. whole numbers, simple fractions like
    0.5 are exactly representable). A row whose round-trip doesn't
    actually change the value must not be reported as affected --
    same principle as truncation.py's "already shorter" guard.
    """
    df = pd.DataFrame({"id": [1], "val": [4.0]})
    target, affected = apply(df, column="val", affected_fraction=1.0, seed=1)
    assert affected == []
    assert target.loc[0, "val"] == 4.0


def test_rejects_non_float_column(financial_source):
    with pytest.raises(TypeError, match="requires a float column"):
        apply(financial_source, column="account_id")


def test_rejects_invalid_affected_fraction(financial_source):
    with pytest.raises(ValueError, match="affected_fraction"):
        apply(financial_source, column="balance", affected_fraction=0.0)


def test_deterministic_given_same_seed(financial_source):
    target_a, affected_a = apply(financial_source, column="balance", seed=99)
    target_b, affected_b = apply(financial_source, column="balance", seed=99)
    assert affected_a == affected_b
    pd.testing.assert_frame_equal(target_a, target_b)


def test_works_on_nullable_float64_dtype():
    source = generate_dataset(HEALTHCARE_PATIENTS, n_rows=30, seed=1)
    target, affected = apply(source, column="billed_amount", affected_fraction=0.5, seed=1)
    assert len(affected) > 0
