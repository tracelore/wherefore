"""
Tests for clustering/cluster_mismatches.py's detect_row_presence_patterns
and RowPresenceCluster -- the architectural extension for patterns whose
signal shows up as extra/missing rows rather than column-level mismatches.
"""

import pytest

from wherefore.clustering.cluster_mismatches import (
    RowPresenceCluster,
    detect_row_presence_patterns,
)
from wherefore.comparison.diff_engine import compare
from wherefore.synthetic.base_dataset import FINANCIAL_ACCOUNTS, generate_dataset
from wherefore.synthetic.corruptors.dedup_failure import apply as inject_dedup_failure


def test_real_dedup_failure_fixture_is_detected_with_dataframes():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)
    target, new_keys = inject_dedup_failure(source, key_column="account_id", affected_fraction=0.15, seed=1)

    result = compare(source, target, join_columns="account_id")
    clusters = detect_row_presence_patterns(result, source_df=source, target_df=target)

    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.side == "target_only"
    assert cluster.is_unrecognized is False
    assert cluster.candidate_patterns[0].pattern_id == "dedup_failure"
    assert cluster.candidate_patterns[0].confidence == 1.0


def test_genuinely_new_rows_are_honestly_unrecognized():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)
    target = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=35, seed=1)

    result = compare(source, target, join_columns="account_id")
    clusters = detect_row_presence_patterns(result, source_df=source, target_df=target)

    assert len(clusters) == 1
    assert clusters[0].is_unrecognized is True
    assert clusters[0].candidate_patterns == []


def test_no_dataframes_provided_degrades_to_unrecognized_not_crash():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)
    target, _ = inject_dedup_failure(source, key_column="account_id", affected_fraction=0.15, seed=1)
    result = compare(source, target, join_columns="account_id")

    clusters = detect_row_presence_patterns(result)
    assert len(clusters) == 1
    assert clusters[0].is_unrecognized is True


def test_no_unmatched_rows_produces_no_clusters():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=20, seed=1)
    result = compare(source, source.copy(), join_columns="account_id")
    clusters = detect_row_presence_patterns(result, source_df=source, target_df=source)
    assert clusters == []


def test_source_only_side_is_detected_independently_of_target_only():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=30, seed=1)
    target, new_keys = inject_dedup_failure(source, key_column="account_id", affected_fraction=0.15, seed=1)
    target = target[target["account_id"] != source.iloc[0]["account_id"]].reset_index(drop=True)

    result = compare(source, target, join_columns="account_id")
    clusters = detect_row_presence_patterns(result, source_df=source, target_df=target)

    sides = {c.side for c in clusters}
    assert sides == {"source_only", "target_only"}

    target_only_cluster = next(c for c in clusters if c.side == "target_only")
    assert target_only_cluster.candidate_patterns[0].pattern_id == "dedup_failure"

    source_only_cluster = next(c for c in clusters if c.side == "source_only")
    assert source_only_cluster.is_unrecognized is True


def test_invalid_confidence_threshold_raises():
    source = generate_dataset(FINANCIAL_ACCOUNTS, n_rows=20, seed=1)
    target, _ = inject_dedup_failure(source, key_column="account_id", seed=1)
    result = compare(source, target, join_columns="account_id")

    with pytest.raises(ValueError, match="confidence_threshold"):
        detect_row_presence_patterns(result, source_df=source, target_df=target, confidence_threshold=1.5)


def test_row_presence_cluster_has_no_narrative_field():
    from wherefore.clustering.cluster_mismatches import RowPresenceMatch

    cluster_fields = set(RowPresenceCluster.__dataclass_fields__)
    match_fields = set(RowPresenceMatch.__dataclass_fields__)
    forbidden_words = {"narrative", "explanation", "cause", "reason"}

    assert not (cluster_fields & forbidden_words)
    assert not (match_fields & forbidden_words)
