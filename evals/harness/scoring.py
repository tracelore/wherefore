"""
evals/harness/scoring.py

Pure scoring functions, given (prediction, ground truth) -> outcome.
Kept separate from run_eval.py's orchestration so scoring logic is
independently unit-testable with hand-constructed cases -- no need to
run the real pipeline or call an LLM to test that the outcome
classification logic is correct.

Outcome types, distinguished deliberately (see original design notes
in the project's eval harness stub): a system that correctly says
"unrecognized" on a genuinely unmatched case should NOT be scored the
same as a system that confidently names the WRONG pattern. Both are
"not a true positive," but they're very different failure modes worth
telling apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    TRUE_POSITIVE = "true_positive"  # correct pattern matched
    FALSE_POSITIVE = "false_positive"  # wrong pattern matched confidently
    HONEST_ABSTAIN = "honest_abstain"  # correctly said "unrecognized" for a genuinely unmatched case
    FALSE_ABSTAIN = "false_abstain"  # said "unrecognized" but a real pattern was injected
    FALSE_NEGATIVE = "false_negative"  # wrong pattern entirely (predicted a different real pattern than ground truth)


@dataclass
class ScoredCase:
    fixture_id: str
    column: str
    actual_pattern_id: str | None
    predicted_pattern_id: str | None
    outcome: Outcome


def score_pattern_match(actual_pattern_id: str | None, predicted_pattern_id: str | None) -> Outcome:
    """
    The core classification, independent of any fixture/cluster
    plumbing -- given what was ACTUALLY injected (None if nothing was)
    and what the system PREDICTED (None if it said unrecognized),
    returns the outcome type.

    Use this for scoring explain()'s output specifically -- it commits
    to exactly ONE matched_pattern_id (forced by the tool-use schema in
    providers/claude.py), so exact equality is the right test of
    whether the final, human-facing answer is correct.
    """
    if actual_pattern_id is None and predicted_pattern_id is None:
        return Outcome.HONEST_ABSTAIN
    if actual_pattern_id is None and predicted_pattern_id is not None:
        return Outcome.FALSE_POSITIVE
    if actual_pattern_id is not None and predicted_pattern_id is None:
        return Outcome.FALSE_ABSTAIN
    if actual_pattern_id == predicted_pattern_id:
        return Outcome.TRUE_POSITIVE
    return Outcome.FALSE_NEGATIVE


def score_pattern_match_against_candidates(
    actual_pattern_id: str | None, predicted_pattern_ids: list[str]
) -> Outcome:
    """
    Like score_pattern_match, but for clustering's statistical output
    specifically, which can legitimately report MULTIPLE candidate
    patterns for one cluster (see cluster_mismatches.py's "On multiple
    legitimate matches" docstring -- a genuine null coerced to a
    sentinel string also satisfies consistent_value_mapping, and
    clustering deliberately does not suppress either candidate).

    Scoring only the first item in that list would test an accident of
    registry insertion order, not anything clustering actually
    promises -- confirmed directly: a real fixture's true pattern
    (null_type_coercion) was scored as a false_negative purely because
    a co-occurring candidate (enum_drift) happened to appear first in
    the list, despite the true pattern being correctly present.

    The honest question for clustering's job is set membership: did
    the statistical layer surface the true cause AS A CANDIDATE, even
    if alongside another legitimate one? That's what this function
    tests. It does NOT apply to explain()'s output -- that commits to
    exactly one matched_pattern_id by design (forced tool-use), so
    score_pattern_match's exact-equality test is the right one there.
    """
    predicted_set = set(predicted_pattern_ids)

    if actual_pattern_id is None and not predicted_set:
        return Outcome.HONEST_ABSTAIN
    if actual_pattern_id is None and predicted_set:
        return Outcome.FALSE_POSITIVE
    if actual_pattern_id is not None and not predicted_set:
        return Outcome.FALSE_ABSTAIN
    if actual_pattern_id in predicted_set:
        return Outcome.TRUE_POSITIVE
    return Outcome.FALSE_NEGATIVE


@dataclass
class PatternMetrics:
    pattern_id: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else None

    @property
    def recall(self) -> float | None:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else None


def compute_metrics_by_pattern(scored_cases: list[ScoredCase]) -> dict[str, PatternMetrics]:
    """
    Computes precision/recall PER PATTERN, standard multi-class
    definitions:
      - precision for pattern X: of cases PREDICTED as X, how many
        were ACTUALLY X
      - recall for pattern X: of cases that were ACTUALLY X, how many
        did the system correctly predict as X

    A false_negative for pattern X is any case where ground truth was
    X but the prediction was something else (wrong pattern OR
    false_abstain) -- both count against X's recall, since from X's
    perspective, X was missed either way.

    honest_abstain cases don't affect any pattern's precision/recall
    -- they're tracked in the overall summary but aren't a false
    negative FOR any specific pattern, since no pattern was supposed
    to be detected in that case.
    """
    metrics: dict[str, PatternMetrics] = {}

    def _get(pattern_id: str) -> PatternMetrics:
        if pattern_id not in metrics:
            metrics[pattern_id] = PatternMetrics(pattern_id=pattern_id)
        return metrics[pattern_id]

    for case in scored_cases:
        if case.outcome == Outcome.TRUE_POSITIVE:
            _get(case.actual_pattern_id).true_positives += 1
        elif case.outcome == Outcome.FALSE_POSITIVE:
            _get(case.predicted_pattern_id).false_positives += 1
        elif case.outcome == Outcome.FALSE_ABSTAIN:
            _get(case.actual_pattern_id).false_negatives += 1
        elif case.outcome == Outcome.FALSE_NEGATIVE:
            # Wrong pattern entirely: counts as a false_negative for the
            # ACTUAL pattern (it was missed) AND a false_positive for
            # the PREDICTED pattern (it was wrongly claimed).
            _get(case.actual_pattern_id).false_negatives += 1
            if case.predicted_pattern_id is not None:
                _get(case.predicted_pattern_id).false_positives += 1
        # HONEST_ABSTAIN intentionally affects no pattern's metrics.

    return metrics


@dataclass
class RunSummary:
    total_cases: int
    outcome_counts: dict[str, int]
    metrics_by_pattern: dict[str, PatternMetrics]

    def overall_accuracy(self) -> float:
        """Fraction of cases where the system did the RIGHT thing --
        either naming the correct pattern, or correctly abstaining."""
        if self.total_cases == 0:
            return 0.0
        correct = self.outcome_counts.get(Outcome.TRUE_POSITIVE.value, 0) + self.outcome_counts.get(
            Outcome.HONEST_ABSTAIN.value, 0
        )
        return correct / self.total_cases


def summarize_run(scored_cases: list[ScoredCase]) -> RunSummary:
    outcome_counts: dict[str, int] = {}
    for case in scored_cases:
        outcome_counts[case.outcome.value] = outcome_counts.get(case.outcome.value, 0) + 1

    return RunSummary(
        total_cases=len(scored_cases),
        outcome_counts=outcome_counts,
        metrics_by_pattern=compute_metrics_by_pattern(scored_cases),
    )
