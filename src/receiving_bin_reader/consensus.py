from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .validation import normalize_bin


@dataclass(frozen=True)
class RecognitionPass:
    model: str
    variant: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class ConsensusDecision:
    status: str
    bins: tuple[str, ...]
    candidate: str
    supporting_models: tuple[str, ...]
    reason: str


def _canonical_candidate(text: str) -> tuple[str, tuple[str, ...]]:
    parts = tuple(
        item
        for raw in str(text or "").split(",")
        if (item := normalize_bin(raw))
    )
    return ",".join(parts), parts


def decide_consensus(
    passes: list[RecognitionPass],
    approved_bins: set[str],
    *,
    min_distinct_models: int = 2,
) -> ConsensusDecision:
    if not approved_bins:
        raise ValueError("Approved BIN list is empty.")

    supporters: dict[str, set[str]] = defaultdict(set)
    parts_by_candidate: dict[str, tuple[str, ...]] = {}

    for result in passes:
        candidate, parts = _canonical_candidate(result.text)
        if not parts:
            continue
        if not all(part in approved_bins for part in parts):
            continue

        parts_by_candidate[candidate] = parts
        supporters[candidate].add(result.model)

    corroborated = [
        candidate
        for candidate, models in supporters.items()
        if len(models) >= min_distinct_models
    ]

    if len(corroborated) == 1:
        candidate = corroborated[0]
        return ConsensusDecision(
            status="ACCEPT",
            bins=parts_by_candidate[candidate],
            candidate=candidate,
            supporting_models=tuple(sorted(supporters[candidate])),
            reason="exact approved BIN corroborated by independent OCR models",
        )

    if not corroborated:
        return ConsensusDecision(
            status="REVIEW",
            bins=(),
            candidate="",
            supporting_models=(),
            reason="no exact approved BIN had sufficient independent-model support",
        )

    return ConsensusDecision(
        status="REVIEW",
        bins=(),
        candidate="",
        supporting_models=(),
        reason="multiple conflicting approved BIN candidates were corroborated",
    )
