"""Blind consensus voting and cross-input core-program detection."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass

CORE_SIZE = 2


def output_key(output: str) -> str:
    """Hash an output after applying the judge's token normalization."""
    normalized = "\0".join(output.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def unique_mode(output_keys: list[str | None]) -> str | None:
    """Return the unique most frequent successful output, regardless of its ratio."""
    votes = Counter(key for key in output_keys if key is not None)
    if not votes:
        return None
    top = max(votes.values())
    winners = [key for key, count in votes.items() if count == top]
    return winners[0] if len(winners) == 1 else None


@dataclass(frozen=True)
class VoteDecision:
    mode: str | None
    trusted: bool
    reason: str
    successful_votes: int
    top_votes: int
    second_votes: int
    support_ratio: float
    margin: int
    vote_groups: list[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_vote(
    output_keys: list[str | None],
    *,
    min_supporters: int = 3,
    min_support_ratio: float = 0.70,
    min_margin: int = 2,
) -> VoteDecision:
    votes = Counter(key for key in output_keys if key is not None)
    groups = sorted(votes.values(), reverse=True)
    successful = sum(groups)
    top = groups[0] if groups else 0
    second = groups[1] if len(groups) > 1 else 0
    winners = [key for key, count in votes.items() if count == top]
    mode = winners[0] if len(winners) == 1 else None
    ratio = top / successful if successful else 0.0
    margin = top - second

    if not votes:
        reason = "no_successful_output"
    elif mode is None:
        reason = "tied_mode"
    elif top < min_supporters:
        reason = "insufficient_supporters"
    elif ratio < min_support_ratio:
        reason = "low_support_ratio"
    elif margin < min_margin:
        reason = "small_margin"
    else:
        reason = "trusted"
    return VoteDecision(
        mode=mode,
        trusted=reason == "trusted",
        reason=reason,
        successful_votes=successful,
        top_votes=top,
        second_votes=second,
        support_ratio=ratio,
        margin=margin,
        vote_groups=groups,
    )


def score_programs(
    outputs_by_case: list[list[str | None]],
    modes: list[str | None],
) -> list[dict[str, object]]:
    """Score mode agreement and mark the stable top two programs as core."""
    if len(outputs_by_case) != len(modes):
        raise ValueError("outputs and modes have different case counts")
    program_count = len(outputs_by_case[0]) if outputs_by_case else 0
    if any(len(outputs) != program_count for outputs in outputs_by_case):
        raise ValueError("inconsistent program counts")
    usable = [index for index, mode in enumerate(modes) if mode is not None]
    match_counts = []
    for program_index in range(program_count):
        matches = sum(outputs_by_case[case_index][program_index] == modes[case_index] for case_index in usable)
        match_counts.append(matches)
    ranked = sorted(range(program_count), key=lambda index: (-match_counts[index], index))
    core_indices = set(ranked[:CORE_SIZE]) if usable and program_count >= CORE_SIZE else set()
    scores = []
    for program_index, matches in enumerate(match_counts):
        ratio = matches / len(usable) if usable else 0.0
        scores.append(
            {
                "program_index": program_index,
                "matches": matches,
                "evaluated_cases": len(usable),
                "match_ratio": ratio,
                "is_core": program_index in core_indices,
            }
        )
    return scores


def unanimous_core_output(
    output_keys: list[str | None],
    core_indices: list[int],
) -> str | None:
    """Return the shared successful output of the two core programs."""
    selected = [output_keys[index] for index in core_indices]
    if len(selected) != CORE_SIZE or selected[0] is None or selected[0] != selected[1]:
        return None
    return selected[0]
