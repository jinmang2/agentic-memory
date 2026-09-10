from __future__ import annotations

from typing import Literal, TypedDict

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

CitationStatus = Literal["valid", "missing", "invalid", "unavailable"]
DistillStatus = Literal["completed", "skipped", "partial", "failed"]


class CitationQuality(TypedDict):
    status: CitationStatus
    reason: str
    raw_steps: list[int] | None
    step_range: list[int] | None


class SourceCoverage(TypedDict):
    total_steps: int
    model_visible_steps: int
    model_visible_ratio: float
    visible_step_semantics: str
    source_chars: int
    rendered_chars: int
    rendered_char_ratio: float
    has_clipped_steps: bool
    cited_steps: int
    cited_ratio: float
    source_episode_count: int
    scope: str


class FactBasis(TypedDict):
    basis: str
    outcome_basis: str
    includes_tool_result: bool
    verified_result_claim: bool


class RunbookQuality(TypedDict):
    citation: CitationQuality
    source_coverage: SourceCoverage
    fact_basis: FactBasis


def citation_quality(
    raw_steps: JsonValue,
    cited: list[int] | None,
    step_range: list[int] | None,
    n_steps: int,
    visible: frozenset[int],
    source_episode_count: int,
    rendered_transcript: str,
    source_chars: int,
) -> RunbookQuality:
    raw = _raw_ints(raw_steps)
    status, reason = _citation_status(raw_steps, raw, cited, n_steps, visible)
    visible_count = len(visible)
    cited_count = len(cited) if cited is not None else 0
    scope = "cited_steps" if status == "valid" else "whole_session_fallback"
    return {
        "citation": {
            "status": status,
            "reason": reason,
            "raw_steps": raw,
            "step_range": step_range,
        },
        "source_coverage": {
            "total_steps": n_steps,
            "model_visible_steps": visible_count,
            "model_visible_ratio": _ratio(visible_count, n_steps),
            "visible_step_semantics": "step_presence_not_full_text",
            "source_chars": source_chars,
            "rendered_chars": len(rendered_transcript),
            "rendered_char_ratio": _ratio(len(rendered_transcript), source_chars),
            "has_clipped_steps": "chars omitted" in rendered_transcript,
            "cited_steps": cited_count,
            "cited_ratio": _ratio(cited_count, n_steps),
            "source_episode_count": source_episode_count,
            "scope": scope,
        },
        "fact_basis": {
            "basis": "transcript_citation" if status == "valid" else "session_fallback",
            "outcome_basis": "model_label_unverified",
            "includes_tool_result": False,
            "verified_result_claim": False,
        },
    }


def attach_fact_basis(
    quality: RunbookQuality, trajectory: list[dict[str, JsonValue]], cited: list[int] | None
) -> None:
    chosen = range(len(trajectory)) if cited is None else cited
    quality["fact_basis"]["includes_tool_result"] = any(
        str(trajectory[i].get("kind") or trajectory[i].get("role") or "") == "tool_result"
        for i in chosen
        if 0 <= i < len(trajectory)
    )


def distill_payload(
    *,
    status: DistillStatus,
    reason: str,
    meta: dict[str, JsonValue],
    summary: str,
    outputs: int,
    skipped_outputs: int = 0,
) -> dict[str, JsonValue]:
    retryable = status == "failed"
    review_required = status in {"failed", "partial"}
    return {
        "kind": "experience_distill",
        "distill_status": status,
        "terminal": not retryable,
        "retryable": retryable,
        "review_required": review_required,
        "reason": reason,
        "summary": summary,
        "outputs": outputs,
        "skipped_outputs": skipped_outputs,
        "session_id": meta["session_id"],
        "source_host": meta["host"],
        "origin": meta["origin"],
    }


def task_fingerprint(block: dict[str, JsonValue]) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "name",
        "outcome",
        "stage",
        "preference_signals",
        "reusable_knowledge",
        "failures",
        "references",
        "procedure",
        "keywords",
    ):
        value = block.get(key)
        if isinstance(value, list):
            values.append("\n".join(str(item) for item in value))
        else:
            values.append(str(value or ""))
    return tuple(values)


def _citation_status(
    raw_steps: JsonValue,
    raw: list[int] | None,
    cited: list[int] | None,
    n_steps: int,
    visible: frozenset[int],
) -> tuple[CitationStatus, str]:
    if raw_steps is None:
        return "missing", "steps_missing"
    if raw is None or not raw:
        return "invalid", "steps_not_integer_list"
    if cited is not None:
        return "valid", "steps_valid"
    if min(raw) < 0 or max(raw) >= n_steps:
        return "invalid", "steps_out_of_bounds"
    return "invalid", "steps_outside_model_visible_transcript"


def _raw_ints(value: JsonValue) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ints: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            return None
        ints.append(item)
    return ints


def _ratio(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part / whole, 3)
