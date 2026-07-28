"""Deterministic entity-dedup stage, ported from current graphiti main.

This is a transcription of ``graphiti_core/utils/maintenance/dedup_helpers.py``
(graphiti @ 9140123, 2026-07-26): the same normalization, entropy gate, MinHash
signature, LSH banding and Jaccard threshold, adapted only in its node shape —
upstream operates on ``EntityNode`` objects, here candidates are the plain
dicts our doc store returns (``{"id", "name", ...}``). Pure python + hashlib,
no model, no network — exactly like upstream's.

The stage upstream runs (``_resolve_with_similarity``, dedup_helpers.py:220-280):

1. exact normalized-name match (lowercase + whitespace-collapse,
   dedup_helpers.py:39-42) — always attempted, regardless of name length;
   exactly ONE candidate sharing the name resolves deterministically, MORE than
   one is ambiguous and escalates to the LLM instead of first-wins
   (dedup_helpers.py:245-249);
2. an entropy gate (Shannon entropy over characters >= 1.5, min length 6 or
   >= 2 tokens) protecting ONLY the fuzzy path — short or repetitive names
   produce unreliable shingle sets;
3. fuzzy MinHash/LSH: 3-gram shingles over the punctuation-stripped name,
   32 permutations, 4-row bands, Jaccard >= 0.9 against the best LSH-bucketed
   candidate.

Anything that clears none of these goes to the LLM dedupe call (one batched
call per message — see ``organizer.py``).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b

# Upstream dedup_helpers.py:31-36 — the exact constants.
_NAME_ENTROPY_THRESHOLD = 1.5
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2
_FUZZY_JACCARD_THRESHOLD = 0.9
_MINHASH_PERMUTATIONS = 32
_MINHASH_BAND_SIZE = 4


def _normalize_string_exact(name: str) -> str:
    """Lowercase + whitespace-collapse (dedup_helpers.py:39-42). This is the
    exact-match key AND the edge verbatim-duplicate key upstream shares it with
    (edge_operations.py imports the same function)."""
    normalized = re.sub(r"[\s]+", " ", name.lower())
    return normalized.strip()


def _normalize_name_for_fuzzy(name: str) -> str:
    """Fuzzier form keeping alphanumerics and apostrophes for n-gram shingles
    (dedup_helpers.py:45-49)."""
    normalized = re.sub(r"[^a-z0-9' ]", " ", _normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r"[\s]+", " ", normalized)


def _name_entropy(normalized_name: str) -> float:
    """Shannon entropy over characters (spaces stripped) — upstream's proxy for
    text specificity (dedup_helpers.py:52-76)."""
    if not normalized_name:
        return 0.0
    counts: dict[str, int] = {}
    for char in normalized_name.replace(" ", ""):
        counts[char] = counts.get(char, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _has_high_entropy(normalized_name: str) -> bool:
    """Gate for the FUZZY path only (dedup_helpers.py:79-85): very short or
    low-entropy names go to the LLM rather than being fuzzy-matched."""
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False
    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD


def _shingles(normalized_name: str) -> set[str]:
    """3-gram shingles over the space-stripped name (dedup_helpers.py:88-94)."""
    cleaned = normalized_name.replace(" ", "")
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


def _hash_shingle(shingle: str, seed: int) -> int:
    """Deterministic 64-bit blake2b hash per permutation seed
    (dedup_helpers.py:97-100)."""
    digest = blake2b(f"{seed}:{shingle}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def _minhash_signature(shingles: Iterable[str]) -> tuple[int, ...]:
    """MinHash signature across the 32 predefined permutations
    (dedup_helpers.py:103-114)."""
    if not shingles:
        return tuple()
    signature: list[int] = []
    for seed in range(_MINHASH_PERMUTATIONS):
        signature.append(min(_hash_shingle(shingle, seed) for shingle in shingles))
    return tuple(signature)


def _lsh_bands(signature: Iterable[int]) -> list[tuple[int, ...]]:
    """Fixed-size (4-row) LSH bands over the signature; a trailing partial band
    is dropped, as upstream drops it (dedup_helpers.py:117-128)."""
    signature_list = list(signature)
    if not signature_list:
        return []
    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _MINHASH_BAND_SIZE])
        if len(band) == _MINHASH_BAND_SIZE:
            bands.append(band)
    return bands


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard over shingle sets, empty-set edge cases as upstream
    (dedup_helpers.py:131-140)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures (upstream ``DedupCandidateIndexes``,
    dedup_helpers.py:149-157), over candidate dicts instead of EntityNodes."""

    existing_candidates: list[dict]
    candidates_by_id: dict[str, dict]
    normalized_existing: dict[str, list[dict]]
    shingles_by_candidate: dict[str, set[str]]
    lsh_buckets: dict[tuple[int, tuple[int, ...]], list[str]]


def build_candidate_indexes(candidates: list[dict]) -> DedupCandidateIndexes:
    """Upstream ``_build_candidate_indexes`` (dedup_helpers.py:192-217): the
    exact-name map plus the MinHash/LSH buckets, built once per dedupe run."""
    normalized_existing: defaultdict[str, list[dict]] = defaultdict(list)
    candidates_by_id: dict[str, dict] = {}
    shingles_by_candidate: dict[str, set[str]] = {}
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in candidates:
        candidate_id = str(candidate.get("id"))
        name = str(candidate.get("name", ""))
        normalized_existing[_normalize_string_exact(name)].append(candidate)
        candidates_by_id[candidate_id] = candidate

        shingles = _shingles(_normalize_name_for_fuzzy(name))
        shingles_by_candidate[candidate_id] = shingles
        signature = _minhash_signature(shingles)
        for band_index, band in enumerate(_lsh_bands(signature)):
            lsh_buckets[(band_index, band)].append(candidate_id)

    return DedupCandidateIndexes(
        existing_candidates=list(candidates),
        candidates_by_id=candidates_by_id,
        normalized_existing=dict(normalized_existing),
        shingles_by_candidate=shingles_by_candidate,
        lsh_buckets=dict(lsh_buckets),
    )


def deterministic_resolve(name: str, indexes: DedupCandidateIndexes) -> str | None:
    """Upstream ``_resolve_with_similarity`` for one extracted name
    (dedup_helpers.py:220-280): the matched candidate's id, or ``None`` when the
    LLM has to decide.

    ``None`` covers all three escalation cases upstream feeds to its LLM pass:
    no exact hit and no fuzzy hit; an AMBIGUOUS exact hit (>1 candidates
    sharing the normalized name — never first-wins, dedup_helpers.py:245-249);
    and an entropy-gated name that may not use the fuzzy path at all."""
    normalized_exact = _normalize_string_exact(name)
    normalized_fuzzy = _normalize_name_for_fuzzy(name)

    # --- exact-name matching (always attempted) ---
    exact_matches = indexes.normalized_existing.get(normalized_exact, [])
    if len(exact_matches) == 1:
        return str(exact_matches[0].get("id"))
    if len(exact_matches) > 1:
        return None  # ambiguous: escalate to the LLM, never first-wins

    # --- entropy gate (protects fuzzy matching only) ---
    if not _has_high_entropy(normalized_fuzzy):
        return None

    # --- fuzzy matching via MinHash/LSH ---
    shingles = _shingles(normalized_fuzzy)
    signature = _minhash_signature(shingles)
    candidate_ids: set[str] = set()
    for band_index, band in enumerate(_lsh_bands(signature)):
        candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

    best_candidate_id: str | None = None
    best_score = 0.0
    for candidate_id in candidate_ids:
        candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
        score = _jaccard_similarity(shingles, candidate_shingles)
        if score > best_score:
            best_score = score
            best_candidate_id = candidate_id

    if best_candidate_id is not None and best_score >= _FUZZY_JACCARD_THRESHOLD:
        return best_candidate_id
    return None


__all__ = [
    "DedupCandidateIndexes",
    "build_candidate_indexes",
    "deterministic_resolve",
    "_normalize_string_exact",
    "_normalize_name_for_fuzzy",
    "_has_high_entropy",
    "_jaccard_similarity",
    "_FUZZY_JACCARD_THRESHOLD",
]
