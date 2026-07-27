"""Community detection for the Zep organizer (paper §2.2.4).

Zep uses **label propagation**, not Leiden, and the paper is explicit about
why: propagation admits a single-step extension, so a new entity can join
the plurality community of its neighbours without recomputing the whole
partition, with a periodic full rebuild to undo the drift that accumulates
(docs/research/zep-graphiti.md §A.4). Both halves live here — the
algorithm and the two summarization prompts — while the organizer owns the
op emission and the store owns persistence.

``label_propagation`` is a transcription of upstream
``community_operations.label_propagation`` (getzep/graphiti), including two
behaviours its own docstring misdescribes; see the function for what they
are and why they are kept.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("agmem.organizers.zep_graph.community")

# Upstream text_utils.MAX_SUMMARY_CHARS — the cap on entity AND community
# summaries, enforced by truncating at a sentence boundary.
MAX_SUMMARY_CHARS = 1000

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}

DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}

# Upstream prompts/summarize_nodes.summarize_pair, condensed to one user turn.
# The negative rules ("avoid filler verbs like mentioned/described/...") are
# upstream's and are kept: they are what stops a map-reduce over N summaries
# from converging on "the summaries discuss various topics".
SUMMARIZE_PAIR_PROMPT = f"""Synthesize the information from the following two summaries into a
single information-dense summary.

IMPORTANT:
- Preserve all materially relevant names, roles, places, dates, counts,
  and changes over time that are explicitly supported.
- Prefer compact factual sentences over vague thematic phrasing.
- When the durable fact is the content of what was said, state the content
  directly instead of narrating that it was said.
- Avoid filler verbs like "mentioned", "described", "stated", "reported",
  "noted", "discussed", "referenced" and "indicated" unless the
  communication act itself matters.
- THE SUMMARY MUST BE LESS THAN {MAX_SUMMARY_CHARS} CHARACTERS.

Summaries:
1. {{left}}
2. {{right}}

Return JSON: {{{{"summary": "..."}}}}"""

# Upstream prompts/summarize_nodes.summary_description. The result is the
# community's NAME — an LLM-written one-sentence description of the summary,
# not a short label — and it is the field the community search channel
# embeds and BM25-indexes.
SUMMARY_DESCRIPTION_PROMPT = f"""Create a short one sentence description of the summary that
explains what kind of information is summarized. The description must be
under {MAX_SUMMARY_CHARS} characters.

Summary:
{{summary}}

Return JSON: {{{{"description": "..."}}}}"""


def truncate_at_sentence(text: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """Cut at the last sentence boundary that fits, else hard-cut at
    ``max_chars`` (upstream ``text_utils.truncate_at_sentence``)."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    matches = list(re.finditer(r"[.!?](?:\s|$)", truncated))
    if matches:
        return text[: matches[-1].end()].rstrip()
    return truncated.rstrip()


def label_propagation(
    projection: dict[str, dict[str, int]], max_rounds: int = 100
) -> list[list[str]]:
    """Cluster the entity subgraph by label propagation, returning member id
    lists (upstream ``community_operations.label_propagation``).

    Each node starts in its own community, then takes the community holding
    the plurality of its neighbours *weighted by edge count*, iterating until
    a round changes nothing.

    Two upstream behaviours are reproduced even though its own docstring
    describes something else, because they decide the published partition:

    - "Ties are broken by going to the largest community" — the code instead
      sorts ``(weight, community_label)`` descending and, on a weight tie,
      takes the numerically larger LABEL. Labels are enumeration indexes, so
      this is arbitrary-but-deterministic, not size-based.
    - A plurality of weight 1 does not win. ``candidate_rank > 1`` gates the
      move, so a node joined to its neighbourhood by exactly one edge falls
      through to ``max(candidate, current)`` — for a leaf node with a single
      relation this usually means keeping its own singleton community, which
      is why sparse graphs cluster far less than the paper's figures suggest.

    The one deliberate deviation is termination. Upstream loops ``while True``
    on a SYNCHRONOUS update — every node reassigned from the same snapshot —
    which is the variant of label propagation that can oscillate between two
    states instead of converging, and upstream has no iteration cap, so that
    is an unbounded hang rather than a different answer. It is not exotic:
    ANY two-node component joined by two or more edges reaches it, because
    each node's neighbour clears the ``> 1`` weight gate and the two swap
    labels every round. Two entities with two facts between them and no other
    relations is an ordinary shape in dialogue data.

    So this stops on a detected 2-cycle (the new assignment equals the one
    from two rounds back) and, as a backstop, at ``max_rounds``; both are
    logged. Termination does not invent a merge: in every state of such a
    cycle the two nodes hold different labels, so they end up as singleton
    communities. That is the honest reading of the published rule — it cannot
    cluster that component — rather than a partition upstream would have
    produced, since upstream produces none at all.

    Determinism: ids keep the ``projection`` insertion order, which the store
    returns in a fixed order, so the same graph yields the same partition.
    """
    community_map = {node_id: i for i, node_id in enumerate(projection)}

    rounds = 0
    converged = False
    oscillating = False
    previous: dict[str, int] | None = None
    while rounds < max_rounds and not converged and not oscillating:
        rounds += 1
        no_change = True
        new_community_map: dict[str, int] = {}
        for node_id, neighbors in projection.items():
            current = community_map[node_id]
            candidates: dict[int, int] = {}
            for neighbor_id, edge_count in neighbors.items():
                label = community_map.get(neighbor_id)
                if label is not None:
                    candidates[label] = candidates.get(label, 0) + edge_count
            ranked = sorted(((w, label) for label, w in candidates.items()), reverse=True)
            weight, candidate = ranked[0] if ranked else (0, -1)
            if candidate != -1 and weight > 1:
                new_label = candidate
            else:
                new_label = max(candidate, current)
            new_community_map[node_id] = new_label
            if new_label != current:
                no_change = False
        oscillating = previous is not None and new_community_map == previous
        previous = community_map
        community_map = new_community_map
        converged = no_change

    if oscillating:
        logger.warning(
            "label propagation is oscillating (2-cycle) after %d rounds over %d nodes; "
            "stopping. Upstream would loop forever here — the components involved "
            "cannot be clustered by this rule and stay singletons",
            rounds,
            len(projection),
        )
    elif not converged:
        logger.warning(
            "label propagation did not converge in %d rounds (%d nodes); "
            "returning the last assignment",
            max_rounds,
            len(projection),
        )

    clusters: dict[int, list[str]] = {}
    for node_id, label in community_map.items():
        clusters.setdefault(label, []).append(node_id)
    logger.debug(
        "label propagation: %d nodes -> %d communities in %d rounds",
        len(projection),
        len(clusters),
        rounds,
    )
    return list(clusters.values())
