"""AgenticMemory facade — the public API (docs/05 §1).

Write path (docs/04 §2): raw episode is stored and indexed synchronously
(immediately searchable), then organizers run and their MemoryOps are
logged append-only before being applied to stores.
"""

from __future__ import annotations

import inspect
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Self

from agmem.capabilities import detect, resolve
from agmem.capabilities.detect import HostCapabilities
from agmem.config import AgmemConfig, load_config
from agmem.core.ops import MemoryOp, OpType
from agmem.core.origin import item_cwd, same_project
from agmem.core.types import (
    BITEMPORAL_TYPES,
    Episode,
    MemoryBundle,
    render_bullet_line,
    utcnow,
)
from agmem.embed import EMBEDDER_CANDIDATES
from agmem.embed.base import Embedder
from agmem.env import DEFAULT_NAMESPACE
from agmem.llm import BudgetTracker, LLMClient, StructuredCaller
from agmem.organizers import ORGANIZERS, MemoryEvent, Organizer, OrganizerContext
from agmem.retrieval import RetrievalPipeline
from agmem.retrieval.rerank import RERANKER_CANDIDATES
from agmem.stores import DOC_STORE_CANDIDATES, VECTOR_STORE_CANDIDATES

if TYPE_CHECKING:  # `agmem.sessions` is a leaf parser; the import stays lazy anyway
    from agmem.sessions import SessionTrajectory

logger = logging.getLogger("agmem")

# BITEMPORAL_TYPES now lives in core.types (imported above, so
# `memory.BITEMPORAL_TYPES` still resolves — docs/04 §2 cites that path). It
# moved because the write side here (drop the vector on INVALIDATE, or keep it)
# and the read side (retrieval.steps.is_servable) must agree on one list, and
# retrieval cannot import this module.

# Poison pill that tells the background worker to exit; enqueued only by close().
# A plain sentinel rather than a flag because the worker blocks in queue.get(),
# so nothing short of an item can wake it.
_SHUTDOWN = object()


@dataclass(frozen=True)
class SessionIngest:
    """What ``add_session`` did with one session, for a caller that has to report it.

    ``already_ingested`` says the session's steps were in the store before this
    call, which is also why ``dispatched`` may be False: a re-read of the same
    session must not pay for the distillation twice.

    ``episode_ids`` are the steps that are in the store under this session —
    written by this call, or found there by the idempotency check. It is empty
    only when nothing was persisted (``persist_steps=False``, or a session with
    no steps)."""

    session_id: str
    host: str
    episode_ids: list[str]
    already_ingested: bool
    dispatched: bool
    # False when an ``admit`` policy refused the session: nothing was
    # persisted or distilled, and ``reason`` says why (the policy's words).
    admitted: bool = True
    reason: str | None = None


def _without_episode_ids(trajectory: list[dict]) -> list[dict]:
    """The step dicts with their ``episode_id`` pointers removed.

    ``SessionTrajectory.as_task_trajectory`` stamps every step with the id it
    WOULD have as a persisted episode. That is only a true pointer once
    ``add_session`` has stored the steps; through ``add_task_result``, or
    ``add_session(persist_steps=False)``, nothing is stored, and an organizer
    that copied the ids into a runbook would be citing episodes that do not
    exist. An organizer cannot tell the two cases apart from the steps alone, so
    the facade — the one party that knows what it persisted — strips the ids."""
    return [
        {k: v for k, v in step.items() if k != "episode_id"} if isinstance(step, dict) else step
        for step in trajectory
    ]


class AgenticMemory:
    """Public facade: one namespace's stores + embedder + organizers, wired by capability.

    Construction resolves each backend slot (doc/vector/graph store, embedder, reranker)
    against detected/declared host capabilities and records any forced degradation in
    ``self._degradations`` (surfaced via ``capabilities()``). All writes go through the
    facade so every mutation is logged to the evolution log before being applied
    (docs/04 §2) — organizers themselves never touch stores directly."""

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        organizers: Sequence[str | Organizer] = ("passthrough",),
        profile: str | None = None,
        config: AgmemConfig | str | Path | None = None,
        embedder: Embedder | None = None,
        caps: HostCapabilities | None = None,
    ) -> None:
        """Resolve backends for ``caps`` (or freshly detected ones) and build the pipeline.

        ``organizers`` may mix registry names and pre-built ``Organizer`` instances.
        Unless ``config.sync_write`` is set, a daemon thread is started to run organizer
        work off the caller's thread — see ``close()``/``flush()`` for the drain contract.

        ``profile`` is a shorthand for ``AgmemConfig(profile=...)`` and is only usable
        when no ``config`` is given. Passing both used to drop ``profile`` silently, so
        ``AgenticMemory(profile="full", config=AgmemConfig())`` resolved every backend
        slot — and ``resolved_embed_model`` — as ``lite`` while ``stats()`` reported
        ``lite`` too, leaving no trace of the requested profile anywhere. There is no
        precedence rule to apply here (the library has one config object, not a merge
        order), so a disagreement is a caller bug and raises. Agreement is allowed so
        the redundant-but-harmless form keeps working."""
        if isinstance(config, (str, Path)):
            config = load_config(config)
        if config is None:
            self.config = AgmemConfig() if profile is None else AgmemConfig(profile=profile)
        else:
            if profile is not None and profile != config.profile:
                raise ValueError(
                    f"profile={profile!r} conflicts with config.profile={config.profile!r}; "
                    "pass only one (the config wins otherwise, silently)"
                )
            self.config = config
        self.namespace = namespace
        self.caps = caps or detect()
        self._degradations: list[str] = []

        # --- stores -------------------------------------------------------
        data_dir = self.config.data_dir
        doc_cls, notes = resolve(
            "doc_store",
            DOC_STORE_CANDIDATES,
            self.caps,
            override=self.config.overrides.get("doc_store"),
            profile_default=self.config.slot_default("doc_store"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        doc_filenames = {"SqliteDocStore": "memory.db", "PostgresDocStore": "pgdata"}
        doc_path = (
            (data_dir / namespace / doc_filenames.get(doc_cls.__name__, "memory.db"))
            if data_dir
            else None
        )
        self.doc_store = doc_cls(doc_path)

        # --- embedder -----------------------------------------------------
        if embedder is not None:
            self.embedder = embedder
        else:
            cls, notes = resolve(
                "embedder",
                EMBEDDER_CANDIDATES,
                self.caps,
                profile_default=self.config.slot_default("embedder"),
                strict=self.config.strict,
            )
            self._degradations.extend(notes)
            if cls.__name__ == "SentenceTransformerEmbedder":
                self.embedder = cls(model_name=self.config.resolved_embed_model)
            else:
                self.embedder = cls()

        # --- vector store -------------------------------------------------
        vec_cls, notes = resolve(
            "vector_store",
            VECTOR_STORE_CANDIDATES,
            self.caps,
            override=self.config.overrides.get("vector_store"),
            profile_default=self.config.slot_default("vector_store"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        # Uniform adapter contract: __init__(path | None, dim). None -> the
        # engine's in-memory/ephemeral mode.
        vec_filenames = {
            "SqliteVecStore": "vectors.db",
            "LanceDBVectorStore": "vectors.lance",
            "QdrantVectorStore": "vectors.qdrant",
            "ChromaVectorStore": "vectors.chroma",
        }
        vec_path = (
            (data_dir / namespace / vec_filenames.get(vec_cls.__name__, "vectors"))
            if data_dir
            else None
        )
        self.vector_store = vec_cls(vec_path, dim=self.embedder.dim)

        # --- llm (optional: built only when llm_roles configured) ----------
        self.budget = BudgetTracker()
        self.llm: LLMClient | None = None
        self.structured: StructuredCaller | None = None
        if self.config.llm_roles:
            self.llm = LLMClient(self.config.llm_roles, budget=self.budget)
            self.structured = StructuredCaller(
                self.llm,
                self.config.use_guided_json,
                reply_retries=self.config.structured_reply_retries,
            )

        # --- organizers -----------------------------------------------------
        # Plural is load-bearing for chaining only: one producer plus N
        # consumers that subscribe to its output types via ``consumes``
        # (_propagate_events). Several organizers reading the raw stream
        # side by side is allowed but has no paper counterpart.
        self.organizers: list[Organizer] = []
        for org in organizers:
            if isinstance(org, str):
                if org not in ORGANIZERS:
                    raise KeyError(f"unknown organizer '{org}' (known: {sorted(ORGANIZERS)})")
                built = ORGANIZERS[org]()
                built.apply_config(self.config)
                self.organizers.append(built)
            else:
                self.organizers.append(org)
        # Two instances of the same organizer would share one consolidate
        # cursor id and clobber each other's progress (base.cursor_key), so
        # give same-named instances a positional suffix. Names that occur once
        # keep the bare key, so cursors persisted by earlier runs still resolve.
        seen_names: dict[str, int] = {}
        for org in self.organizers:
            if sum(x.name == org.name for x in self.organizers) > 1:
                idx = seen_names.get(org.name, 0)
                org._cursor_scope = f"{org.name}#{idx}"
                seen_names[org.name] = idx + 1

        # --- graph store (Zep temporal KG; persistent under data_dir — X4) --
        from agmem.stores import GRAPH_STORE_CANDIDATES

        graph_cls, notes = resolve(
            "graph_store",
            GRAPH_STORE_CANDIDATES,
            self.caps,
            override=self.config.overrides.get("graph_store"),
            profile_default=self.config.slot_default("graph_store"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        graph_filenames = {
            "SqliteGraphStore": "graph.db",
            "KuzuGraphStore": "graph.kuzu",
        }
        graph_path = (
            (data_dir / namespace / graph_filenames.get(graph_cls.__name__, "graph"))
            if data_dir
            else None
        )
        self.graph_store = graph_cls(graph_path)

        self._ctx = OrganizerContext(
            doc_store=self.doc_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            namespace=self.namespace,
            llm=self.structured,
            graph_store=self.graph_store,
        )

        # --- reranker (Noop keeps fusion order; MMR adds diversity) ---------
        reranker_cls, notes = resolve(
            "reranker",
            RERANKER_CANDIDATES,
            self.caps,
            override=self.config.overrides.get("reranker"),
            profile_default=self.config.slot_default("reranker"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        self.reranker = self._build_reranker(reranker_cls)
        self.pipeline = RetrievalPipeline(
            self.doc_store,
            self.vector_store,
            self.embedder,
            reranker=self.reranker,
            graph_store=self.graph_store,
            lexical_types=self.config.lexical_types,
            bfs_types=self.config.bfs_types,
            bfs_max_depth=self.config.bfs_max_depth,
            rrf_k=self.config.rrf_k,
            dense_min_score=self.config.dense_min_score,
            link_expansion_cap=self.config.link_expansion_cap,
            link_expansion_per_hit=self.config.link_expansion_per_hit,
            attach_sources_top_r=self.config.attach_sources_top_r,
            graph_expansion_cap=self.config.graph_expansion_cap,
            graph_expansion_hops=self.config.graph_expansion_hops,
            page_recall_cap=self.config.page_recall_cap,
            page_recall_threshold=self.config.page_recall_threshold,
            page_recall_segment_threshold=self.config.page_recall_segment_threshold,
            page_recall_keyword_similarity=self.config.page_recall_keyword_similarity,
            memmachine_expand_context=self.config.memmachine_expand_context,
            memmachine_context_limit=self.config.memmachine_context_limit,
            # The one organizer-driven entry in the read-step registry: the
            # MemMachine organizer knows which upstream backend wrote the
            # derivatives, and the two backends READ differently
            # (MemMachineContextualize vs MemMachineEventContextualize) —
            # `MEMMACHINE_PRESETS`' "provenance never mixed inside one preset"
            # has to hold on the read side too. No organizer -> declarative,
            # upstream's own "missing discriminator means declarative" rule.
            memmachine_backend=next(
                (
                    str(getattr(org, "backend", "declarative"))
                    for org in self.organizers
                    if org.name == "memmachine"
                ),
                "declarative",
            ),
            task_graph_expansion_cap=self.config.task_graph_expansion_cap,
            task_graph_insight_cap=self.config.task_graph_insight_cap,
        )

        # --- async write worker (docs/03 §3.2) ------------------------------
        # Any, not Callable: close() enqueues the _SHUTDOWN sentinel on this queue.
        self._queue: queue.Queue[Any] | None = None
        self._worker: threading.Thread | None = None
        if not self.config.sync_write:
            self._queue = queue.Queue()
            self._worker = threading.Thread(
                target=self._drain, args=(self._queue,), daemon=True, name="agmem-worker"
            )
            self._worker.start()

    def recent_episode_entity_ids(self, n_episodes: int = 4, limit: int = 20) -> list[str]:
        """Entity nodes mentioned by the ``n_episodes`` most recent episodes —
        BFS seeds for Zep's recency case (paper §3.1: "particularly valuable when
        using recent episodes as seeds … allowing the system to incorporate
        recently mentioned entities and relationships into the retrieved
        context").

        Membership comes from the entity items' own ``source_episode_ids``, which
        is the provenance every organizer records, rather than from a MENTIONS
        edge: upstream walks ``RELATES_TO|MENTIONS`` because episodes are nodes in
        its graph, while raw episodes live in the doc store here. Same relation,
        different storage.

        Returns at most ``limit`` ids, newest episode first, so a caller passing
        it straight to ``search(bfs_origin_ids=...)`` gets a bounded frontier.
        Empty when nothing has been ingested or no entity references those
        episodes."""
        episodes = self.doc_store.list_episodes(namespace=self.namespace)
        if not episodes or n_episodes < 1:
            return []
        recent = [e.id for e in episodes[-n_episodes:]][::-1]
        rank = {episode_id: i for i, episode_id in enumerate(recent)}
        scored: list[tuple[int, str]] = []
        for item in self.doc_store.list_items("entities", namespace=self.namespace):
            best = min(
                (
                    rank[episode_id]
                    for episode_id in item.get("source_episode_ids", [])
                    if episode_id in rank
                ),
                default=None,
            )
            if best is not None and item.get("id"):
                scored.append((best, str(item["id"])))
        scored.sort()
        return [entity_id for _, entity_id in scored[:limit]]

    def _build_reranker(self, reranker_cls: type):
        """Construct the resolved reranker, injecting what only the facade has.

        Three of the six need something at construction time and the sources
        differ, which is why this is not a bare ``reranker_cls()``: the LLM
        reranker needs the structured caller, node-distance needs the graph
        store and namespace (framework handles, never config), and the rest
        take their tuning from ``config.reranker_params`` — the Zep paper's
        BGE-m3 cross-encoder is a ``model_name``, MMR at mmr_lambda=1 is a
        ``lambda_``. Unknown params for the resolved class are dropped with a
        warning rather than raising, because the resolver may have degraded to a
        different class than the config was written for."""
        params = dict(self.config.reranker_params)
        if reranker_cls.__name__ == "LLMReranker":
            return reranker_cls(self.structured)
        if reranker_cls.__name__ == "NodeDistanceReranker":
            params.setdefault("graph_store", self.graph_store)
            params.setdefault("namespace", self.namespace)
        if params:
            accepted = set(inspect.signature(reranker_cls).parameters)
            unknown = sorted(set(params) - accepted)
            if unknown:
                logger.warning(
                    "reranker_params %s not accepted by %s (resolved class); ignored",
                    unknown,
                    reranker_cls.__name__,
                )
            params = {key: value for key, value in params.items() if key in accepted}
        return reranker_cls(**params)

    # ---- write ------------------------------------------------------------

    def add_message(
        self,
        content: str,
        role: str = "user",
        timestamp: Any = None,
        meta: dict | None = None,
    ) -> Episode:
        """Persist and index a raw episode synchronously, then dispatch organizers.

        The episode is written to the doc store, embedded into the vector store, and
        logged (ADD op) before this call returns — it is immediately searchable. Organizer
        ``on_message`` hooks then run either inline (``config.sync_write=True``, exceptions
        propagate to the caller) or on the background worker (exceptions are only logged,
        see ``_drain``)."""
        episode = self._make_episode(content, role, timestamp, meta)
        self._ingest_episode(episode, {"role": role})
        self._dispatch(lambda: self._apply_from_all(lambda org: org.on_message(episode, self._ctx)))
        return episode

    def attach_llm(self, client: Any, use_guided_json: bool | None = None) -> None:
        """Give an already-built memory an LLM, wiring every place that needs it.

        `AgenticMemory` builds `llm`, `structured` and the `OrganizerContext`
        together in `__init__` when `config.llm_roles` is set. A driver that wants
        500 memories to SHARE one client cannot use that path — a client per
        memory means a budget per memory, and a spend cap that binds nothing — so
        it assigns afterwards. Assigning `self.llm` alone leaves `structured`
        None; assigning both leaves `_ctx.llm` holding the None captured at
        construction, and an organizer reads `ctx.llm`.

        That third one is the reason this method exists rather than two
        attributes. Nemori answers a None `ctx.llm` by degrading — boundary
        detection and distillation off, messages bypassing the buffer — and says
        so in a warning. In a 500-row run that warning scrolls past, and the arm
        comes out looking like a memory system that added nothing, when what
        actually happened is that it never ran.
        """
        self.llm = client
        self.structured = StructuredCaller(
            client,
            self.config.use_guided_json if use_guided_json is None else use_guided_json,
            reply_retries=self.config.structured_reply_retries,
        )
        self._ctx.llm = self.structured

    def organizers_have_llm(self) -> bool:
        """Whether an organizer asking `ctx.llm` would find one.

        A caller that pays for a write path can assert on this before spending;
        the alternative is reading a degradation warning out of a log after the
        bill."""
        return self._ctx.llm is not None

    def _make_episode(
        self, content: str, role: str, timestamp: Any = None, meta: dict | None = None
    ) -> Episode:
        """The one place a raw episode is built, so a bulk caller cannot drift from
        ``add_message`` on namespace, timestamp defaulting or meta."""
        return Episode(
            content=content,
            role=role,
            namespace=self.namespace,
            timestamp=timestamp or utcnow(),
            meta=meta or {},
        )

    def bulk_add_messages(
        self, messages: list[tuple[str, str, dict | None]], batch_size: int = 128
    ) -> int:
        """``add_message`` over a whole corpus, with the embedding calls batched.

        Each entry is ``(content, role, meta)``. Episodes are built by the same
        helper ``add_message`` uses, then handed to ``bulk_ingest`` — see there for
        what does and does not stay identical, and for why a batched arm may not be
        compared against a per-turn one."""
        episodes = [self._make_episode(c, r, None, m) for c, r, m in messages]
        return self.bulk_ingest(episodes, batch_size=batch_size)

    def bulk_ingest(self, episodes: list[Episode], batch_size: int = 128) -> int:
        """Index a whole corpus, embedding in batches, then dispatch ``on_message``.

        Same episodes, same ids, same store writes, same op log and same hook order
        as ``add_message`` in a loop. One thing moves: the embedding calls are
        batched, and hooks run after the corpus is indexed rather than interleaved
        with it. An ``on_message`` that only looks at the episode it was handed
        cannot tell; one that QUERIES the stores can, and those organizers declare
        ``observes_store_on_message`` and are routed to the per-message path here.

        This exists because per-turn embedding is what a long benchmark actually
        costs in wall clock: a LongMemEval `_m` instance is 4,894 turns, which is
        4,894 sequential round trips at ~292 ms — 24 minutes of a 26-minute row.
        Batched at 128 it is ~6 seconds, measured at **25x** end to end.

        **The vectors are NOT bit-identical across batch sizes.** Measured on this
        endpoint: cosine 0.999999546 between a text embedded alone and in a batch,
        components differing by up to 1.4e-4. That is enough to reorder near-ties,
        so a run must not mix the two regimes: with k=50 over ~500 candidates the
        retrieved CONTENT still agrees 99.3% of the time and the evidence session
        reached the reader in 5/5 sampled instances both ways, but the order
        differs more often than not. Batched and per-turn arms are comparable only
        to their own kind, which is why the batched arms carry their own tags.
        """
        if not episodes:
            return 0
        observers = [o for o in self.organizers if o.observes_store_on_message]
        if observers:
            # Not a degradation to log and move past: for these organizers the
            # interleaving IS the semantics, so the fast path is simply wrong.
            logger.info(
                "bulk_ingest: %s read the stores in on_message — ingesting per message",
                ", ".join(type(o).__name__ for o in observers),
            )
            for episode in episodes:
                self.add_message(
                    episode.content,
                    role=episode.role,
                    timestamp=episode.timestamp,
                    meta=episode.meta,
                )
            return len(episodes)

        self._ingest_batched(episodes, lambda i, e: {"role": e.role}, batch_size)
        for episode in episodes:
            self._dispatch(
                lambda ep=episode: self._apply_from_all(lambda org: org.on_message(ep, self._ctx))
            )
        return len(episodes)

    def add_task_result(
        self, trajectory: list[dict], outcome: str, task: str, agent_id: str = "agent"
    ) -> None:
        """Record a completed task and dispatch organizer ``on_task_end`` hooks.

        The stored episode's ``meta`` keeps only ``outcome``/``agent_id``/step count — the
        full ``trajectory`` is never persisted by the facade, so ``on_task_end`` is the only
        place methodologies (ReasoningBank/ACE/G-Memory) see step-by-step detail.

        That is deliberate here and stays as it is: the bench harnesses and the MCP
        tool pass trajectories they built themselves, whose steps have no durable
        identity to point at. ``add_session`` is the path that DOES persist the
        trajectory — one episode per step, with stable ids — for callers that have a
        real session log."""
        episode = Episode(
            content=task,
            role="task",
            namespace=self.namespace,
            meta={"outcome": outcome, "agent_id": agent_id, "steps": len(trajectory)},
        )
        self._ingest_episode(episode, {"outcome": outcome})
        trajectory = _without_episode_ids(trajectory)
        self._dispatch(
            lambda: self._apply_from_all(
                lambda org: org.on_task_end(trajectory, outcome, task, self._ctx)
            )
        )

    def add_session(
        self,
        traj: SessionTrajectory,
        *,
        outcome: str = "unknown",
        persist_steps: bool = True,
        distill: bool = True,
        force: bool = False,
        batch_size: int = 128,
        admit: Callable[[SessionTrajectory], str | None] | None = None,
    ) -> SessionIngest:
        """Ingest one coding-agent session: its raw steps into the store, then the
        distillation over them.

        ``admit`` is a session-level admission policy (``sessions.SessionAdmission``
        or anything with its shape): called first, and a non-None reason refuses
        the session outright — no episodes, no model call, ``admitted=False``. It
        is the place for "is this session worth remembering at all", which the
        message-level gates never had (docs/research/agent-memory-axes-v1.md §7.1).

        This is the entry point ``add_task_result`` could not be. A session log has
        a durable identity — host, session id, step position — so every step becomes
        an ``Episode`` with a deterministic id (``SessionTrajectory.episode_id``),
        and what an organizer writes about the session can point back at the exact
        steps it read (docs/research/agent-memory-axes-v1.md §7.1: without this
        there is no raw store for a later just-in-time read, and no caller of
        ``as_task_trajectory`` outside the MCP tool).

        NO ``on_message`` FAN-OUT. Session steps are not conversation turns. A tool
        call and its output are one agent's working notes, not something a user said
        to the system, and the conversational methodologies (A-Mem's note induction,
        Nemori's episode boundaries, MemoryOS's STM) would segment and summarise them
        as if they were — at one or more model calls per step, for a signal the
        session-level distillation already extracts once. The methodologies that
        consume sessions do so through ``on_task_end``; the raw episodes exist to be
        read back by id (the pointer discipline GAM and AgentRunbook-C use), and the
        recency hook and lexical search already see them without any hook running.

        IDEMPOTENCY. A session read twice must not be stored twice or distilled
        twice — the daemon's backfill will re-scan the same files, and the second
        distillation is a second bill. The check is the FIRST AND LAST step ids:
        both in the doc store means the persist loop ran to its end, and without
        ``force`` this then persists nothing, dispatches nothing, and returns with
        ``already_ingested=True``. Only the first present means an earlier run
        died mid-loop (there is no transaction around the batches); that is
        logged and treated as not ingested, so the session is re-persisted and
        distilled rather than sealed at a fraction of its steps. With
        ``force=True`` the steps are re-persisted (``add_episode`` is INSERT OR
        REPLACE, so ids do not multiply) and the distillation runs again — the way
        to pick up a changed prompt or a changed clip policy. In both re-runs the
        derived items an earlier distillation wrote for this session are DELETEd
        first (``_retire_session_items``), or a re-distillation would leave two
        runbooks per session competing in every search.

        ``persist_steps=False`` is outside the idempotency too: with nothing in the
        store there is nothing to recognise the session by, so every pass distils
        it again. It is for a caller that keeps the raw steps elsewhere and
        accepts that; a backfill should not use it.

        What this does not cover: a distillation queued under ``sync_write=False``
        and lost to a process exit before ``flush()``/``close()``. The raw steps
        are then complete, so the next pass reports ``already_ingested`` and the
        session is never distilled — ``force=True`` is the recovery.

        ORDER. Persisting is synchronous and completes before the dispatch, so the
        ``episode_id`` pointers on the step dicts resolve in the store by the time
        an organizer looks at them. Under ``sync_write=False`` the distillation
        itself is queued; ``close()`` and ``flush()`` drain that queue, and a CLI
        that exits without either would lose it.

        ``outcome`` is the caller's label for the whole session and is a hint only —
        the ``experience`` organizer labels each task block it finds itself.
        """
        if admit is not None:
            reason = admit(traj)
            if reason is not None:
                logger.info(
                    "add_session: %s/%s refused by admission: %s", traj.host, traj.id, reason
                )
                return SessionIngest(
                    traj.id, traj.host, [], False, False, admitted=False, reason=reason
                )
        episode_ids = [traj.episode_id(i) for i in range(len(traj.steps))]
        if not traj.steps:
            # Nothing to point at and nothing to distil. Returning early keeps a
            # zero-step session from being re-distilled on every backfill pass: it
            # would never register as already-ingested, having stored no episode.
            return SessionIngest(traj.id, traj.host, [], False, False)

        bounds = {episode_ids[0], episode_ids[-1]}
        present = {e.id for e in self.doc_store.get_episodes(sorted(bounds))}
        already = present == bounds
        if already and not force:
            logger.info(
                "add_session: %s/%s already ingested (%d steps) — skipping persist and distill",
                traj.host,
                traj.id,
                len(traj.steps),
            )
            return SessionIngest(traj.id, traj.host, episode_ids, True, False)
        if present and not already:
            logger.warning(
                "add_session: %s/%s is partially ingested (first step present, last missing)"
                " — an earlier run died mid-loop; re-persisting all %d steps",
                traj.host,
                traj.id,
                len(traj.steps),
            )
        # Ids of the items an earlier distillation of this session left; they
        # are retired only after the new one has produced something (below).
        prior = self._session_item_ids(traj.id) if present and distill else set()

        persisted: list[str] = []
        if persist_steps:
            origin = traj.origin()
            persisted = self._ingest_batched(
                traj.to_episodes(self.namespace),
                lambda i, e: {
                    "role": e.role,
                    "session_id": traj.id,
                    "step_index": i,
                    "source": traj.host,
                    # Origin binding (research §6 #8): the same record the
                    # runbooks carry, so project gating and freshness read one
                    # shape whether the item is raw or derived.
                    "cwd": origin["cwd"],
                    "git_branch": origin["git_branch"],
                    "session_started_at": origin["started_at"],
                    "session_ended_at": origin["ended_at"],
                },
                batch_size,
            )

        if distill:
            steps = traj.as_task_trajectory()
            if not persist_steps:
                steps = _without_episode_ids(steps)
            task_text = traj.task_text

            def redistil() -> None:
                self._apply_from_all(
                    lambda org: org.on_task_end(steps, outcome, task_text, self._ctx)
                )
                if not prior:
                    self._supersede_stale_runbooks(traj.id)
                    return
                # The earlier items go only once the new call has replaced them.
                # A re-distillation whose reply was dropped (the 2026-09-04
                # smoke: two malformed JSON replies) used to leave the session
                # with no runbook at all, the old ones already deleted.
                if self._session_item_ids(traj.id) - prior:
                    self._retire_session_items(traj.id, only=prior)
                    self._supersede_stale_runbooks(traj.id)
                else:
                    logger.warning(
                        "add_session: re-distillation of %s/%s produced nothing — keeping "
                        "the %d earlier item(s)",
                        traj.host,
                        traj.id,
                        len(prior),
                    )

            # One unit of work, so the check runs after the hooks under a
            # background queue as well as in sync mode.
            self._dispatch(redistil)
        return SessionIngest(traj.id, traj.host, persisted, already, distill)

    def _supersede_stale_runbooks(self, session_id: str) -> int:
        """Freshness by deterministic signal (research §6 #7): a runbook from
        ``session_id`` INVALIDATEs an older live runbook about the same task in
        the same project — same ``cwd``, same normalized ``name``, a different
        session whose origin ended earlier (or has no end time). No model
        judges which is current; the session clock does. The older item stays
        in the store with ``superseded_by`` and drops out of serving, as any
        non-bi-temporal INVALIDATE does."""
        live = [
            item
            for item in self.doc_store.list_items("runbooks", namespace=self.namespace)
            if not item.get("invalid_at") and item.get("cwd") and item.get("name")
        ]
        fresh = [i for i in live if i.get("session_id") == session_id]
        ops: list[MemoryOp] = []
        for new in fresh:
            new_end = (new.get("origin") or {}).get("ended_at")
            key = (new["cwd"], " ".join(str(new["name"]).casefold().split()))
            for old in live:
                if old.get("session_id") == session_id:
                    continue
                if (old["cwd"], " ".join(str(old["name"]).casefold().split())) != key:
                    continue
                old_end = (old.get("origin") or {}).get("ended_at")
                if old_end and new_end and old_end > new_end:
                    continue  # the other session is the newer one
                ops.append(
                    MemoryOp(
                        op=OpType.INVALIDATE,
                        target_type="runbooks",
                        target_id=str(old["id"]),
                        actor="ingest",
                        payload={
                            "superseded_by": str(new["id"]),
                            "reason": "newer session, same project and task",
                            **({"t_invalid": new_end} if new_end else {}),
                        },
                    )
                )
        if ops:
            self._apply_ops(ops, actor="ingest")
        return len(ops)

    def _session_item_ids(self, session_id: str) -> set[str]:
        """Ids of the live items the active organizers hold about ``session_id``."""
        ids: set[str] = set()
        produced = dict.fromkeys(t for org in self.organizers for t in org.produces)
        for memory_type in produced:
            for item in self.doc_store.list_items(memory_type, namespace=self.namespace):
                if item.get("session_id") == session_id and item.get("id"):
                    ids.add(str(item["id"]))
        return ids

    def _retire_session_items(self, session_id: str, only: set[str] | None = None) -> int:
        """DELETE every derived item an organizer wrote about ``session_id``, so a
        re-distillation replaces the earlier one instead of sitting next to it.
        With ``only``, just those ids — the ones a re-distillation replaced.

        Generic over the active organizers' ``produces``: whatever type they
        write, an item that carries ``session_id`` in its data came from this
        session. Goes through ``_apply_ops`` so the deletions are in the
        evolution log like any other change — a replay can then tell that the
        first runbook was retired, not lost."""
        ops: list[MemoryOp] = []
        produced = dict.fromkeys(t for org in self.organizers for t in org.produces)
        for memory_type in produced:
            for item in self.doc_store.list_items(memory_type, namespace=self.namespace):
                if item.get("session_id") != session_id or not item.get("id"):
                    continue
                if only is not None and str(item["id"]) not in only:
                    continue
                ops.append(
                    MemoryOp(
                        op=OpType.DELETE,
                        target_type=memory_type,
                        target_id=str(item["id"]),
                        actor="ingest",
                        payload={"reason": "re-distillation", "session_id": session_id},
                    )
                )
        if ops:
            self._apply_ops(ops, actor="ingest")
        return len(ops)

    def add_scaled_task_result(
        self, trajectories: list[list[dict]], task: str, agent_id: str = "agent"
    ) -> None:
        """Record ONE task the agent attempted several times, and dispatch
        ``on_scaled_task_end`` so a methodology can distil from the contrast
        between the attempts (ReasoningBank's MaTTS parallel scaling).

        One episode for the task, not one per trajectory: the attempts are the
        same task, and the facade stores only the task line anyway (see
        ``add_task_result``). ``meta["attempts"]`` records how many there were,
        since that is the only trace of the scaling left in the store.

        No ``outcome`` parameter, because the mechanism has none — the point is a
        MIXTURE of successes and failures, and upstream's own induction never
        reads the per-trajectory labels it computes. Use ``add_task_result`` when
        there is one trajectory and a known outcome."""
        episode = Episode(
            content=task,
            role="task",
            namespace=self.namespace,
            meta={
                "agent_id": agent_id,
                "attempts": len(trajectories),
                "steps": sum(len(t) for t in trajectories),
            },
        )
        self._ingest_episode(episode, {})
        trajectories = [_without_episode_ids(t) for t in trajectories]
        self._dispatch(
            lambda: self._apply_from_all(
                lambda org: org.on_scaled_task_end(trajectories, task, self._ctx)
            )
        )

    def warm_start(self, corpus: list[Episode]) -> None:
        """Bulk-ingest ``corpus`` into the stores, then replay it through each organizer.

        One thing differs from ``add_message``: indexing and organizer work both run
        synchronously on the caller's thread, bypassing the write queue. Intended for
        backfilling history before serving traffic, not for steady-state ingest.

        Backfilled episodes used to get NO ingest ADD op, which made the evolution log
        an incomplete record of what is in the stores — replaying it could not rebuild
        them. That asymmetry was kept "so op counts stay comparable with past runs",
        but no script or bench harness calls ``warm_start`` at all, so there were no
        such runs; it is now logged like any other ingest, marked ``warm_start`` so the
        backfill is still distinguishable from live traffic.

        Drains the write queue first, for the reason ``consolidate()`` does: this
        replay runs on the caller's thread, so without the drain a backfill issued
        while the worker still holds live-traffic work would interleave two threads
        through the same organizer's in-memory state."""
        self._drain_queue()
        for episode in corpus:
            self._ingest_episode(episode, {"role": episode.role, "warm_start": True})
        self._apply_from_all(lambda org: org.warm_start(corpus, self._ctx))

    def _ingest_batched(
        self,
        episodes: list[Episode],
        log_payload: Callable[[int, Episode], dict],
        batch_size: int,
    ) -> list[str]:
        """Embed ``episodes`` in batches and store each one; return their ids in order.

        The one batching loop behind ``bulk_ingest`` and ``add_session``, which
        differ only in the ingest op's payload (``log_payload`` gets the position
        and the episode) and in whether ``on_message`` fans out afterwards. Each
        episode still commits its own doc, vector and log write; the batch is
        the embedder call, which is where the time was."""
        ids: list[str] = []
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            vectors = self.embedder.embed([e.embedding_text() for e in chunk])
            for offset, (episode, vector) in enumerate(zip(chunk, vectors, strict=True)):
                self._ingest_episode(episode, log_payload(start + offset, episode), vector=vector)
                ids.append(episode.id)
        return ids

    def _ingest_episode(
        self, episode: Episode, log_payload: dict, vector: list[float] | None = None
    ) -> None:
        """Store + index one raw episode synchronously, so it is searchable the
        moment the caller returns (write-then-organize, docs/04 §2).

        ``log_payload`` is the ingest ADD op's payload. It used to accept ``None``
        to skip the log entry, for warm-start backfill; that skip was removed when
        warm_start started logging like any other ingest, leaving a branch no
        caller could take."""
        self.doc_store.add_episode(episode)
        self.vector_store.add(
            episode.id,
            self.embedder.embed([episode.embedding_text()])[0] if vector is None else vector,
            memory_type="episodic",
            namespace=self.namespace,
        )
        self.doc_store.append(
            [
                MemoryOp(
                    op=OpType.ADD,
                    target_type="episodic",
                    target_id=episode.id,
                    actor="ingest",
                    payload=log_payload,
                )
            ]
        )

    def _apply_from_all(
        self, hook: Callable[[Organizer], list[MemoryOp]], propagate: bool = True
    ) -> int:
        """Run one hook across every organizer in list order, applying each
        organizer's ops under its own name, and return the total op count.

        The single place organizers are fanned out over — so hook call sites stay
        one line and can never drift on ordering or actor attribution.
        ``propagate=False`` applies the ops without turning them into
        ``MemoryEvent``s; see ``search()`` for the one caller that needs it."""
        applied = 0
        for org in self.organizers:
            ops = hook(org)
            self._apply_ops(ops, actor=org.name, propagate=propagate)
            applied += len(ops)
        return applied

    def _dispatch(self, work: Callable[[], Any]) -> None:
        """Run organizer work sync or hand it to the background worker.

        The raw episode is already stored/indexed synchronously before this
        is called, so reads never wait on organization (docs/03 §3.2)."""
        if self._queue is not None:
            self._queue.put(work)
        else:
            work()

    def _drain_queue(self) -> None:
        """Block until every queued organizer work item has been applied.

        No-op in sync-write mode. Callers that read state the queue may still be
        writing (``flush``, ``consolidate``) or that tear down stores
        (``close``) must go through here first."""
        if self._queue is not None:
            self._queue.join()

    @staticmethod
    def _drain(work_queue: queue.Queue[Any]) -> None:
        """Background worker loop, until ``close()`` enqueues ``_SHUTDOWN``.

        Static, taking the queue as an argument, so the thread never holds a
        reference to the memory: ``Thread(target=self._drain)`` stores the BOUND
        method, which kept a closed ``AgenticMemory`` — with its embedder, stores
        and organizers — reachable for the process lifetime. ``close()`` freed the
        handles, but the object itself was never collected."""
        while True:
            work = work_queue.get()
            if work is _SHUTDOWN:
                work_queue.task_done()
                return
            try:
                work()
            except Exception:
                logger.exception("organizer work failed in background worker")
            finally:
                work_queue.task_done()

    def flush(self) -> None:
        """Block until all queued organizer work is applied, then flush
        any organizer-held buffers (Nemori/MemoryOS tail segments)."""
        self._drain_queue()
        self._apply_from_all(lambda org: org.flush_buffer(self._ctx))
        self.vector_store.persist()

    def consolidate(self) -> int:
        """Deferred management pass (spec §1.4) — explicit trigger only.

        Runs each organizer's consolidate() in list order and applies the
        returned ops through the evolution log. Benchmarks call this at
        deterministic points (end of ingest / between sessions). Returns the
        total ops applied, buffer drain included.

        Two things must land before the cursor scan, for one reason — an
        organizer's consolidate() resumes from the evolution log, so anything
        not yet IN the log is invisible to it:

        - queued organizer work (review I3), since consolidate() runs on the
          caller's thread while the worker may still be applying;
        - buffered units. Nemori's segment buffer and MemoryOS's STM hold the
          tail of the stream until a boundary or capacity trigger fires, and
          those units have produced no ops yet. ``consolidate()`` alone used to
          scan an empty log and report success: three buffered messages yielded
          0 ops and 0 items, where ``flush()`` first yielded 3. Every caller
          already flushed beforehand (``locomo.ingest`` does it internally), so
          this makes the working order the only order rather than changing what
          correct callers observe."""
        self._drain_queue()
        applied = self._apply_from_all(lambda org: org.flush_buffer(self._ctx))
        return applied + self._apply_from_all(lambda org: org.consolidate(self._ctx))

    def _apply_ops(self, ops: list[MemoryOp], actor: str, propagate: bool = True) -> None:
        """Stamp ``actor``, log, apply, then propagate — in that order.

        Attribution is stamped onto copies, so an organizer's returned ops are
        never mutated behind its back (it may still hold references to them, as
        Nemori does while building its within-batch supersession guard)."""
        if not ops:
            return
        ops = [replace(op, actor=actor) for op in ops]
        self.doc_store.append(ops)  # log first — replayable audit trail
        for op in ops:
            self._apply_one(op)
        if propagate:
            self._propagate_events(ops, actor)

    def _propagate_events(self, ops: list[MemoryOp], actor: str) -> None:
        """Applied ADD/UPDATE/MERGE ops become MemoryEvents for subscribed
        organizers (spec §1.2). depth=1: handler ops apply without re-propagation."""
        for op in ops:
            if op.op not in (OpType.ADD, OpType.UPDATE, OpType.MERGE):
                continue
            ev = MemoryEvent(
                source=actor,
                op=op.op,
                target_type=op.target_type,
                target_id=op.target_id,
                payload=dict(op.payload),
                supersedes=tuple(op.payload.get("supersedes", ())) if op.op is OpType.MERGE else (),
            )
            for org in self.organizers:
                if org.name == actor or ev.target_type not in org.consumes:
                    continue
                try:
                    out = org.on_memory_event(ev, self._ctx)
                except Exception:
                    logger.exception("on_memory_event failed (organizer=%s)", org.name)
                    continue
                self._apply_ops(out, actor=org.name, propagate=False)

    def _apply_one(self, op: MemoryOp) -> None:
        if op.op in (OpType.ADD, OpType.UPDATE, OpType.MERGE):
            if op.op is OpType.ADD:
                data = dict(op.payload)
            else:  # UPDATE/MERGE: merge into existing item, don't clobber
                existing = self.doc_store.get_items([op.target_id], op.target_type)
                if not existing and op.op is OpType.UPDATE:
                    # UPDATE used to upsert, so an UPDATE naming an id that is not
                    # there wrote a fragment with no content and no provenance
                    # ({"helpful": 1, "id": ..., "actor": ...}) that retrieval would
                    # then serve. Nothing emits one today — every organizer reads its
                    # target first, and the two hooks taking caller-supplied ids
                    # (ACE/G-Memory on_feedback) guard explicitly — so this makes the
                    # emitter's bug loud instead of storing its consequence. The op
                    # stays in the evolution log either way: append happens first, so
                    # skipping the store write loses no history.
                    logger.warning(
                        "UPDATE on missing item (type=%s id=%s actor=%s); op logged, not applied",
                        op.target_type,
                        op.target_id,
                        op.actor,
                    )
                    return
                # MERGE keeps upserting on purpose: a merge writes its result under a
                # NEW id (Nemori emits MERGE(new) + INVALIDATE(absorbed)), so "target
                # does not exist yet" is the normal case for it, not an error.
                data = dict(existing[0]) if existing else {}
                data.update(op.payload)
            data.setdefault("id", op.target_id)
            # Who wrote this item, persisted from the op's attribution. Two
            # methodologies can share one memory type (Nemori + MemoryOS both
            # write "semantic"; ReasoningBank + G-Memory both write
            # "strategies"), and a store query keyed on the type alone cannot
            # tell them apart — which let Nemori's integrators consider a
            # MemoryOS profile fact a merge candidate. ``setdefault`` after the
            # UPDATE/MERGE merge above keeps the original author when someone
            # else revises the item, and leaves pre-existing items (no ``actor``
            # key) untouched so old stores keep resolving as before.
            data.setdefault("actor", op.actor)
            self.doc_store.put_item(op.target_id, op.target_type, self.namespace, data)
            # An explicitly-null ``embedding_text`` means DOC STORE ONLY — the
            # item never gets a vector row (G-Memory insights: upstream never
            # embeds rules; correlation counting is their only read channel,
            # GMemory.py:490-506; MemoryOS's profile document: upstream serves
            # it solely through the unconditional QA-prompt channel, so a
            # vector row let it double-serve by also winning semantic-k slots —
            # round-12 finding 14). Only an ABSENT key falls back to embedding
            # ``content``.
            text = data["embedding_text"] if "embedding_text" in data else data.get("content")
            if text:
                self.vector_store.add(
                    op.target_id,
                    self.embedder.embed([text])[0],
                    memory_type=op.target_type,
                    namespace=self.namespace,
                )
            self._apply_graph(op.target_type, op.target_id, data)
        elif op.op == OpType.INVALIDATE:
            items = self.doc_store.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                # 최초 무효화 시각 보존 — 이중 무효화 멱등 (spec §1.2)
                data.setdefault("invalid_at", op.payload.get("t_invalid", utcnow().isoformat()))
                if "superseded_by" in op.payload:
                    data["superseded_by"] = op.payload["superseded_by"]
                self.doc_store.put_item(op.target_id, op.target_type, self.namespace, data)
                if op.target_type not in BITEMPORAL_TYPES:
                    # 서빙 제외 보장 — ghost-hit 방지(X1 계열, spec §1.3); doc/로그엔 남음
                    # Bare-id delete: vector rows carry no memory_type key, so an id
                    # shared across two types would drop the OTHER type's vector too.
                    # Deferred deliberately — see `stores.base.VectorStore`.
                    self.vector_store.delete([op.target_id])
                if op.target_type == "facts" and self.graph_store is not None:
                    self.graph_store.invalidate_edge(op.target_id, str(data["invalid_at"]))
        elif op.op in (OpType.LINK, OpType.TAG):
            items = self.doc_store.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                if op.op == OpType.LINK:
                    # Insertion order WITH duplicates — A-Mem upstream is
                    # ``note.links.extend(...)`` in both editions
                    # (memory_layer.py:835, memory_layer_robust.py:505). The
                    # order is load-bearing on the read path: LinkExpansion
                    # consumes links in stored order under a cap, so overflow
                    # selection is first-linked-wins. ``sorted(set(...))`` here
                    # (pre-round-12 finding 3) silently made it
                    # lowest-id-wins instead.
                    data["links"] = list(data.get("links", [])) + list(op.payload.get("links", []))
                else:
                    # TAG has no emitter today (A-Mem tag refinement rides
                    # UPDATE); the set-merge stays as generic framework
                    # semantics until some methodology pins it down.
                    merged = set(data.get("tags", [])) | set(op.payload.get("tags", []))
                    data["tags"] = sorted(merged)
                self.doc_store.put_item(op.target_id, op.target_type, self.namespace, data)
        elif op.op == OpType.DELETE:
            # physical delete is reserved for capacity eviction (MemoryOS
            # heat eviction, G-Memory REMOVE); the log keeps the audit trail.
            # The vector MUST go too — round-5 X1: a surviving vector made
            # deleted items resurface as empty ghost hits. Bare-id delete, same
            # cross-type caveat as the INVALIDATE branch (`stores.base.VectorStore`).
            self.doc_store.put_item(
                op.target_id,
                op.target_type,
                self.namespace,
                {"id": op.target_id, "deleted": True},
            )
            self.vector_store.delete([op.target_id])
            if op.target_type == "communities" and self.graph_store is not None:
                # Communities are derived state, so a DELETE really removes the
                # node and its membership — unlike facts, where INVALIDATE keeps
                # the edge and only marks it. Leaving the graph copy behind would
                # let a stale community keep answering `community_of_node` and so
                # steer the next incremental extension.
                self.graph_store.remove_community(op.target_id)
        elif op.op == OpType.NOOP:
            # Deliberately nothing. A NOOP records that a methodology's write path
            # CONSIDERED an item and decided against changing it — Mem0's `NONE`
            # event (`main.py:326-327` @ v0.1.94), which upstream does not even
            # return to its caller, so "judged, unchanged" and "never judged" are
            # indistinguishable there. Ours are distinguishable because the
            # decision is a log row (`_apply_ops` appends before applying, so the
            # row lands whatever this branch does). Written as an explicit branch
            # rather than left to fall off the end of the chain: an op type nobody
            # handles and an op type handled by doing nothing look identical in the
            # store and opposite in review.
            return

    def _apply_graph(self, target_type: str, target_id: str, data: dict) -> None:
        """Mirror an applied ``entities``/``facts`` item into the graph store.

        The graph used to be written by ``ZepGraphOrganizer`` itself, inline in
        ``on_message`` — the only organizer that touched a store directly, and a
        hole in the guarantee the rest of this class exists to provide. Three
        things followed from it: the evolution log could not rebuild the graph
        (a replayed store left ``GraphRecall`` reading an empty one and silently
        degrading Zep's read path to plain vector RAG), a hook that raised
        between a node write and its returned ops left the graph ahead of the
        doc store, and no audit of "what did this memory do" could see a graph
        edit at all. Applying it here instead makes graph state a pure function
        of the op stream, like every other store (2026-07-27 audit B3).

        Idempotent by construction — every store method it calls is a full-row
        upsert keyed by id (membership included), so replaying a log converges
        rather than duplicating."""
        if self.graph_store is None or target_type not in ("entities", "facts", "communities"):
            return
        if target_type == "communities":
            # The third subgraph (Zep §2.2.4). Membership travels in the payload
            # rather than as separate ops for the same reason fact endpoints do:
            # it is not recoverable at apply time, and the op has to be enough to
            # rebuild the graph on its own.
            self.graph_store.upsert_community(
                target_id,
                self.namespace,
                str(data.get("name", "")),
                str(data.get("summary", "")),
            )
            self.graph_store.set_community_members(
                target_id, self.namespace, [str(m) for m in data.get("member_ids", [])]
            )
            return
        if target_type == "entities":
            self.graph_store.upsert_node(
                target_id,
                self.namespace,
                str(data.get("name", "")),
                str(data.get("summary", "")),
                str(data.get("entity_type", "Entity")),
            )
            return
        src, dst = data.get("subject_id"), data.get("object_id")
        if not src or not dst:
            # A fact with no endpoint ids cannot be an edge. Loud rather than
            # skipped: it means an emitter dropped them from the payload, and
            # the symptom would otherwise be a graph that is quietly incomplete.
            logger.warning(
                "fact %s has no subject_id/object_id; stored as an item but NOT as a graph edge",
                target_id,
            )
            return
        self.graph_store.upsert_edge(
            target_id,
            self.namespace,
            str(src),
            str(dst),
            str(data.get("predicate") or "related_to"),
            str(data.get("content", "")),
            valid_at=data.get("valid_at"),
        )
        if data.get("invalid_at"):
            # upsert_edge is a full-row replace and clears invalid_at/expired_at,
            # so an already-invalidated fact must be re-stamped after it.
            self.graph_store.invalidate_edge(target_id, str(data["invalid_at"]))

    # ---- read ---------------------------------------------------------------

    @property
    def default_memory_types(self) -> tuple[str, ...]:
        """What ``search()`` reads when the caller names no types.

        ``episodic`` always leads: raw episodes are written by the facade itself
        (``_ingest_episode``), so no organizer declares them, yet they are always
        present. Then each active organizer's ``produces``, in organizer order,
        deduped — so ``--organizers amem`` searches notes without the caller
        having to know that.

        ``playbook`` is excluded even when ACE is active: ACE's read contract is
        whole-playbook injection via ``get_playbook()`` — a top-k retrieval of
        bullets is exactly the partial view its organizer forbids (round-12 #5:
        with playbook in the default set, a plain ``search()`` served bullets
        through the generic pipeline, holding the "only injection route" claim
        by convention rather than structure). An EXPLICIT
        ``memory_types=("playbook",)`` from a caller still works — the exclusion
        binds only the default."""
        types = ["episodic"]
        for org in self.organizers:
            for memory_type in org.produces:
                if memory_type == "playbook":
                    continue  # whole-playbook injection only by default; see docstring
                if memory_type not in types:
                    types.append(memory_type)
        return tuple(types)

    def search(
        self,
        query: str,
        memory_types: Sequence[str] | None = None,
        k: int | dict[str, int] = 10,
        center_node_id: str | None = None,
        bfs_origin_ids: list[str] | None = None,
        query_keywords: set[str] | frozenset[str] | None = None,
        metrics: dict[str, Any] | None = None,
        project: str | None = None,
    ) -> MemoryBundle:
        """Retrieve across ``memory_types`` via the fused/reranked pipeline, then feed
        read->write hooks.

        ``project`` gates the bundle by origin (research §6 #9, cross-project
        leakage at read time): an item written from a session whose ``cwd`` is
        neither this path nor inside/outside it on the same tree is dropped
        before it is served, and ``metrics["project_gated"]`` counts them. Items
        with no cwd (bench-built, or written before origin binding) pass — the
        gate refuses what it knows is foreign, not what it cannot place.

        ``metrics``, when a dict is passed, records how this bundle was obtained
        — here always one plain search. It exists so that this signature and
        ``retrieval/planned.py::PlannedSearch.search`` are interchangeable: a
        caller that may or may not have a read policy attached needs no branch,
        and gets the same accounting either way.

        ``memory_types=None`` falls back to ``default_memory_types`` (the active
        organizers' declared output); passing types explicitly overrides that, which
        is how the paper-faithful configs stay methodology-pure and how the
        deliberately-mixed ablations keep their raw episodic channel.

        Ranking/fusion policy lives in ``RetrievalPipeline`` (lexical+vector fusion,
        reranker); this method's own contract is the read->write loop: every served
        ``(item_id, memory_type, score)`` triple is passed to each organizer's
        ``on_retrieval`` synchronously, before this call returns, so their returned ops
        are applied for the *next* search — never the one in progress.

        Those ops are applied WITHOUT propagation. ``base.on_retrieval`` requires
        implementations to be cheap ("no LLM calls here") because this runs inline on
        the read path, but that contract only binds the hook itself: propagated ops
        become ``MemoryEvent``s, and a subscriber's ``on_memory_event`` may do anything
        — ``ChainedConsumer`` feeds the wrapped organizer's ``on_message``, which calls
        an LLM. Nothing hits that today only because both ``on_retrieval``
        implementations return ``[]``. Cutting propagation here makes the read path's
        cost bound structural instead of a property two organizers happen to have. The
        cost is that a chained consumer cannot observe read-path mutations; when some
        methodology needs that, it should be an explicit decision rather than something
        inherited from the write path's fan-out.

        ``center_node_id`` is Zep's ``center_node_uuid``: the centroid the
        node-distance reranker measures graph distance from (paper §3.2). Only
        that reranker reads it, and it is a per-query choice, so it travels as an
        argument rather than as config. ``bfs_origin_ids`` is its
        ``bfs_origin_node_uuids``, the explicit seed set for the graph BFS
        channel — see ``recent_episode_entity_ids`` for the recency seeding the
        paper motivates it with.

        ``query_keywords`` is MemoryOS's eval-lineage keyword term: that harness
        runs an LLM keyword extraction per query and adds the overlap to the
        segment score, where the pypi library disabled the same term. The caller
        supplies them because the extraction is an LLM call and this layer makes
        none (``MemoryOSPageRecall``)."""
        started = perf_counter()
        if metrics is not None:
            metrics.update({"agent": "search", "memory_search_called": 1, "queries": [query]})
        types = tuple(memory_types) if memory_types is not None else self.default_memory_types
        bundle = self.pipeline.search(
            query,
            k=k,
            memory_types=types,
            namespace=self.namespace,
            center_node_id=center_node_id,
            bfs_origin_ids=bfs_origin_ids,
            query_keywords=query_keywords,
        )
        if project is not None:
            kept = [
                scored for scored in bundle.items if same_project(item_cwd(scored.item), project)
            ]
            if metrics is not None:
                metrics["project_gated"] = len(bundle.items) - len(kept)
            bundle.items = kept
        # read->write feedback (round-5): organizers see what was served.
        hits = [
            (
                getattr(s.item, "id", None) or s.item.data.get("id"),
                s.memory_type,
                s.score,
            )
            for s in bundle.items
        ]
        self._apply_from_all(
            lambda org: self._warn_if_subscribed(org.on_retrieval(hits, self._ctx), org),
            propagate=False,
        )
        if metrics is not None:
            # Wall clock of the whole read, feedback hooks included: the number
            # LongMemEval-V2's LAFS scores accuracy against, and the one the
            # explorer path (``research``) is compared on.
            metrics["latency_s"] = perf_counter() - started
        return bundle

    def research(
        self,
        query: str,
        *,
        root: Path | str | None = None,
        refresh: bool = True,
        **explorer_kwargs: Any,
    ) -> Any:
        """The explorer read path: export the store as files, then let a model
        grep and read them (``agmem.explore``), returning cited context with its
        latency.

        This is the other read path, not a mode of ``search``. It is the arm
        docs/research/agent-memory-axes-v1.md §6 calls the honest baseline —
        raw session logs plus a grep-capable agent — and it exists so the two
        can be measured against each other; hence no fallback in either
        direction. A memory without an ``explore`` LLM role raises rather than
        searching, because a caller who asked for exploration and got a vector
        bundle would not be able to tell.

        ``root`` defaults to ``<data_dir>/<namespace>/workspace``; an in-memory
        store has no such place and must be given one. ``refresh`` re-exports
        first, which is a scan and not a rewrite when nothing changed."""
        from agmem.explore import Explorer, export_workspace

        role = explorer_kwargs.get("role", "explore")
        llm = self.structured
        usable = llm is not None and (not hasattr(llm, "client") or llm.client.has_role(role))
        if not usable:
            raise RuntimeError(
                f"research needs an LLM role {role!r} — add an [llm.{role}] section to the "
                "config (see agmem.example.toml). It does not fall back to search()."
            )
        if root is None:
            if self.config.data_dir is None:
                raise ValueError(
                    "research needs a workspace root: this memory has no data_dir to put one under"
                )
            root = Path(self.config.data_dir) / self.namespace / "workspace"
        export_s = 0.0
        if refresh:
            started = perf_counter()
            export_workspace(self, root)
            export_s = perf_counter() - started
        result = Explorer(root, **explorer_kwargs).research(query, llm)
        # The export is part of what this read costs per query, exactly as the
        # vector arm's latency includes its hooks; a number that left it out
        # would compare the two arms at different boundaries.
        result.export_s = export_s
        result.latency_s += export_s
        return result

    def _warn_if_subscribed(self, ops: list[MemoryOp], source: Organizer) -> list[MemoryOp]:
        """Pass ``ops`` through, warning if any would have reached a subscriber.

        Cutting propagation on the read path is silent by nature, and a silent
        behavioural gap is the thing this codebase keeps getting caught by. If a
        methodology ever emits content-bearing ops from ``on_retrieval`` and
        expects a chained consumer to see them, this says so instead of leaving
        the author to infer it from an empty downstream."""
        for op in ops:
            for org in self.organizers:
                if org.name != source.name and op.target_type in org.consumes:
                    logger.warning(
                        "on_retrieval op (type=%s from=%s) is not propagated, so %s will not "
                        "see it — read-path ops are applied but never fanned out (docs/04 §3.5)",
                        op.target_type,
                        source.name,
                        org.name,
                    )
                    return ops
        return ops

    def report_feedback(self, memory_ids: Sequence[str], helpful: bool) -> int:
        """Close the loop: usage outcome adjusts memory quality signals.

        Dispatches to each active organizer's ``on_feedback`` and returns the
        total op count. The rules themselves are methodology-owned — ACE counts
        helpful/harmful per bullet, G-Memory reward-shapes served insights and
        prunes at score <= 0 — so an organizer that declares no feedback
        semantics contributes nothing.

        This used to branch on ``target_type`` here instead, which conflated the
        two methodologies that share the ``strategies`` type: a ReasoningBank
        item (append-only by design, no feedback loop in the paper) picked up
        G-Memory's +1/-2, and G-Memory's own served-insight gate was bypassed.
        The consequence of the fan-out is that feedback now requires the owning
        organizer to be active — feeding back a playbook bullet with no ACE
        organizer configured is a no-op returning 0, where it previously
        updated the counter."""
        ids = list(memory_ids)
        return self._apply_from_all(lambda org: org.on_feedback(ids, helpful, self._ctx))

    def get_playbook(self, section: str | None = None) -> str:
        """Render the ACE playbook (ALL bullets, grouped by section).

        This full render IS the methodology's read contract: ACE injects
        the whole playbook and lets the Generator LLM pick bullets
        (round-5 ACE §2) — do not swap it for top-k retrieval.

        Empty when no active organizer produces ``playbook``. Without that gate
        the two halves of the playbook loop disagreed: reading was type-owned
        (this method queries the store directly) while writing is producer-owned
        (``report_feedback`` fans out to ``on_feedback``, docs/04 §3.4). With ACE
        deconfigured but its bullets still in the store, one MCP session would
        render a bullet and then silently return 0 from the feedback call meant
        to update it. Same rule as ``default_memory_types``: what the active
        methodology actually produced."""
        if not any("playbook" in org.produces for org in self.organizers):
            return ""
        bullets = self.doc_store.list_items("playbook", namespace=self.namespace)
        if section:
            bullets = [b for b in bullets if b.get("section") == section]
        by_section: dict[str, list[str]] = {}
        for b in bullets:
            bullet_section = b.get("section", "general")
            by_section.setdefault(bullet_section, []).append(
                render_bullet_line(
                    b.get("content", ""),
                    bullet_section,
                    b["id"],
                    b.get("helpful", 0),
                    b.get("harmful", 0),
                )
            )
        return "\n".join(f"## {s}\n" + "\n".join(lines) for s, lines in sorted(by_section.items()))

    # ---- introspection --------------------------------------------------------

    @property
    def log(self):
        """Expose the doc store as an EvolutionLog (``tail()``/``count()``) for admin/CLI use."""
        return self.doc_store  # EvolutionLog protocol: tail()/count()

    def stats(self) -> dict[str, Any]:
        """Snapshot of counts/costs at call time — not cached, each call re-queries stores."""
        return {
            "namespace": self.namespace,
            "profile": self.config.profile,
            "episodes": self.doc_store.count_episodes(self.namespace),
            "vectors": self.vector_store.count(),
            "evolution_ops": self.doc_store.count(),
            "llm": self.budget.summary(),
            "structured_drops": dict(self.structured.drops) if self.structured else {},
            "embedder": self.embedder.name,
            "vector_store": type(self.vector_store).__name__,
        }

    def capabilities(self) -> dict[str, Any]:
        """Detected host caps, currently-active adapter classes, and forced degradations.

        ``degradations`` is the list accumulated in ``__init__`` when capability resolution
        couldn't satisfy a config override/profile default — the only place that history
        is exposed after construction."""
        return {
            "detected": {
                "ram_gb": self.caps.ram_gb,
                "vram_gb": self.caps.vram_gb,
                "gpu": self.caps.gpu_name,
                "cpu_cores": self.caps.cpu_cores,
                "services": {k: v for k, v in self.caps.services.items() if v},
                "llm_endpoints": [e.base_url for e in self.caps.llm_endpoints if e.alive],
            },
            "active": {
                "embedder": self.embedder.name,
                "vector_store": type(self.vector_store).__name__,
                "organizers": [o.name for o in self.organizers],
            },
            "degradations": self._degradations,
        }

    def close(self) -> None:
        """Drain pending async organizer work, stop the worker, then close stores in
        order (vector, graph, doc).

        The queue must join before the worker is stopped and the stores closed —
        closing a store while queued work still references it would touch a closed
        handle. The worker used to be left running (``_drain`` loops forever, and the
        thread is a daemon so the process still exits): harmless for the MCP server's
        one process-wide memory, but every benchmark that builds a memory per config
        or per question would accumulate one live thread — and one unreachable-but-
        retained memory — per instance. That is latent only while ``sync_write``
        defaults to True.

        Idempotent: a second call finds no worker and re-closes stores, which every
        store adapter tolerates."""
        self._drain_queue()
        if self._worker is not None and self._queue is not None:
            self._queue.put(_SHUTDOWN)
            self._worker.join(timeout=5)
            self._worker = None
            # Dropping the queue puts _dispatch back on its inline path, so work
            # submitted after close() fails loudly on the closed store instead of
            # queueing onto a worker that is gone (flush() would hang forever).
            self._queue = None
        self.vector_store.close()
        self.graph_store.close()
        self.doc_store.close()

    def __enter__(self) -> Self:
        """``with AgenticMemory(...) as mem:`` — close() is mandatory (store handles
        plus the write worker), so the guarded form exists rather than leaving every
        call site to hand-roll try/finally as the bench scripts do."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
