"""AgenticMemory facade — the public API (docs/05 §1).

Write path (docs/04 §2): raw episode is stored and indexed synchronously
(immediately searchable), then organizers run and their MemoryOps are
logged append-only before being applied to stores.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Self, Sequence

from agmem.capabilities import detect, resolve
from agmem.capabilities.detect import HostCapabilities
from agmem.config import AgmemConfig, load_config
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, MemoryBundle, utcnow
from agmem.embed import EMBEDDER_CANDIDATES
from agmem.embed.base import Embedder
from agmem.llm import BudgetTracker, LLMClient, StructuredCaller
from agmem.organizers import ORGANIZERS, MemoryEvent, Organizer, OrganizerContext
from agmem.retrieval import RetrievalPipeline
from agmem.retrieval.rerank import RERANKER_CANDIDATES
from agmem.stores import DOC_STORE_CANDIDATES, VECTOR_STORE_CANDIDATES

logger = logging.getLogger("agmem")

BITEMPORAL_TYPES = ("facts",)  # Types kept in vector store even after INVALIDATE (Zep bi-temporal)

# Poison pill that tells the background worker to exit; enqueued only by close().
# A plain sentinel rather than a flag because the worker blocks in queue.get(),
# so nothing short of an item can wake it.
_SHUTDOWN = object()


class AgenticMemory:
    """Public facade: one namespace's stores + embedder + organizers, wired by capability.

    Construction resolves each backend slot (doc/vector/graph store, embedder, reranker)
    against detected/declared host capabilities and records any forced degradation in
    ``self._degradations`` (surfaced via ``capabilities()``). All writes go through the
    facade so every mutation is logged to the evolution log before being applied
    (docs/04 §2) — organizers themselves never touch stores directly."""

    def __init__(
        self,
        namespace: str = "main",
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
            self.structured = StructuredCaller(self.llm, self.config.use_guided_json)

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
                self.organizers.append(ORGANIZERS[org]())
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
        if reranker_cls.__name__ == "LLMReranker":
            self.reranker = reranker_cls(self.structured)
        else:
            self.reranker = reranker_cls()
        self.pipeline = RetrievalPipeline(
            self.doc_store,
            self.vector_store,
            self.embedder,
            reranker=self.reranker,
            graph_store=self.graph_store,
            lexical_types=self.config.lexical_types,
            link_expansion_cap=self.config.link_expansion_cap,
            attach_sources_top_r=self.config.attach_sources_top_r,
            graph_expansion_cap=self.config.graph_expansion_cap,
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
        episode = Episode(
            content=content,
            role=role,
            namespace=self.namespace,
            timestamp=timestamp or utcnow(),
            meta=meta or {},
        )
        self._ingest_episode(episode, {"role": role})
        self._dispatch(lambda: self._apply_from_all(lambda org: org.on_message(episode, self._ctx)))
        return episode

    def add_task_result(
        self, trajectory: list[dict], outcome: str, task: str, agent_id: str = "agent"
    ) -> None:
        """Record a completed task and dispatch organizer ``on_task_end`` hooks.

        The stored episode's ``meta`` keeps only ``outcome``/``agent_id``/step count — the
        full ``trajectory`` is never persisted by the facade, so ``on_task_end`` is the only
        place methodologies (ReasoningBank/ACE/G-Memory) see step-by-step detail."""
        episode = Episode(
            content=task,
            role="task",
            namespace=self.namespace,
            meta={"outcome": outcome, "agent_id": agent_id, "steps": len(trajectory)},
        )
        self._ingest_episode(episode, {"outcome": outcome})
        self._dispatch(
            lambda: self._apply_from_all(
                lambda org: org.on_task_end(trajectory, outcome, task, self._ctx)
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
        backfill is still distinguishable from live traffic."""
        for episode in corpus:
            self._ingest_episode(episode, {"role": episode.role, "warm_start": True})
        self._apply_from_all(lambda org: org.warm_start(corpus, self._ctx))

    def _ingest_episode(self, episode: Episode, log_payload: dict | None) -> None:
        """Store + index one raw episode synchronously, so it is searchable the
        moment the caller returns (write-then-organize, docs/04 §2).

        ``log_payload`` is the ingest ADD op's payload; ``None`` skips the log
        entry entirely (warm-start backfill)."""
        self.doc_store.add_episode(episode)
        self.vector_store.add(
            episode.id,
            self.embedder.embed([episode.embedding_text()])[0],
            memory_type="episodic",
            namespace=self.namespace,
        )
        if log_payload is not None:
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

    def _apply_from_all(self, hook: Callable[[Organizer], list[MemoryOp]]) -> int:
        """Run one hook across every organizer in list order, applying each
        organizer's ops under its own name, and return the total op count.

        The single place organizers are fanned out over — so hook call sites stay
        one line and can never drift on ordering or actor attribution."""
        applied = 0
        for org in self.organizers:
            ops = hook(org)
            self._apply_ops(ops, actor=org.name)
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
        deterministic points (end of ingest / between sessions).

        Drains the async write queue first (review I3): consolidate() runs on
        the caller's thread and reads the log via ops_since, so any organizer
        work still queued must land before the cursor scan to avoid missing
        just-appended-but-not-yet-applied facts."""
        self._drain_queue()
        return self._apply_from_all(lambda org: org.consolidate(self._ctx))

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
            text = data.get("embedding_text") or data.get("content")
            if text:
                self.vector_store.add(
                    op.target_id,
                    self.embedder.embed([text])[0],
                    memory_type=op.target_type,
                    namespace=self.namespace,
                )
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
                    self.vector_store.delete([op.target_id])
        elif op.op in (OpType.LINK, OpType.TAG):
            items = self.doc_store.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                key = "links" if op.op == OpType.LINK else "tags"
                merged = set(data.get(key, [])) | set(op.payload.get(key, []))
                data[key] = sorted(merged)
                self.doc_store.put_item(op.target_id, op.target_type, self.namespace, data)
        elif op.op == OpType.DELETE:
            # physical delete is reserved for capacity eviction (MemoryOS
            # heat eviction, G-Memory REMOVE); the log keeps the audit trail.
            # The vector MUST go too — round-5 X1: a surviving vector made
            # deleted items resurface as empty ghost hits.
            self.doc_store.put_item(
                op.target_id,
                op.target_type,
                self.namespace,
                {"id": op.target_id, "deleted": True},
            )
            self.vector_store.delete([op.target_id])

    # ---- read ---------------------------------------------------------------

    @property
    def default_memory_types(self) -> tuple[str, ...]:
        """What ``search()`` reads when the caller names no types.

        ``episodic`` always leads: raw episodes are written by the facade itself
        (``_ingest_episode``), so no organizer declares them, yet they are always
        present. Then each active organizer's ``produces``, in organizer order,
        deduped — so ``--organizers amem`` searches notes without the caller
        having to know that."""
        types = ["episodic"]
        for org in self.organizers:
            for memory_type in org.produces:
                if memory_type not in types:
                    types.append(memory_type)
        return tuple(types)

    def search(
        self,
        query: str,
        memory_types: Sequence[str] | None = None,
        k: int | dict[str, int] = 10,
    ) -> MemoryBundle:
        """Retrieve across ``memory_types`` via the fused/reranked pipeline, then feed
        read->write hooks.

        ``memory_types=None`` falls back to ``default_memory_types`` (the active
        organizers' declared output); passing types explicitly overrides that, which
        is how the paper-faithful configs stay methodology-pure and how the
        deliberately-mixed ablations keep their raw episodic channel.

        Ranking/fusion policy lives in ``RetrievalPipeline`` (lexical+vector fusion,
        reranker); this method's own contract is the read->write loop: every served
        ``(item_id, memory_type, score)`` triple is passed to each organizer's
        ``on_retrieval`` synchronously, before this call returns, so their returned ops
        are applied for the *next* search — never the one in progress."""
        types = tuple(memory_types) if memory_types is not None else self.default_memory_types
        bundle = self.pipeline.search(query, k=k, memory_types=types, namespace=self.namespace)
        # read->write feedback (round-5): organizers see what was served.
        hits = [
            (
                getattr(s.item, "id", None) or s.item.data.get("id"),
                s.memory_type,
                s.score,
            )
            for s in bundle.items
        ]
        self._apply_from_all(lambda org: org.on_retrieval(hits, self._ctx))
        return bundle

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
        (round-5 ACE §2) — do not swap it for top-k retrieval."""
        bullets = self.doc_store.list_items("playbook", namespace=self.namespace)
        if section:
            bullets = [b for b in bullets if b.get("section") == section]
        by_section: dict[str, list[str]] = {}
        for b in bullets:
            by_section.setdefault(b.get("section", "general"), []).append(
                f"[{b.get('section', 'general')}-{b['id'][:5]}] "
                f"helpful={b.get('helpful', 0)} harmful={b.get('harmful', 0)} "
                f":: {b.get('content', '')}"
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
