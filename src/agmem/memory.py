"""AgenticMemory facade — the public API (docs/05 §1).

Write path (docs/04 §2): raw episode is stored and indexed synchronously
(immediately searchable), then organizers run and their MemoryOps are
logged append-only before being applied to stores.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

from agmem.capabilities import detect, resolve
from agmem.capabilities.detect import HostCapabilities
from agmem.config import AgmemConfig, load_config
from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, MemoryBundle, utcnow
from agmem.embed import EMBEDDER_CANDIDATES
from agmem.embed.base import Embedder
from agmem.llm import BudgetTracker, LLMClient, StructuredCaller
from agmem.organizers import ORGANIZERS, Organizer, OrganizerContext
from agmem.retrieval import RetrievalPipeline
from agmem.retrieval.rerank import RERANKER_CANDIDATES
from agmem.stores import VECTOR_STORE_CANDIDATES, SqliteDocStore

logger = logging.getLogger("agmem")


class AgenticMemory:
    def __init__(
        self,
        namespace: str = "main",
        organizers: Sequence[str | Organizer] = ("passthrough",),
        profile: str = "lite",
        config: AgmemConfig | str | Path | None = None,
        embedder: Embedder | None = None,
        caps: HostCapabilities | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config or AgmemConfig(profile=profile)
        self.namespace = namespace
        self.caps = caps or detect()
        self._degradations: list[str] = []

        # --- stores -------------------------------------------------------
        data_dir = self.config.data_dir
        doc_path = (data_dir / namespace / "memory.db") if data_dir else ":memory:"
        self.doc = SqliteDocStore(doc_path)

        # --- embedder -----------------------------------------------------
        if embedder is not None:
            self.embedder = embedder
        else:
            cls, notes = resolve(
                "embedder", EMBEDDER_CANDIDATES, self.caps,
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
            "vector_store", VECTOR_STORE_CANDIDATES, self.caps,
            override=self.config.overrides.get("vector_store"),
            profile_default=self.config.slot_default("vector_store"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        # Uniform adapter contract: __init__(path | None, dim). None -> the
        # engine's in-memory/ephemeral mode.
        vec_filenames = {"SqliteVecStore": "vectors.db",
                         "LanceDBVectorStore": "vectors.lance",
                         "QdrantVectorStore": "vectors.qdrant",
                         "ChromaVectorStore": "vectors.chroma"}
        vec_path = (
            data_dir / namespace / vec_filenames.get(vec_cls.__name__, "vectors")
        ) if data_dir else None
        self.vec = vec_cls(vec_path, dim=self.embedder.dim)

        # --- llm (optional in Phase 0: passthrough needs none) -------------
        self.budget = BudgetTracker()
        self.llm: LLMClient | None = None
        self.structured: StructuredCaller | None = None
        if self.config.llm_roles:
            self.llm = LLMClient(self.config.llm_roles, budget=self.budget)
            self.structured = StructuredCaller(self.llm, self.config.use_guided_json)

        # --- organizers -----------------------------------------------------
        self.organizers: list[Organizer] = []
        for org in organizers:
            if isinstance(org, str):
                if org not in ORGANIZERS:
                    raise KeyError(f"unknown organizer '{org}' (known: {sorted(ORGANIZERS)})")
                self.organizers.append(ORGANIZERS[org]())
            else:
                self.organizers.append(org)

        self._ctx = OrganizerContext(
            doc=self.doc, vec=self.vec, embedder=self.embedder,
            namespace=self.namespace, llm=self.structured,
        )

        # --- reranker (Noop keeps fusion order; MMR adds diversity) ---------
        reranker_cls, notes = resolve(
            "reranker", RERANKER_CANDIDATES, self.caps,
            override=self.config.overrides.get("reranker"),
            profile_default=self.config.slot_default("reranker"),
            strict=self.config.strict,
        )
        self._degradations.extend(notes)
        if reranker_cls.__name__ == "LLMReranker":
            self.reranker = reranker_cls(self.structured)
        else:
            self.reranker = reranker_cls()
        self.pipeline = RetrievalPipeline(self.doc, self.vec, self.embedder,
                                          reranker=self.reranker)

        # --- async write worker (docs/03 §3.2) ------------------------------
        self._queue: queue.Queue[Callable[[], None]] | None = None
        self._worker: threading.Thread | None = None
        if not self.config.sync_write:
            self._queue = queue.Queue()
            self._worker = threading.Thread(target=self._drain, daemon=True,
                                            name="agmem-worker")
            self._worker.start()

    # ---- write ------------------------------------------------------------

    def add_message(self, content: str, role: str = "user",
                    timestamp: Any = None, meta: dict | None = None) -> Episode:
        ep = Episode(
            content=content, role=role, namespace=self.namespace,
            timestamp=timestamp or utcnow(), meta=meta or {},
        )
        # sync: raw episode is immediately searchable
        self.doc.add_episode(ep)
        self.vec.add(ep.id, self.embedder.embed([ep.embedding_text()])[0],
                     memory_type="episodic", namespace=self.namespace)
        self.doc.append([MemoryOp(op=OpType.ADD, target_type="episodic",
                                  target_id=ep.id, actor="ingest",
                                  payload={"role": role})])
        self._dispatch(lambda: [
            self._apply_ops(org.on_message(ep, self._ctx), actor=org.name)
            for org in self.organizers
        ])
        return ep

    def add_task_result(self, trajectory: list[dict], outcome: str,
                        task: str, agent_id: str = "agent") -> None:
        ep = Episode(
            content=task, role="task", namespace=self.namespace,
            meta={"outcome": outcome, "agent_id": agent_id, "steps": len(trajectory)},
        )
        self.doc.add_episode(ep)
        self.vec.add(ep.id, self.embedder.embed([ep.embedding_text()])[0],
                     memory_type="episodic", namespace=self.namespace)
        self.doc.append([MemoryOp(op=OpType.ADD, target_type="episodic",
                                  target_id=ep.id, actor="ingest",
                                  payload={"outcome": outcome})])
        self._dispatch(lambda: [
            self._apply_ops(org.on_task_end(trajectory, outcome, task, self._ctx),
                            actor=org.name)
            for org in self.organizers
        ])

    def warm_start(self, corpus: list[Episode]) -> None:
        for ep in corpus:
            self.doc.add_episode(ep)
            self.vec.add(ep.id, self.embedder.embed([ep.embedding_text()])[0],
                         memory_type="episodic", namespace=self.namespace)
        for org in self.organizers:
            self._apply_ops(org.warm_start(corpus, self._ctx), actor=org.name)

    def _dispatch(self, work: Callable[[], Any]) -> None:
        """Run organizer work sync or hand it to the background worker.

        The raw episode is already stored/indexed synchronously before this
        is called, so reads never wait on organization (docs/03 §3.2)."""
        if self._queue is not None:
            self._queue.put(work)
        else:
            work()

    def _drain(self) -> None:
        assert self._queue is not None
        while True:
            work = self._queue.get()
            try:
                work()
            except Exception:
                logger.exception("organizer work failed in background worker")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until all queued organizer work is applied, then flush
        any organizer-held buffers (Nemori/MemoryOS tail segments)."""
        if self._queue is not None:
            self._queue.join()
        for org in self.organizers:
            flush_buffer = getattr(org, "flush_buffer", None)
            if callable(flush_buffer):
                self._apply_ops(flush_buffer(self._ctx), actor=org.name)
        self.vec.persist()

    def _apply_ops(self, ops: list[MemoryOp], actor: str) -> None:
        if not ops:
            return
        for op in ops:
            op.actor = actor
        self.doc.append(ops)  # log first — replayable audit trail
        for op in ops:
            self._apply_one(op)

    def _apply_one(self, op: MemoryOp) -> None:
        if op.op in (OpType.ADD, OpType.UPDATE, OpType.MERGE):
            if op.op is OpType.ADD:
                data = dict(op.payload)
            else:  # UPDATE/MERGE: merge into existing item, don't clobber
                existing = self.doc.get_items([op.target_id], op.target_type)
                data = dict(existing[0]) if existing else {}
                data.update(op.payload)
            data.setdefault("id", op.target_id)
            self.doc.put_item(op.target_id, op.target_type, self.namespace, data)
            text = data.get("embedding_text") or data.get("content")
            if text:
                self.vec.add(op.target_id, self.embedder.embed([text])[0],
                             memory_type=op.target_type, namespace=self.namespace)
        elif op.op == OpType.INVALIDATE:
            items = self.doc.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                data["invalid_at"] = op.payload.get("t_invalid", utcnow().isoformat())
                self.doc.put_item(op.target_id, op.target_type, self.namespace, data)
        elif op.op in (OpType.LINK, OpType.TAG):
            items = self.doc.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                key = "links" if op.op == OpType.LINK else "tags"
                merged = set(data.get(key, [])) | set(op.payload.get(key, []))
                data[key] = sorted(merged)
                self.doc.put_item(op.target_id, op.target_type, self.namespace, data)
        elif op.op == OpType.DELETE:
            # physical delete is reserved for capacity eviction (MemoryOS LFU);
            # the log keeps the audit trail either way
            self.doc.put_item(op.target_id, op.target_type, self.namespace,
                              {"id": op.target_id, "deleted": True})

    # ---- read ---------------------------------------------------------------

    def search(self, query: str, memory_types: Sequence[str] = ("episodic",),
               k: int | dict[str, int] = 10) -> MemoryBundle:
        return self.pipeline.search(query, k=k, memory_types=tuple(memory_types),
                                    namespace=self.namespace)

    def report_feedback(self, memory_ids: Sequence[str], helpful: bool) -> int:
        """Close the loop: usage outcome adjusts memory quality signals.

        Playbook bullets get helpful/harmful counters (ACE); strategy items
        get reward-shaped scores +1/-2 (G-Memory backward). Returns the
        number of memories updated."""
        ops: list[MemoryOp] = []
        for mid in memory_ids:
            bullets = self.doc.get_items([mid], "playbook")
            if bullets:
                field = "helpful" if helpful else "harmful"
                ops.append(MemoryOp(op=OpType.UPDATE, target_type="playbook",
                                    target_id=mid,
                                    payload={field: int(bullets[0].get(field, 0)) + 1}))
                continue
            strategies = self.doc.get_items([mid], "strategies")
            if strategies:
                delta = 1.0 if helpful else -2.0
                ops.append(MemoryOp(op=OpType.UPDATE, target_type="strategies",
                                    target_id=mid,
                                    payload={"score": float(strategies[0].get("score", 0)) + delta}))
        self._apply_ops(ops, actor="feedback")
        return len(ops)

    def get_playbook(self, section: str | None = None, k: int = 200) -> str:
        """Render the ACE playbook (all bullets, grouped by section)."""
        hits = self.vec.search(self.embedder.embed(["playbook"], kind="query")[0],
                               k=k, memory_type="playbook", namespace=self.namespace)
        bullets = self.doc.get_items([h[0] for h in hits], "playbook")
        if section:
            bullets = [b for b in bullets if b.get("section") == section]
        by_section: dict[str, list[str]] = {}
        for b in bullets:
            by_section.setdefault(b.get("section", "general"), []).append(
                f"[{b.get('section','general')}-{b['id'][:5]}] "
                f"helpful={b.get('helpful', 0)} harmful={b.get('harmful', 0)} "
                f":: {b.get('content', '')}")
        return "\n".join(f"## {s}\n" + "\n".join(lines)
                         for s, lines in sorted(by_section.items()))

    # ---- introspection --------------------------------------------------------

    @property
    def log(self):
        return self.doc  # EvolutionLog protocol: tail()/count()

    def stats(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "profile": self.config.profile,
            "episodes": self.doc.count_episodes(self.namespace),
            "vectors": self.vec.count(),
            "evolution_ops": self.doc.count(),
            "llm": self.budget.summary(),
            "structured_drops": dict(self.structured.drops) if self.structured else {},
            "embedder": self.embedder.name,
            "vector_store": type(self.vec).__name__,
        }

    def capabilities(self) -> dict[str, Any]:
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
                "vector_store": type(self.vec).__name__,
                "organizers": [o.name for o in self.organizers],
            },
            "degradations": self._degradations,
        }

    def close(self) -> None:
        if self._queue is not None:
            self._queue.join()  # drain pending organizer work before closing
        self.vec.close()
        self.doc.close()
