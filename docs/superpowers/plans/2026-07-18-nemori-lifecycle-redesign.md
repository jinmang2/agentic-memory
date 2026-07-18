# Nemori Lifecycle Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Organizer 계약에 MemoryEvent 구독(chaining)과 consolidate(유예 관리, 로그 커서)를 추가하고, Nemori를 v1/v4/upstream/mixing fidelity 스위치로 재구성하며, MemoryOS·A-Mem을 `input="episodes"` consumer로 마이그레이션한다.

**Architecture:** 스펙 `docs/superpowers/specs/2026-07-18-nemori-lifecycle-redesign-design.md` 참조. organizer는 store를 직접 만지지 않고 MemoryOp만 반환(불변 원칙); 퍼사드가 적용 후 ADD/UPDATE/MERGE를 MemoryEvent로 변환해 `consumes` 선언한 다른 organizer에 전달(depth=1). supersession은 MERGE op payload의 `supersedes` 키로 원자 전달. consolidate 커서 = evolution_log seq, `state` 타입 항목으로 영속.

**Tech Stack:** Python 3.11+, pytest, 기존 agmem 스택 (sqlite/lancedb/qdrant/pgserver, sentence-transformers, StructuredCaller).

## Global Constraints

- 포매팅: `ruff format` 100자; 주석 길이 제한 없음. 커밋 전 `ruff check src tests`.
- 테스트: `.venv/bin/python -m pytest tests/ -q` (프로젝트 관례). 테스트 더블은 `tests/helpers.py`의 `StubLLM`(role별 큐), `agmem.embed.fake.FakeEmbedder(dim=128)`, 퍼사드는 `make_mem` 패턴 (test_organizers.py 상단 참조).
- organizer는 ctx.doc/ctx.vec **읽기**만, 쓰기는 전부 MemoryOp 반환 (docs/04 §2).
- 기존 테스트 전부 통과 유지: `nemori`(무인자) = v1 동치, `memoryos`/`amem` 기본 동작 불변.
- 프리셋 값 혼용 금지: v4는 논문값(w=20, K_e=5, K_m=5, τ=0.70, 시간갭 제한 없음), upstream은 코드값(batch_threshold=20, buffer 2/25, sim 0.85, top-5, **>1h 갭 병합 금지 — upstream MERGE_DECISION 프롬프트 원문 "Do NOT merge if: They are separated by significant time gaps (>1 hour)"에서 확정**).
- DELETE/INVALIDATE 이벤트 비전파. INVALIDATE는 invalid_at 최초 보존 + facts 외 타입 벡터 제거.

---

### Task 1: DocStore `ops_since`/`last_seq` + 벡터 delete 멱등성

**Files:**
- Modify: `src/agmem/stores/base.py` (DocStore Protocol), `src/agmem/stores/sqlite_doc.py`, `src/agmem/stores/postgres_doc.py`
- Test: `tests/test_stores.py`

**Interfaces:**
- Produces: `DocStore.ops_since(seq: int, target_type: str | None = None, limit: int = 10000) -> list[tuple[int, MemoryOp]]` (seq 오름차순, seq > 인자), `DocStore.last_seq() -> int` (빈 로그면 0). `VectorStore.delete(missing_ids)`가 예외 없이 무시됨을 계약 테스트로 보장.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_stores.py`에 추가:

```python
def test_ops_since_and_last_seq():
    from agmem.core.ops import MemoryOp, OpType
    from agmem.stores.sqlite_doc import SqliteDocStore

    doc = SqliteDocStore(None)
    assert doc.last_seq() == 0
    doc.append([MemoryOp(op=OpType.ADD, target_type="semantic", target_id="s1"),
                MemoryOp(op=OpType.ADD, target_type="episodes", target_id="e1")])
    doc.append([MemoryOp(op=OpType.UPDATE, target_type="semantic", target_id="s1")])
    end = doc.last_seq()
    assert end == 3
    all_ops = doc.ops_since(0)
    assert [o.target_id for _, o in all_ops] == ["s1", "e1", "s1"]
    assert [s for s, _ in all_ops] == [1, 2, 3]
    sem = doc.ops_since(1, target_type="semantic")
    assert [(s, o.op) for s, o in sem] == [(3, OpType.UPDATE)]

def test_vector_delete_missing_is_noop(vec_store):  # 기존 스토어 파라미트라이즈 픽스처 재사용
    vec_store.delete(["no-such-id"])  # 예외 없어야 함
```

(기존 test_stores.py에 벡터 스토어 파라미트라이즈 픽스처가 있으면 재사용, 없으면 SqliteVecStore/NumpyVectorStore 2종에 대해 직접 인스턴스화.)

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_stores.py -k "ops_since or delete_missing" -q` → FAIL (`ops_since` 미정의)

- [ ] **Step 3: 구현** — `sqlite_doc.py` `append`/`tail` 옆에:

```python
def ops_since(
    self, seq: int, target_type: str | None = None, limit: int = 10000
) -> list[tuple[int, MemoryOp]]:
    """Read log entries after ``seq`` in order — the consolidate cursor surface."""
    q = (
        "SELECT seq, op, target_type, target_id, payload, actor, t_transaction"
        " FROM evolution_log WHERE seq > ?"
    )
    args: list = [seq]
    if target_type is not None:
        q += " AND target_type = ?"
        args.append(target_type)
    q += " ORDER BY seq ASC LIMIT ?"
    args.append(limit)
    rows = self._conn.execute(q, args).fetchall()
    return [(int(r[0]), MemoryOp.from_row(*r[1:])) for r in rows]

def last_seq(self) -> int:
    row = self._conn.execute("SELECT MAX(seq) FROM evolution_log").fetchone()
    return int(row[0] or 0)
```

`postgres_doc.py`에도 동일 의미로 미러 (그 파일의 evolution_log 테이블/컬럼명과 파라미터 스타일(%s)을 append 구현에서 확인해 맞출 것; seq 컬럼이 SERIAL/IDENTITY가 아니면 이 태스크에서 추가). `stores/base.py` DocStore Protocol에 두 메서드 시그니처 추가. 벡터 delete가 없는 id에 예외를 던지는 스토어가 있으면 해당 어댑터에서 존재 확인/예외 흡수로 수정.

- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS. 회귀: `.venv/bin/python -m pytest tests/test_stores.py -q`
- [ ] **Step 5: 커밋** — `git add -A src/agmem/stores tests/test_stores.py && git commit -m "feat(stores): ops_since/last_seq cursor surface + idempotent vector delete"`

---

### Task 2: 계약 — MemoryEvent, consumes, on_memory_event, consolidate, 커서 헬퍼

**Files:**
- Modify: `src/agmem/organizers/base.py`
- Test: `tests/test_lifecycle.py` (신규)

**Interfaces:**
- Produces:

```python
@dataclass
class MemoryEvent:
    source: str
    op: OpType
    target_type: str
    target_id: str          # MERGE면 살아남는 항목 id
    payload: dict
    supersedes: tuple[str, ...] = ()

class Organizer:
    consumes: tuple[str, ...] = ()
    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]: return []
    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]: return []
    def read_cursor(self, ctx: OrganizerContext) -> int: ...   # state 항목에서 seq, 없으면 0
    def cursor_op(self, seq: int) -> MemoryOp: ...             # UPDATE state consolidate:{name}
```

- [ ] **Step 1: 실패 테스트**

```python
"""Lifecycle contract: events, cursor, consolidate (spec §1)."""
from agmem.core.ops import MemoryOp, OpType
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext


def test_base_defaults_are_noop():
    org = Organizer()
    assert org.consumes == ()
    ev = MemoryEvent(source="x", op=OpType.ADD, target_type="episodes",
                     target_id="e1", payload={})
    assert org.on_memory_event(ev, None) == []
    assert org.consolidate(None) == []
    assert ev.supersedes == ()


def test_cursor_helpers_roundtrip():
    from agmem.stores.sqlite_doc import SqliteDocStore

    class C(Organizer):
        name = "curs"

    org, doc = C(), SqliteDocStore(None)
    ctx = OrganizerContext(doc=doc, vec=None, embedder=None, namespace="t")
    assert org.read_cursor(ctx) == 0
    op = org.cursor_op(7)
    assert (op.op, op.target_type, op.target_id) == (OpType.UPDATE, "state", "consolidate:curs")
    doc.put_item("consolidate:curs", "state", "t", {"id": "consolidate:curs", "seq": 7})
    assert org.read_cursor(ctx) == 7
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_lifecycle.py -q` → FAIL (MemoryEvent import 불가)
- [ ] **Step 3: 구현** — base.py에 (모듈 docstring에 스펙 §1 훅 위상 표를 요약 반영):

```python
@dataclass
class MemoryEvent:
    """One applied ADD/UPDATE/MERGE, delivered to subscribed organizers.

    ``supersedes`` rides only on MERGE and lists same-type ids the merge
    absorbed — the atomic channel managers use to retire derived state
    (spec §1.2); INVALIDATE/DELETE ops are never propagated as events."""

    source: str
    op: OpType
    target_type: str
    target_id: str
    payload: dict
    supersedes: tuple[str, ...] = ()


class Organizer:
    name = "base"
    consumes: tuple[str, ...] = ()

    # ... 기존 훅들 유지 ...

    def on_memory_event(self, ev: MemoryEvent, ctx: OrganizerContext) -> list[MemoryOp]:
        """Chaining hook: another organizer's applied output, if subscribed
        via ``consumes``. Runs inline (same dispatch as on_message); returned
        ops are applied but NOT re-propagated (depth=1)."""
        return []

    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        """Deferred management pass — only via AgenticMemory.consolidate().
        Implementations resume from read_cursor() and end their batch with
        cursor_op(new_seq) so progress survives restarts (spec §1.4)."""
        return []

    def read_cursor(self, ctx: OrganizerContext) -> int:
        items = ctx.doc.get_items([f"consolidate:{self.name}"], "state")
        return int(items[0].get("seq", 0)) if items else 0

    def cursor_op(self, seq: int) -> MemoryOp:
        return MemoryOp(
            op=OpType.UPDATE,
            target_type="state",
            target_id=f"consolidate:{self.name}",
            payload={"seq": seq},
        )
```

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(contract): MemoryEvent + on_memory_event/consolidate hooks + cursor helpers"`

---

### Task 3: 퍼사드 — INVALIDATE 규칙 (최초 보존, superseded_by, 벡터 제거/facts 예외)

**Files:**
- Modify: `src/agmem/memory.py` (`_apply_one` INVALIDATE 분기, 모듈 상수)
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `BITEMPORAL_TYPES = ("facts",)` (memory.py 모듈 상수). INVALIDATE 적용 규칙: invalid_at은 setdefault(최초 보존), payload의 `superseded_by` 반영, target_type ∉ BITEMPORAL_TYPES면 `vec.delete([target_id])`.

- [ ] **Step 1: 실패 테스트**

```python
def _mk(organizers=("passthrough",)):
    from agmem import AgenticMemory
    from agmem.embed.fake import FakeEmbedder
    return AgenticMemory(namespace="t", organizers=list(organizers),
                         embedder=FakeEmbedder(dim=128))


def test_invalidate_preserves_first_and_removes_vector():
    mem = _mk()
    mem._apply_ops([MemoryOp(op=OpType.ADD, target_type="semantic", target_id="s1",
                             payload={"id": "s1", "content": "fact", "embedding_text": "fact"})],
                   actor="t")
    assert mem.vec.count() == 1
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="semantic", target_id="s1",
                             payload={"t_invalid": "2026-01-01T00:00:00",
                                      "superseded_by": "s2"})], actor="t")
    item = mem.doc.get_items(["s1"], "semantic")[0]
    assert item["invalid_at"] == "2026-01-01T00:00:00"
    assert item["superseded_by"] == "s2"
    assert mem.vec.count() == 0  # semantic은 bi-temporal 렌더 타입이 아님 → 벡터 제거
    # 이중 무효화: 최초 시각 보존, 예외 없음
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="semantic", target_id="s1",
                             payload={"t_invalid": "2027-01-01T00:00:00"})], actor="t")
    assert mem.doc.get_items(["s1"], "semantic")[0]["invalid_at"] == "2026-01-01T00:00:00"


def test_invalidate_facts_keeps_vector():
    mem = _mk()
    mem._apply_ops([MemoryOp(op=OpType.ADD, target_type="facts", target_id="f1",
                             payload={"id": "f1", "content": "A는 B다", "embedding_text": "A는 B다"})],
                   actor="t")
    mem._apply_ops([MemoryOp(op=OpType.INVALIDATE, target_type="facts", target_id="f1",
                             payload={})], actor="t")
    assert mem.vec.count() == 1  # Zep bi-temporal: 무효화돼도 validity 렌더 대상
```

- [ ] **Step 2: 실패 확인** → FAIL (벡터 잔존/invalid_at 덮어씀)
- [ ] **Step 3: 구현** — `_apply_one`의 INVALIDATE 분기 교체:

```python
BITEMPORAL_TYPES = ("facts",)  # 무효화 후에도 validity 구간으로 렌더되는 타입 (Zep)

        elif op.op == OpType.INVALIDATE:
            items = self.doc.get_items([op.target_id], op.target_type)
            if items:
                data = items[0]
                # 최초 무효화 시각 보존 — 이중 무효화 멱등 (spec §1.2)
                data.setdefault(
                    "invalid_at", op.payload.get("t_invalid", utcnow().isoformat())
                )
                if "superseded_by" in op.payload:
                    data["superseded_by"] = op.payload["superseded_by"]
                self.doc.put_item(op.target_id, op.target_type, self.namespace, data)
                if op.target_type not in BITEMPORAL_TYPES:
                    # 서빙 제외 보장 — ghost-hit 방지(X1 계열, spec §1.3); doc/로그엔 남음
                    self.vec.delete([op.target_id])
```

- [ ] **Step 4: 통과 + 회귀** — `pytest tests/test_lifecycle.py tests/test_organizers.py tests/test_memory.py -q`
- [ ] **Step 5: 커밋** — `git commit -m "feat(facade): INVALIDATE keeps first invalid_at, records superseded_by, drops vector (facts exempt)"`

---

### Task 4: 퍼사드 — 이벤트 전파 (depth=1, payload-carried supersedes)

**Files:**
- Modify: `src/agmem/memory.py` (`_apply_ops`, 신규 `_propagate_events`)
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Consumes: Task 2 MemoryEvent.
- Produces: `_apply_ops(self, ops, actor, propagate=True)`. 전파 규칙: ADD/UPDATE/MERGE만; `ev.supersedes = tuple(op.payload.get("supersedes", ()))` (MERGE op payload에 organizer가 명시 — 배치 추론 아님, 복수 MERGE 정확 귀속); 수신자는 `target_type ∈ consumes`이고 `name != actor`인 organizer, 리스트 순서; 반환 ops는 `propagate=False`로 적용(depth=1).

- [ ] **Step 1: 실패 테스트**

```python
class Emitter(Organizer):
    name = "emitter"
    def on_message(self, ep, ctx):
        return [
            MemoryOp(op=OpType.MERGE, target_type="episodes", target_id="new",
                     payload={"id": "new", "content": "merged", "supersedes": ["old"]}),
            MemoryOp(op=OpType.INVALIDATE, target_type="episodes", target_id="old",
                     payload={"superseded_by": "new"}),
        ]

class Consumer(Organizer):
    name = "consumer"
    consumes = ("episodes",)
    def __init__(self):
        self.seen: list[MemoryEvent] = []
    def on_memory_event(self, ev, ctx):
        self.seen.append(ev)
        # depth=1 검증용: consumer도 episodes를 반환하지만 재전파되면 안 됨
        return [MemoryOp(op=OpType.ADD, target_type="episodes", target_id=f"d1-{len(self.seen)}",
                         payload={"id": f"d1-{len(self.seen)}", "content": "derived"})]


def test_event_propagation_supersedes_and_depth1():
    consumer = Consumer()
    mem = _mk(organizers=[Emitter(), consumer])
    mem.add_message("hi")
    assert len(consumer.seen) == 1            # MERGE만 전파 (INVALIDATE 비전파)
    ev = consumer.seen[0]
    assert (ev.op, ev.target_id, ev.supersedes) == (OpType.MERGE, "new", ("old",))
    assert ev.source == "emitter"
    # consumer의 반환 op는 적용됐지만 (자기 자신에게도) 재전파되지 않음
    assert mem.doc.get_items(["d1-1"], "episodes")
    assert len(consumer.seen) == 1


def test_no_self_delivery_and_consumes_filter():
    class SelfSub(Emitter):
        name = "selfsub"
        consumes = ("episodes",)
        def __init__(self): self.seen = []
        def on_memory_event(self, ev, ctx):
            self.seen.append(ev); return []
    org = SelfSub()
    mem = _mk(organizers=[org])
    mem.add_message("hi")
    assert org.seen == []  # 자기 이벤트 제외
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현**

```python
    def _apply_ops(self, ops: list[MemoryOp], actor: str, propagate: bool = True) -> None:
        if not ops:
            return
        for op in ops:
            op.actor = actor
        self.doc.append(ops)  # log first — replayable audit trail
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
                supersedes=tuple(op.payload.get("supersedes", ()))
                if op.op is OpType.MERGE
                else (),
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
```

(import에 MemoryEvent 추가: `from agmem.organizers import ... , MemoryEvent` — organizers/__init__.py export도 갱신.)

- [ ] **Step 4: 통과 + 회귀** — `pytest tests/test_lifecycle.py tests/test_organizers.py -q`
- [ ] **Step 5: 커밋** — `git commit -m "feat(facade): MemoryEvent propagation — consumes match, depth=1, payload supersedes"`

---

### Task 5: 퍼사드 — consolidate() 공개 API

**Files:**
- Modify: `src/agmem/memory.py`
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `AgenticMemory.consolidate() -> int` (적용한 op 수 반환; organizer 리스트 순서로 각 `consolidate(ctx)` 호출·적용, 이벤트 전파 포함).

- [ ] **Step 1: 실패 테스트**

```python
def test_consolidate_api_applies_ops_and_cursor():
    class Cons(Organizer):
        name = "cons"
        def consolidate(self, ctx):
            end = ctx.doc.last_seq()
            return [MemoryOp(op=OpType.ADD, target_type="semantic", target_id="c1",
                             payload={"id": "c1", "content": "merged fact",
                                      "consolidated": True, "embedding_text": "merged fact"}),
                    self.cursor_op(end)]
    org = Cons()
    mem = _mk(organizers=[org])
    n = mem.consolidate()
    assert n == 2
    assert mem.doc.get_items(["c1"], "semantic")
    assert org.read_cursor(mem._ctx) > 0
    # state 항목은 벡터를 만들지 않는다
    assert mem.vec.count() == 1
```

- [ ] **Step 2: 실패 확인** → FAIL (`consolidate` 속성 없음)
- [ ] **Step 3: 구현** — flush() 아래:

```python
    def consolidate(self) -> int:
        """Deferred management pass (spec §1.4) — explicit trigger only.

        Runs each organizer's consolidate() in list order and applies the
        returned ops through the evolution log. Benchmarks call this at
        deterministic points (end of ingest / between sessions)."""
        applied = 0
        for org in self.organizers:
            ops = org.consolidate(self._ctx)
            self._apply_ops(ops, actor=org.name)
            applied += len(ops)
        return applied
```

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(facade): explicit consolidate() API"`

---

### Task 6: Nemori 스테이지 분리 (v1 동치 리팩터)

**Files:**
- Create: `src/agmem/organizers/nemori_stages.py`
- Modify: `src/agmem/organizers/nemori.py`
- Test: 기존 `tests/test_organizers.py -k nemori` 무수정 통과가 기준

**Interfaces:**
- Produces (nemori_stages.py):

```python
class PerMessageBoundary:
    def __init__(self, confidence: float = 0.7,
                 buffer_min: int = 2, buffer_max: int = 25): ...
    # LLM은 생성자 주입이 아니라 push(buffer, ctx)의 ctx.llm 사용 (스테이지 전 공통)
    def push(self, buffer: list[Episode], ctx) -> tuple[list[list[Episode]], list[Episode]]:
        """(flush할 세그먼트들, 남는 버퍼). v1 f_theta 로직."""
    def flush(self, buffer: list[Episode], ctx) -> list[list[Episode]]:
        """잔여 버퍼 전체를 마지막 세그먼트(들)로."""
```

- [ ] **Step 1: BOUNDARY_PROMPT/BOUNDARY_SCHEMA와 on_message의 경계 판정 로직을 `PerMessageBoundary`로 이동** — push()는 현행 nemori.py:234-259와 동일 의미: `len<buffer_min → ([], buffer)`, `len>=buffer_max → ([buffer], [])`, 그 외 LLM 판정 → boundary∧conf≥σ면 `([buffer[:-1]], [buffer[-1:]])`, 아니면 `([], buffer)`. `_fmt`도 stages로 이동(공용). flush()는 `[buffer] if buffer else []`.
- [ ] **Step 2: NemoriOrganizer를 합성으로 전환** — `__init__`에서 `self._segmenter = PerMessageBoundary(boundary_confidence, buffer_min, buffer_max)` (LLM은 push(buffer, ctx)에서 ctx.llm 사용 — 스테이지 전체 ctx 주입으로 통일). on_message:

```python
    def on_message(self, ep: Episode, ctx: OrganizerContext) -> list[MemoryOp]:
        if ctx.llm is None:
            ...기존 경고/return 유지...
        self.buffer.append(ep)
        segments, self.buffer = self._segmenter.push(self.buffer, ctx)
        ops: list[MemoryOp] = []
        for seg in segments:
            ops.extend(self._flush_segment(seg, ctx))
        return ops
```

flush_buffer도 `self._segmenter.flush(...)` 사용으로 교체.
- [ ] **Step 3: 기존 테스트 무수정 통과 확인** — `pytest tests/test_organizers.py -k nemori -q` → PASS (v1 동치 증명). `pytest tests/ -q` 회귀.
- [ ] **Step 4: 커밋** — `git commit -m "refactor(nemori): extract PerMessageBoundary segmenter stage (v1-equivalent)"`

---

### Task 7: BatchPartitioner (v4 LMP / upstream 배치 분할)

**Files:**
- Modify: `src/agmem/organizers/nemori_stages.py`
- Test: `tests/test_nemori_fidelity.py` (신규)

**Interfaces:**
- Produces:

```python
class BatchPartitioner:
    def __init__(self, window: int = 20, buffer_min: int = 2, chunk_max: int = 80): ...
    def push(self, buffer, ctx) -> tuple[list[list[Episode]], list[Episode]]
    def flush(self, buffer, ctx) -> list[list[Episode]]
```

- [ ] **Step 1: 실패 테스트**

```python
from helpers import StubLLM
from agmem.core.types import Episode
from agmem.organizers.nemori_stages import BatchPartitioner


def _eps(n):
    return [Episode(content=f"m{i}") for i in range(n)]


def test_batch_partitioner_waits_then_partitions():
    seg = BatchPartitioner(window=4)
    llm = StubLLM({"extract": [{"episodes": [
        {"indices": [0, 1], "topic": "a"}, {"indices": [2, 3], "topic": "b"}]}]})
    ctx = type("C", (), {"llm": llm})()
    out, rest = seg.push(_eps(3), ctx)
    assert out == [] and len(rest) == 3          # window 미달 — LLM 콜 없음
    assert llm.calls == []
    out, rest = seg.push(_eps(4), ctx)
    assert [len(s) for s in out] == [2, 2] and rest == []


def test_batch_partitioner_flush_small_tail_single_group():
    seg = BatchPartitioner(window=20, buffer_min=2)
    llm = StubLLM({"extract": []})
    ctx = type("C", (), {"llm": llm})()
    assert [len(s) for s in seg.flush(_eps(3), ctx)] == [3]  # <window: 단일 그룹, LLM 없음
    assert llm.calls == []


def test_batch_partitioner_llm_failure_falls_back_to_one_segment():
    seg = BatchPartitioner(window=2)
    ctx = type("C", (), {"llm": StubLLM({"extract": []})})()  # 응답 소진 → None
    out, rest = seg.push(_eps(2), ctx)
    assert [len(s) for s in out] == [2] and rest == []
```

- [ ] **Step 2: 실패 확인** — `pytest tests/test_nemori_fidelity.py -q` → FAIL
- [ ] **Step 3: 구현**

```python
BATCH_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {"episodes": {"type": "array", "items": {"type": "object", "properties": {
        "indices": {"type": "array", "items": {"type": "integer"}},
        "topic": {"type": "string"}}, "required": ["indices"]}}},
    "required": ["episodes"],
}

# Condensed from upstream BATCH_SEGMENTATION_PROMPT (v4 Local Message
# Partitioning P_par): topic independence, 2-15 messages, when in doubt
# split, relevance < ~30% cut; indexed batch in, index groups out.
BATCH_SEGMENT_PROMPT = """Partition this conversation into topically coherent
episodes. Each episode centers on ONE core topic or event. Signals for a cut:
explicit topic shifts, intent transitions, temporal markers ("earlier",
"by the way", 30+ minute gaps), structural signals (transition phrases,
concluding statements), content relatedness below ~30%. Episodes work best
with 2-15 messages; when in doubt, split. Every message index must appear in
exactly one episode, in order.

Messages (indexed):
{messages}

Return JSON: {{"episodes": [{{"indices": [0, 1, ...], "topic": "..."}}]}}"""


class BatchPartitioner:
    """v4 §3.2.1 Local Message Partitioning / upstream batch segmentation.

    Buffers until ``window`` then LLM-partitions the whole buffer. flush()
    on a tail shorter than the window stores it as one segment without an
    LLM call — upstream's single-'conversation'-group path below
    batch_threshold."""

    def __init__(self, window: int = 20, buffer_min: int = 2, chunk_max: int = 80) -> None:
        self.window = window
        self.buffer_min = buffer_min
        self.chunk_max = chunk_max  # upstream: >80msg batches are chunked

    def push(self, buffer, ctx):
        if len(buffer) < self.window:
            return [], buffer
        return self._partition(buffer, ctx), []

    def flush(self, buffer, ctx):
        if not buffer:
            return []
        if len(buffer) < self.window:
            return [buffer]
        return self._partition(buffer, ctx)

    def _partition(self, buffer, ctx):
        segments: list[list] = []
        for start in range(0, len(buffer), self.chunk_max):
            chunk = buffer[start : start + self.chunk_max]
            indexed = "\n".join(f"[{i}] {_fmt(e)}" for i, e in enumerate(chunk))
            result = ctx.llm.call(
                "extract",
                BATCH_SEGMENT_PROMPT.format(messages=indexed),
                BATCH_SEGMENT_SCHEMA,
                required_keys=("episodes",),
            )
            groups = (result or {}).get("episodes") or []
            covered: set[int] = set()
            for g in groups:
                idxs = sorted(
                    i for i in g.get("indices", [])
                    if isinstance(i, int) and 0 <= i < len(chunk) and i not in covered
                )
                if not idxs:
                    continue
                covered.update(idxs)
                segments.append([chunk[i] for i in idxs])
            leftover = [chunk[i] for i in range(len(chunk)) if i not in covered]
            if leftover:  # LLM 실패/누락 인덱스 — 세그먼트를 잃지 않는다 (프로젝트 원칙)
                segments.append(leftover)
        return segments
```

- [ ] **Step 4: 통과 확인** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(nemori): BatchPartitioner segmenter (v4 LMP / upstream batch)"`

---

### Task 8: EpisodeMerger (v4 §3.2.3 / upstream merger)

**Files:**
- Modify: `src/agmem/organizers/nemori_stages.py`, `src/agmem/organizers/nemori.py` (`_flush_segment`에 훅)
- Test: `tests/test_nemori_fidelity.py`

**Interfaces:**
- Produces:

```python
class EpisodeMerger:
    def __init__(self, top_k: int = 5, similarity: float | None = None,
                 time_gap_hours: float | None = None): ...
    def merge_or_none(self, title, narrative, ep_ts, source_ids, ctx
                     ) -> tuple[list[MemoryOp], str, str, str] | None:
        """병합이면 ([MERGE(new_id, payload+supersedes), INVALIDATE(old)], new_id,
        merged_title, merged_narrative); 병합 아님/후보 없음/LLM 실패면 None."""
```

- Consumes: Task 4의 payload-carried `supersedes` 규약.

- [ ] **Step 1: 실패 테스트** (퍼사드 경유 — make_mem 패턴):

```python
def test_episode_merger_merges_and_supersedes():
    from agmem.organizers.nemori import NemoriOrganizer
    llm = StubLLM({
        "extract": [{"boundary": True, "confidence": 0.9}] * 8,
        "distill": [
            # 1st episode: narrate + direct-extract(cold start)
            {"title": "hiking plan", "narrative": "User planned a hike.", "timestamp": "2026-05-01"},
            {"facts": []},
            # 2nd episode: narrate → merge decision(merge, target 0) → merged content
            {"title": "hiking plan 2", "narrative": "More hiking talk.", "timestamp": "2026-05-01"},
            {"decision": "merge", "target_index": 0},
            {"title": "hiking plan (merged)", "narrative": "Combined hike story.",
             "timestamp": "2026-05-01"},
            {"facts": []},  # PC over merged episode (semantic 있으면 predict/calibrate로 조정)
        ],
    })
    org = NemoriOrganizer(episode_merge="llm", buffer_min=1, boundary_confidence=0.7)
    mem = make_mem(org, llm)
    ...  # 메시지 2턴 인제스트해 에피소드 2개 유도 (경계 판정 응답으로 제어)
    eps = [o for o in mem.log.tail(50) if o.target_type == "episodes"]
    merged = [o for o in eps if o.op == OpType.MERGE]
    inv = [o for o in eps if o.op == OpType.INVALIDATE]
    assert len(merged) == 1 and len(inv) == 1
    assert merged[0].payload["supersedes"] == [inv[0].target_id]
    assert inv[0].payload["superseded_by"] == merged[0].target_id
```

(정확한 메시지 시퀀스/StubLLM 큐는 구현하며 조정 — 검증 포인트는 MERGE.payload.supersedes ↔ INVALIDATE.superseded_by 쌍과 병합 서사로 에피소드가 남는 것.)

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현** — nemori_stages.py:

```python
MERGE_DECISION_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["merge", "new"]},
                   "target_index": {"type": "integer"}},
    "required": ["decision"],
}
MERGE_CONTENT_SCHEMA = {  # EPISODE_SCHEMA와 동형 — 순환 import 회피 위해 stages에 정의
    "type": "object",
    "properties": {"title": {"type": "string"}, "narrative": {"type": "string"},
                   "timestamp": {"type": "string"}},
    "required": ["title", "narrative"],
}

# Condensed from upstream MERGE_DECISION; the >1h ban line is injected only
# when time_gap_hours is set (upstream preset) — the v4 paper has no time
# constraint (verified 2026-07-18, docs/research/write-path-lifecycle-survey.md §1.2).
MERGE_DECISION_PROMPT = """A new episodic memory arrived. Decide whether it
describes the SAME event as one of the candidate episodes and should be
merged into it, or is a distinct episode.
Merge only when they cover the same underlying event or activity thread.
{time_gap_rule}
New episode:
title: {title}
narrative: {narrative}
timestamp: {timestamp}

Candidates (indexed):
{candidates}

Return JSON: {{"decision": "merge" or "new", "target_index": <candidate index, only when merging>}}"""

TIME_GAP_RULE = ("Do NOT merge if they are separated by significant time gaps "
                 "(>{hours} hour(s)) between their timestamps.")

# Condensed from upstream MERGE_CONTENT: synthesize without duplication,
# chronological flow, keep participants/decisions/emotions/outcomes,
# earliest timestamp wins.
MERGE_CONTENT_PROMPT = """Merge these two episodic memories about the same event
into ONE. Synthesize without duplication, preserve chronological event flow,
retain all critical details (participants, decisions, emotions, outcomes).
Use the EARLIEST timestamp.

Episode A:
title: {title_a}
narrative: {narrative_a}
timestamp: {ts_a}

Episode B:
title: {title_b}
narrative: {narrative_b}
timestamp: {ts_b}

Return JSON: {{"title": "...", "narrative": "...", "timestamp": "ISO"}}"""


class EpisodeMerger:
    def __init__(self, top_k=5, similarity=None, time_gap_hours=None):
        self.top_k = top_k
        self.similarity = similarity          # upstream 0.85; v4 논문엔 없음(None)
        self.time_gap_hours = time_gap_hours  # upstream 1.0; v4 None

    def merge_or_none(self, title, narrative, ep_ts, source_ids, ctx):
        emb = ctx.embedder.embed([f"{title}\n{narrative}"])[0]
        hits = ctx.vec.search(emb, k=self.top_k, memory_type="episodes",
                              namespace=ctx.namespace)
        if self.similarity is not None:
            hits = [(hid, s) for hid, s in hits if s >= self.similarity]
        cands = [c for c in ctx.doc.get_items([h[0] for h in hits], "episodes")
                 if not c.get("invalid_at")]
        if not cands:
            return None
        gap_rule = (TIME_GAP_RULE.format(hours=self.time_gap_hours)
                    if self.time_gap_hours else "")
        cand_text = "\n".join(
            f"[{i}] title: {c.get('title','')} | timestamp: {c.get('timestamp','')}\n"
            f"    narrative: {c.get('content','')}" for i, c in enumerate(cands))
        verdict = ctx.llm.call("distill", MERGE_DECISION_PROMPT.format(
            time_gap_rule=gap_rule, title=title, narrative=narrative,
            timestamp=ep_ts, candidates=cand_text),
            MERGE_DECISION_SCHEMA, required_keys=("decision",))
        if not verdict or verdict.get("decision") != "merge":
            return None
        idx = verdict.get("target_index")
        if not isinstance(idx, int) or not 0 <= idx < len(cands):
            return None
        old = cands[idx]
        merged = ctx.llm.call("distill", MERGE_CONTENT_PROMPT.format(
            title_a=old.get("title", ""), narrative_a=old.get("content", ""),
            ts_a=old.get("timestamp", ""), title_b=title, narrative_b=narrative,
            ts_b=ep_ts), MERGE_CONTENT_SCHEMA, required_keys=("title", "narrative"))
        if merged is None:
            return None  # 병합 실패 → 호출측이 일반 ADD로 저장 (세그먼트 불손실)
        new_id = new_id()  # from agmem.core.types
        m_title = str(merged.get("title", "")).strip() or title
        m_narr = str(merged.get("narrative", "")).strip() or narrative
        m_ts = str(merged.get("timestamp", "")).strip() or min(
            str(old.get("timestamp", "")), str(ep_ts))
        ops = [
            MemoryOp(op=OpType.MERGE, target_type="episodes", target_id=new_id,
                     payload={"id": new_id, "title": m_title, "content": m_narr,
                              "timestamp": m_ts, "supersedes": [old["id"]],
                              "source_episode_ids": list(old.get("source_episode_ids", []))
                              + list(source_ids),
                              "embedding_text": f"{m_title}\n{m_narr}"}),
            MemoryOp(op=OpType.INVALIDATE, target_type="episodes", target_id=old["id"],
                     payload={"reason": "merged", "superseded_by": new_id}),
        ]
        return ops, new_id, m_title, m_narr
```

nemori.py `_flush_segment` 배선 (narrate 직후, PC 직전 — v4 Alg.1 순서):

```python
        episode_id = new_id()
        merged = self._merger.merge_or_none(title, narrative, ep_ts, source_ids, ctx) \
            if self._merger else None
        if merged is not None:
            merge_ops, episode_id, title, narrative = merged
            ops = list(merge_ops)   # ADD 대신 MERGE+INVALIDATE
        else:
            ops = [MemoryOp(op=OpType.ADD, target_type="episodes", target_id=episode_id,
                            payload={... 기존 ADD payload ...})]
        ops.extend(self._predict_calibrate(title, narrative, plain_text,
                                           episode_id, source_ids, ctx))
```

- [ ] **Step 4: 통과 + 회귀** — `pytest tests/test_nemori_fidelity.py tests/test_organizers.py -k "nemori or fidelity" -q`
- [ ] **Step 5: 커밋** — `git commit -m "feat(nemori): EpisodeMerger — MERGE+supersedes / INVALIDATE superseded (v4 §3.2.3, upstream 0.85/>1h)"`

---

### Task 9: SemanticIntegrator — DedupIdReuse + ThreeWay

**Files:**
- Modify: `src/agmem/organizers/nemori_stages.py`, `src/agmem/organizers/nemori.py` (`_predict_calibrate` Stage 3을 integrator에 위임)
- Test: `tests/test_nemori_fidelity.py`

**Interfaces:**
- Produces:

```python
class AppendIntegrator:      # 현행 동작 (fact별 ADD)
    def integrate(self, fact, episode_id, source_ids, ctx) -> list[MemoryOp]
class DedupIdReuseIntegrator:  # PR#19: top-1 유사도>=threshold면 기존 id UPDATE
    def __init__(self, threshold: float = 0.85)
class ThreeWayIntegrator:      # v4 §3.3.3: tau 필터 -> top-K_m -> LLM {new,merge,conflict}
    def __init__(self, top_k: int = 5, tau: float = 0.70)
```

세 클래스 모두 동일 `integrate` 시그니처 — NemoriOrganizer는 fact 루프에서 위임만.

- [ ] **Step 1: 실패 테스트**

```python
def test_dedup_id_reuse_updates_existing(): ...
    # 준비: semantic "User likes hiking" 저장(퍼사드 ADD) 후, 동일 fact를
    # DedupIdReuseIntegrator.integrate → 반환 op가 UPDATE(기존 id)이고
    # payload["episode_id"]가 새 에피소드로 갱신(PR#19 provenance refresh)임을 단언.

def test_three_way_merge_branch():
    # 준비: semantic 2건 저장. StubLLM distill 큐에
    # {"decision": "merge", "target_indexes": [0, 1], "statement": "unified"}.
    # integrate 반환: MERGE(new_id, supersedes=[두 id]) + INVALIDATE x2 (superseded_by=new_id).

def test_three_way_conflict_branch():
    # {"decision": "conflict", "target_indexes": [0], "statement": "corrected"}
    # → ADD(새 fact) + INVALIDATE(구, superseded_by=새 id).

def test_three_way_tau_filters_candidates():
    # vec 검색 점수 < tau뿐이면 LLM 콜 없이 ADD만 (llm.calls 빈 것 단언).
```

(각 테스트는 `_mk()` 퍼사드로 semantic을 심고 `ctx=mem._ctx`로 integrator를 직접 호출 — organizer 우회로 단위 검증.)

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현**

```python
INTEGRATE_SCHEMA = {
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": ["new", "merge", "conflict"]},
                   "target_indexes": {"type": "array", "items": {"type": "integer"}},
                   "statement": {"type": "string"}},
    "required": ["decision"],
}

# v4 §3.3.3 P_con condensed: new=distinct, merge=supersede with unified
# statement, conflict=the new insight invalidates the old entries.
INTEGRATE_PROMPT = """A new knowledge statement arrived. Compare it with the
existing similar statements and decide:
- "new": genuinely distinct knowledge → keep all
- "merge": same knowledge (possibly partial overlaps) → produce ONE unified
  statement superseding the indexed existing ones
- "conflict": the new statement invalidates/corrects the indexed existing
  ones → they must be replaced

New statement: {fact}

Existing (indexed):
{existing}

Return JSON: {{"decision": "new"|"merge"|"conflict",
"target_indexes": [affected indexes], "statement": "unified or corrected statement (merge/conflict only)"}}"""


class AppendIntegrator:
    def integrate(self, fact, episode_id, source_ids, ctx):
        fid = new_id()
        return [MemoryOp(op=OpType.ADD, target_type="semantic", target_id=fid,
                         payload={"id": fid, "content": fact, "episode_id": episode_id,
                                  "source_episode_ids": list(source_ids),
                                  "embedding_text": fact})]


class DedupIdReuseIntegrator:
    """PR#19 semantics: top-1 embedding match >= threshold reuses the id —
    latest content wins, provenance re-pointed. No LLM call."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def integrate(self, fact, episode_id, source_ids, ctx):
        emb = ctx.embedder.embed([fact])[0]
        hits = ctx.vec.search(emb, k=1, memory_type="semantic", namespace=ctx.namespace)
        if hits and hits[0][1] >= self.threshold:
            return [MemoryOp(op=OpType.UPDATE, target_type="semantic", target_id=hits[0][0],
                             payload={"content": fact, "episode_id": episode_id,
                                      "source_episode_ids": list(source_ids),
                                      "embedding_text": fact})]
        return AppendIntegrator().integrate(fact, episode_id, source_ids, ctx)


class ThreeWayIntegrator:
    def __init__(self, top_k: int = 5, tau: float = 0.70):
        self.top_k = top_k
        self.tau = tau

    def integrate(self, fact, episode_id, source_ids, ctx):
        emb = ctx.embedder.embed([fact])[0]
        hits = [(hid, s) for hid, s in
                ctx.vec.search(emb, k=self.top_k, memory_type="semantic",
                               namespace=ctx.namespace) if s >= self.tau]
        cands = [c for c in ctx.doc.get_items([h[0] for h in hits], "semantic")
                 if not c.get("invalid_at")]
        add = AppendIntegrator().integrate(fact, episode_id, source_ids, ctx)
        if not cands:
            return add
        existing = "\n".join(f"[{i}] {c.get('content', '')}" for i, c in enumerate(cands))
        verdict = ctx.llm.call("distill",
                               INTEGRATE_PROMPT.format(fact=fact, existing=existing),
                               INTEGRATE_SCHEMA, required_keys=("decision",))
        if verdict is None or verdict.get("decision") == "new":
            return add
        idxs = [i for i in verdict.get("target_indexes", [])
                if isinstance(i, int) and 0 <= i < len(cands)]
        if not idxs:
            return add
        statement = str(verdict.get("statement", "")).strip() or fact
        targets = [cands[i]["id"] for i in idxs]
        fid = new_id()
        if verdict["decision"] == "merge":
            head = MemoryOp(op=OpType.MERGE, target_type="semantic", target_id=fid,
                            payload={"id": fid, "content": statement, "episode_id": episode_id,
                                     "source_episode_ids": list(source_ids),
                                     "supersedes": targets, "embedding_text": statement})
        else:  # conflict — 신규가 구 항목들을 대체 (spec §2.3)
            head = MemoryOp(op=OpType.ADD, target_type="semantic", target_id=fid,
                            payload={"id": fid, "content": statement, "episode_id": episode_id,
                                     "source_episode_ids": list(source_ids),
                                     "embedding_text": statement})
        return [head] + [
            MemoryOp(op=OpType.INVALIDATE, target_type="semantic", target_id=t,
                     payload={"reason": verdict["decision"], "superseded_by": fid})
            for t in targets]
```

nemori.py `_predict_calibrate`의 Stage 3 루프를 `self._integrator.integrate(fact, episode_id, source_ids, ctx)` 위임으로 교체 (기본 AppendIntegrator — 기존 동작 보존).

- [ ] **Step 4: 통과 + 회귀** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(nemori): semantic integrators — append / PR#19 dedup / v4 three-way"`

---

### Task 10: fidelity 프리셋 해석

**Files:**
- Modify: `src/agmem/organizers/nemori.py`
- Test: `tests/test_nemori_fidelity.py`

**Interfaces:**
- Produces: `NemoriOrganizer(fidelity=None, segmenter=None, episode_merge=None, semantic_integration=None, consolidation=None, **numeric overrides)` — 프리셋이 미지정 파라미터를 채우고 명시 인자가 항상 이긴다. `NEMORI_PRESETS: dict[str, dict]` export.

- [ ] **Step 1: 실패 테스트**

```python
def test_presets_resolve_to_stages():
    from agmem.organizers.nemori import NemoriOrganizer
    from agmem.organizers.nemori_stages import (
        AppendIntegrator, BatchPartitioner, DedupIdReuseIntegrator,
        EpisodeMerger, PerMessageBoundary, ThreeWayIntegrator)

    v1 = NemoriOrganizer(fidelity="v1")
    assert isinstance(v1._segmenter, PerMessageBoundary)
    assert v1._merger is None and isinstance(v1._integrator, AppendIntegrator)

    v4 = NemoriOrganizer(fidelity="v4")
    assert isinstance(v4._segmenter, BatchPartitioner) and v4._segmenter.window == 20
    assert v4._merger.top_k == 5 and v4._merger.similarity is None \
        and v4._merger.time_gap_hours is None
    assert isinstance(v4._integrator, ThreeWayIntegrator) \
        and v4._integrator.tau == 0.70 and v4._integrator.top_k == 5

    up = NemoriOrganizer(fidelity="upstream")
    assert isinstance(up._segmenter, BatchPartitioner)
    assert up._merger.similarity == 0.85 and up._merger.time_gap_hours == 1.0
    assert isinstance(up._integrator, AppendIntegrator)

    # 명시 인자가 프리셋을 이긴다 (mixing)
    mix = NemoriOrganizer(fidelity="v4", semantic_integration="append",
                          consolidation="semantic_offline")
    assert isinstance(mix._integrator, AppendIntegrator)
    assert mix._consolidator is not None

    # 무인자 = v1 동치 (기존 config 호환)
    assert isinstance(NemoriOrganizer()._segmenter, PerMessageBoundary)
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현** — nemori.py:

```python
# 프리셋별 값 출처 혼용 금지 (docs/research/write-path-lifecycle-survey.md §5):
# v4 = 논문값 (w=20, K_e=5, K_m=5, tau=0.70, 시간갭 제한 없음)
# upstream = 코드값 (batch 20/2/25, chunk 80, sim 0.85, top-5, >1h gap ban)
NEMORI_PRESETS: dict[str, dict] = {
    "v1": dict(segmenter="per_message", episode_merge="off",
               semantic_integration="append", consolidation="off",
               boundary_confidence=0.7, buffer_min=2, buffer_max=25),
    "v4": dict(segmenter="batch", window=20, episode_merge="llm", merge_top_k=5,
               merge_similarity=None, merge_time_gap_hours=None,
               semantic_integration="llm3way", integrate_top_k=5, integrate_tau=0.70,
               consolidation="off"),
    "upstream": dict(segmenter="batch", window=20, buffer_min=2, buffer_max=25,
                     chunk_max=80, episode_merge="llm", merge_top_k=5,
                     merge_similarity=0.85, merge_time_gap_hours=1.0,
                     semantic_integration="append", consolidation="off"),
}
```

`__init__(self, fidelity=None, **kw)`: `params = dict(NEMORI_PRESETS.get(fidelity, NEMORI_PRESETS["v1"])); params.update({k: v for k, v in kw.items() if v is not None})` 후 스테이지 구성:

```python
        if params["segmenter"] == "batch":
            self._segmenter = BatchPartitioner(
                window=params.get("window", 20),
                buffer_min=params.get("buffer_min", 2),
                chunk_max=params.get("chunk_max", 80))
        else:
            self._segmenter = PerMessageBoundary(
                confidence=params.get("boundary_confidence", 0.7),
                buffer_min=params.get("buffer_min", 2),
                buffer_max=params.get("buffer_max", 25))
        self._merger = (EpisodeMerger(top_k=params.get("merge_top_k", 5),
                                      similarity=params.get("merge_similarity"),
                                      time_gap_hours=params.get("merge_time_gap_hours"))
                        if params["episode_merge"] == "llm" else None)
        self._integrator = {"append": AppendIntegrator,
                            "dedup": lambda: DedupIdReuseIntegrator(
                                threshold=params.get("dedup_threshold", 0.85)),
                            "llm3way": lambda: ThreeWayIntegrator(
                                top_k=params.get("integrate_top_k", 5),
                                tau=params.get("integrate_tau", 0.70)),
                            }[params["semantic_integration"]]()
        self._consolidator = (SemanticOfflineConsolidator(
            top_k=params.get("integrate_top_k", 5), tau=params.get("integrate_tau", 0.70))
            if params["consolidation"] == "semantic_offline" else None)
        self.fidelity = fidelity
        self.params = params  # stats/스탬프용
```

(SemanticOfflineConsolidator는 Task 11에서 구현 — 이 태스크에서는 임시로 이름만 import 가능하게 스텁 클래스 생성해도 되고, Task 10-11 순서를 지키면 스텁 불필요: consolidation 분기만 Task 11로 미룸. **권장: 이 태스크에서 `_consolidator = None` 고정 + mix 테스트의 consolidator 단언은 Task 11에서 활성화.**)

- [ ] **Step 4: 통과 + 회귀** — 기존 `nemori` 테스트 포함 전체 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(nemori): fidelity presets v1/v4/upstream with per-source thresholds"`

---

### Task 11: SemanticOfflineConsolidator (우리 기여 — 유예 통합)

**Files:**
- Modify: `src/agmem/organizers/nemori_stages.py`, `src/agmem/organizers/nemori.py` (`consolidate` 훅)
- Test: `tests/test_nemori_fidelity.py`

**Interfaces:**
- Consumes: Task 1 `ops_since`/`last_seq`, Task 2 커서 헬퍼, Task 9 ThreeWayIntegrator.
- Produces: `NemoriOrganizer.consolidate(ctx)` — 커서 이후 자기 actor의 semantic ADD를 ThreeWay 로직으로 재통합, `consolidated: True` 플래그로 자기 출력 재처리 방지, 마지막 op가 커서 전진.

- [ ] **Step 1: 실패 테스트**

```python
def test_semantic_offline_consolidate_merges_and_advances_cursor():
    llm = StubLLM({"distill": [
        {"decision": "merge", "target_indexes": [0], "statement": "User's dog is named Max"}]})
    org = NemoriOrganizer(fidelity="v1", consolidation="semantic_offline")
    mem = make_mem(org, llm)
    # 인라인 append로 유사 fact 2건 심기 (organizer actor로 로그에 남도록 _apply_ops 사용)
    for i, f in enumerate(["The user's dog is called Max", "User has a dog named Max"]):
        mem._apply_ops(AppendIntegrator().integrate(f, f"ep{i}", [f"m{i}"], mem._ctx),
                       actor="nemori")
    n = mem.consolidate()
    ops = mem.log.tail(20)
    merges = [o for o in ops if o.op == OpType.MERGE and o.target_type == "semantic"]
    assert len(merges) == 1 and merges[0].payload["consolidated"] is True
    cursor = [o for o in ops if o.target_type == "state"]
    assert cursor and cursor[-1].payload["seq"] > 0
    # 2번째 호출: 새 입력 없음 → 자기 출력(consolidated) 스킵, LLM 콜 없이 커서만 전진
    calls_before = len(llm.calls)
    mem.consolidate()
    assert len(llm.calls) == calls_before
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현** — nemori_stages.py:

```python
class SemanticOfflineConsolidator:
    """Deferred three-way consolidation over facts accumulated since the
    cursor — our addition (absent from both the paper and upstream; the
    LightMem/MOOM dual-phase pattern, spec §2.3). Own outputs carry
    consolidated=True and are skipped on later passes, so repeated calls
    converge instead of re-judging their own merges."""

    def __init__(self, top_k: int = 5, tau: float = 0.70):
        self._inner = ThreeWayIntegrator(top_k=top_k, tau=tau)

    def run(self, organizer, ctx) -> list[MemoryOp]:
        cursor = organizer.read_cursor(ctx)
        end = ctx.doc.last_seq()
        if end <= cursor:
            return []
        ops: list[MemoryOp] = []
        for _seq, op in ctx.doc.ops_since(cursor, target_type="semantic"):
            if op.op is not OpType.ADD or op.payload.get("consolidated"):
                continue
            current = ctx.doc.get_items([op.target_id], "semantic")
            if not current or current[0].get("invalid_at"):
                continue  # 이미 이 패스나 인라인에서 대체됨
            fact = str(current[0].get("content", ""))
            out = self._inner.integrate(fact, current[0].get("episode_id", ""),
                                        current[0].get("source_episode_ids", []), ctx)
            # ThreeWay가 자기 자신(top-1이 동일 항목)을 후보로 잡는 경우 제외:
            out = [o for o in out if o.target_id != op.target_id or o.op is OpType.INVALIDATE]
            produced_new = [o for o in out if o.op in (OpType.ADD, OpType.MERGE)]
            if not any(o.op is OpType.INVALIDATE for o in out):
                continue  # decision=new → 저장돼 있는 원본 유지, 아무것도 안 함
            for o in produced_new:
                o.payload["consolidated"] = True
            # 원본 fact도 통합문으로 대체되므로 무효화
            head_id = produced_new[0].target_id if produced_new else ""
            out.append(MemoryOp(op=OpType.INVALIDATE, target_type="semantic",
                                target_id=op.target_id,
                                payload={"reason": "consolidated", "superseded_by": head_id}))
            ops.extend(out)
        ops.append(organizer.cursor_op(end))
        return ops
```

주의(구현 시 정밀 조정 지점): ThreeWay의 vec.search가 처리 대상 fact 자신을 top-1로 되돌려주므로 후보에서 자기 id를 제외해야 한다 — `_inner.integrate`에 `exclude_ids` 파라미터를 추가하는 편이 위의 사후 필터보다 깔끔하면 그렇게 구현하고 Task 9 시그니처를 갱신할 것 (Interfaces 블록의 시그니처 변경을 커밋 메시지에 명시).

nemori.py:

```python
    def consolidate(self, ctx: OrganizerContext) -> list[MemoryOp]:
        if self._consolidator is None or ctx.llm is None:
            return []
        return self._consolidator.run(self, ctx)
```

Task 10에서 미룬 `_consolidator` 분기 활성화 + mix 프리셋 테스트 단언 복원.

- [ ] **Step 4: 통과 + 회귀** → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(nemori): SemanticOfflineConsolidator — cursor-resumed deferred three-way (our mixing)"`

---

### Task 12: MemoryOS `input="episodes"` + supersedes 처리

**Files:**
- Modify: `src/agmem/organizers/memoryos.py`
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Consumes: Task 2/4 이벤트 계약.
- Produces: `MemoryOSOrganizer(..., input="messages")`. episodes 모드: `consumes=("episodes",)`, on_message no-op, `on_memory_event`가 STM append(+용량 캐스케이드) 수행. 역인덱스 `self._page_sources: dict[page_id, set[unit_id]]`, `self._unit_pages: dict[unit_id, set[page_id]]` (인메모리 — `_heat`와 동일한 재시작 휘발성, 문서화).

- [ ] **Step 1: 실패 테스트**

```python
def test_memoryos_consumes_nemori_episodes():
    from agmem.organizers.memoryos import MemoryOSOrganizer
    from agmem.organizers.nemori import NemoriOrganizer
    llm = StubLLM({
        "extract": [{"boundary": True, "confidence": 0.9}] * 4,
        "distill": [
            {"title": "t", "narrative": "n", "timestamp": "2026-01-01"},  # 에피소드
            {"facts": []},                                                 # 증류
            {"groups": [{"topic": "g", "summary": "s", "message_indexes": [0]}]},  # MemoryOS 분절
        ],
    })
    mos = MemoryOSOrganizer(stm_capacity=1, input="episodes")
    mem = make_mem_multi([NemoriOrganizer(fidelity="v1", buffer_min=1), mos], llm)
    # (make_mem을 organizer 리스트 버전으로 일반화 — helpers에 추가)
    mem.add_message("hello", meta={"date": "2026-01-01"})
    mem.add_message("new topic", meta={"date": "2026-01-01"})  # 경계 → 에피소드 flush
    pages = [o for o in mem.log.tail(30)
             if o.target_type == "pages" and o.actor == "memoryos"]
    assert pages  # Nemori 에피소드가 MemoryOS page로 흘러들어감
    # 에피소드 원문이 아니라 Nemori 서사가 STM에 들어갔는지: page의 source가 episode id
    ep_ids = [o.target_id for o in mem.log.tail(30)
              if o.target_type == "episodes" and o.op == OpType.ADD]
    assert set(pages[0].payload["source_episode_ids"]) <= set(ep_ids)


def test_memoryos_retires_superseded_units():
    from agmem.organizers.memoryos import MemoryOSOrganizer
    mos = MemoryOSOrganizer(stm_capacity=1, input="episodes")
    mem = _mk(organizers=[mos])
    # page화 유도: LLM 없음 → mechanical segment (explicit degradation 경로)
    mem._propagate_events([MemoryOp(op=OpType.ADD, target_type="episodes", target_id="e1",
                                    payload={"id": "e1", "content": "ep one"})], actor="src")
    pages = [o for o in mem.log.tail(10) if o.target_type == "pages"]
    assert len(pages) == 1
    # e1을 흡수한 MERGE 도착 → page의 유일 소스가 superseded → page INVALIDATE
    mem._propagate_events([MemoryOp(op=OpType.MERGE, target_type="episodes", target_id="e2",
                                    payload={"id": "e2", "content": "merged",
                                             "supersedes": ["e1"]})], actor="src")
    inv = [o for o in mem.log.tail(10)
           if o.target_type == "pages" and o.op == OpType.INVALIDATE]
    assert len(inv) == 1 and inv[0].payload["reason"] == "sources_superseded"
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현** — memoryos.py:

```python
    def __init__(self, ..., input: str = "messages") -> None:
        ...
        self.input_mode = input
        if input == "episodes":
            self.consumes = ("episodes",)
        self._page_sources: dict[str, set[str]] = {}
        self._unit_pages: dict[str, set[str]] = {}

    def on_message(self, ep, ctx):
        if self.input_mode == "episodes":
            return []  # 이벤트로만 입력받음 (spec §3)
        ...기존...

    def on_memory_event(self, ev, ctx):
        if self.input_mode != "episodes":
            return []
        ops: list[MemoryOp] = []
        if ev.supersedes:
            ops.extend(self._retire(set(ev.supersedes)))
        content = str(ev.payload.get("content", ""))
        unit = Episode(content=content, role="episode", id=ev.target_id,
                       namespace=ctx.namespace,
                       meta={"date": ev.payload.get("timestamp", "")})
        if ev.op is OpType.UPDATE:
            if any(e.id == ev.target_id for e in self._stm):
                self._stm = [unit if e.id == ev.target_id else e for e in self._stm]
            return ops  # 이미 page화된 경우: 문서화된 staleness 허용 (spec §3)
        self._stm.append(unit)  # ADD / MERGE
        if len(self._stm) >= self.stm_capacity:
            batch, self._stm = self._stm, []
            ops.extend(self._evict_to_mtm(batch, ctx))
        return ops

    def _retire(self, superseded: set[str]) -> list[MemoryOp]:
        """흡수된 단위의 파생 상태 정리 — STM에서는 제거, page는 소스가 전부
        superseded일 때만 INVALIDATE (부분이면 유지; spec §3)."""
        ops: list[MemoryOp] = []
        self._stm = [e for e in self._stm if e.id not in superseded]
        for uid in superseded:
            for pid in self._unit_pages.pop(uid, set()):
                srcs = self._page_sources.get(pid)
                if srcs is None:
                    continue
                srcs.discard(uid)
                if not srcs:
                    self._page_sources.pop(pid, None)
                    self._heat.pop(pid, None)
                    ops.append(MemoryOp(op=OpType.INVALIDATE, target_type="pages",
                                        target_id=pid,
                                        payload={"reason": "sources_superseded"}))
        return ops
```

`_evict_to_mtm`/`_segment_add`의 page 생성·병합 지점마다 역인덱스 등록:

```python
            for e in members:
                self._unit_pages.setdefault(e.id, set()).add(seg_id)
            self._page_sources.setdefault(seg_id, set()).update(e.id for e in members)
```

`tests/helpers.py`에 `make_mem_multi(organizers, llm)` 추가 (make_mem의 리스트 버전).

- [ ] **Step 4: 통과 + 회귀** — memoryos 기존 테스트 포함 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(memoryos): input=episodes consumer + supersedes retirement (chained manager)"`

---

### Task 13: A-Mem `input="episodes"`

**Files:**
- Modify: `src/agmem/organizers/amem.py`
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Produces: `AMemOrganizer(top_k=5, input="messages")`. episodes 모드: `consumes=("episodes",)`, on_message no-op, on_memory_event가 에피소드를 note 파이프라인(`_ingest`)에 태움. `self._episode_notes: dict[episode_id, note_id]`; supersedes 수신 시 해당 note INVALIDATE.

- [ ] **Step 1: 실패 테스트**

```python
def test_amem_consumes_episodes_and_retires_notes():
    from agmem.organizers.amem import AMemOrganizer
    org = AMemOrganizer(input="episodes")
    mem = _mk(organizers=[org])  # LLM 없음 → bare note 경로 (explicit degradation)
    mem._propagate_events([MemoryOp(op=OpType.ADD, target_type="episodes", target_id="e1",
                                    payload={"id": "e1", "content": "ep narrative",
                                             "timestamp": "2026-01-01"})], actor="src")
    notes = [o for o in mem.log.tail(10) if o.target_type == "notes"]
    assert len(notes) == 1 and notes[0].payload["source_episode_ids"] == ["e1"]
    mem._propagate_events([MemoryOp(op=OpType.MERGE, target_type="episodes", target_id="e2",
                                    payload={"id": "e2", "content": "merged",
                                             "supersedes": ["e1"]})], actor="src")
    inv = [o for o in mem.log.tail(10)
           if o.target_type == "notes" and o.op == OpType.INVALIDATE]
    assert len(inv) == 1 and inv[0].target_id == notes[0].target_id
```

- [ ] **Step 2: 실패 확인** → FAIL
- [ ] **Step 3: 구현** — on_message 본문을 `_ingest(self, ep, ctx) -> list[MemoryOp]`로 추출(동작 무변경), 그리고:

```python
    def __init__(self, top_k: int = 5, input: str = "messages") -> None:
        self.top_k = top_k
        self.input_mode = input
        if input == "episodes":
            self.consumes = ("episodes",)
        self._episode_notes: dict[str, str] = {}

    def on_message(self, ep, ctx):
        if self.input_mode == "episodes":
            return []
        return self._ingest(ep, ctx)

    def on_memory_event(self, ev, ctx):
        if self.input_mode != "episodes":
            return []
        ops: list[MemoryOp] = []
        for sid in ev.supersedes:
            nid = self._episode_notes.pop(sid, None)
            if nid:
                ops.append(MemoryOp(op=OpType.INVALIDATE, target_type="notes",
                                    target_id=nid,
                                    payload={"reason": "episode_superseded"}))
        if ev.op is OpType.UPDATE:
            return ops  # note 재작성은 하지 않음 — 문서화된 staleness (spec §3)
        unit = Episode(content=str(ev.payload.get("content", "")), role="episode",
                       id=ev.target_id, namespace=ctx.namespace,
                       meta={"date": ev.payload.get("timestamp", "")})
        note_ops = self._ingest(unit, ctx)
        if note_ops:
            self._episode_notes[ev.target_id] = note_ops[0].target_id  # 첫 op = ADD note
        return ops + note_ops
```

- [ ] **Step 4: 통과 + 회귀** — amem 기존 테스트 포함 → PASS
- [ ] **Step 5: 커밋** — `git commit -m "feat(amem): input=episodes consumer — second chained manager"`

---

### Task 14: 실험 config 6종 + budget 위상 태그 + 문서 갱신 + 전체 회귀

**Files:**
- Modify: `scripts/exp_locomo_conv0.py`, `src/agmem/llm/structured.py`, `src/agmem/llm/client.py`, `docs/11-nemori-study.md`, `docs/04-architecture.md`, `docs/superpowers/specs/2026-07-18-nemori-lifecycle-redesign-design.md`
- Test: 전체 스위트 + ruff

- [ ] **Step 1: budget 위상 태그** — `LLMClient.chat(role, messages, budget_key=None, **overrides)`로 확장해 `self.budget.record(budget_key or role, ...)` (record 호출 2곳: client.py :67, :71). `StructuredCaller.call(..., phase: str | None = None)` 추가 → `self.client.chat(role, messages, budget_key=f"{role}/{phase}" if phase else None, **overrides)`. Nemori 스테이지 콜사이트에 phase 지정: 경계/분할="segment", 서사="narrate", merge 2콜="merge", predict/calibrate/direct="distill", 3way="integrate", offline="consolidate". StubLLM은 이미 `**kw` 수용이라 테스트 무영향. `stats()`의 llm summary에 `role/phase` 키가 나타나는 것 확인.
- [ ] **Step 2: config 6종** — exp 스크립트 `known` dict에 추가. organizers 원소로 **0-인자 팩토리 callable**을 허용하도록 run() 초입에서 해석 (다중 run 간 organizer 상태 격리 — 인스턴스 재사용은 버퍼 오염):

```python
    organizers = [o() if callable(o) and not isinstance(o, str) else o for o in organizers]
```

```python
        "nemori_v4": (
            [lambda: NemoriOrganizer(fidelity="v4")],
            ("episodes", "semantic"), {"episodes": 10, "semantic": 20},
            False, NEMORI_TEMPS, NEMORI_STORE),
        "nemori_upstream": (
            [lambda: NemoriOrganizer(fidelity="upstream")],
            ("episodes", "semantic"), {"episodes": 10, "semantic": 20},
            False, NEMORI_TEMPS, NEMORI_STORE),
        "nemori_mix": (  # batch+merge는 v4, 통합은 유예로 — 인라인 vs 유예 ablation 축
            [lambda: NemoriOrganizer(fidelity="v4", semantic_integration="append",
                                     consolidation="semantic_offline")],
            ("episodes", "semantic"), {"episodes": 10, "semantic": 20},
            False, NEMORI_TEMPS, NEMORI_STORE),
        "nemori_memoryos": (
            [lambda: NemoriOrganizer(fidelity="v1"),
             lambda: MemoryOSOrganizer(input="episodes")],
            ("episodes", "semantic", "pages"), {"episodes": 10, "semantic": 20, "pages": 10},
            False, NEMORI_TEMPS, None),
        "nemori_amem": (
            [lambda: NemoriOrganizer(fidelity="v1"), lambda: AMemOrganizer(input="episodes")],
            ("episodes", "semantic", "notes"), {"episodes": 10, "semantic": 20, "notes": 10},
            False, NEMORI_TEMPS, None),
```

`nemori_mix`는 인제스트 종료 후 `mem.consolidate()` 호출이 필요 — run()의 flush 지점 직후에 `if any(getattr(o, "_consolidator", None) for o in mem.organizers): mem.consolidate()` 추가. 결과 스탬프에 `fidelity`/`params` 기록 (NemoriOrganizer.params 노출 사용).
- [ ] **Step 3: 문서 갱신** — docs/11 §5 편차 표: 병합/배치/통합이 fidelity 스위치로 해소됨을 반영(잔여 편차만 남김). docs/04 §3 표에 on_memory_event/consolidate 열 추가(Nemori/MemoryOS/A-Mem 행). 스펙 §2.2 미해결 표기 갱신: ">1h 갭 병합 금지는 upstream MERGE_DECISION 프롬프트 소재로 확정(v4 논문엔 없음) — upstream 프리셋만 적용" + §1.2에 "supersedes는 MERGE op payload로 명시(배치 추론 아님)" 반영.
- [ ] **Step 4: 전체 회귀** — `.venv/bin/python -m pytest tests/ -q` 전체 PASS, `ruff check src tests`, `ruff format --check src tests`.
- [ ] **Step 5: 커밋** — `git commit -m "feat: LoCoMo fidelity/chained configs, budget phase tags, docs sync"`
