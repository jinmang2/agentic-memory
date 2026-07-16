# Capability System & Profile 스펙

> 원칙: **방법론은 버리지 않는다.** 환경 부적합 구성요소도 전부 구현하고,
> capability detection + 프로파일로 코드 레벨에서 선택/강등한다.

## 1. Capability Detection

시작 시 1회 감지 후 캐시 (`~/.agentic_memory/capabilities.json`, TTL 24h, `--redetect`로 갱신):

```python
@dataclass
class HostCapabilities:
    ram_gb: float              # psutil.virtual_memory().total
    vram_gb: float | None      # pynvml, 없으면 None
    cpu_cores: int
    gpu_name: str | None
    services: dict[str, bool]  # {"neo4j": False, "qdrant": False, "ollama": True, ...}
                               #   → TCP port probe + docker ps 검사
    llm_endpoints: list[EndpointInfo]  # OpenAI-compatible endpoint 헬스체크 결과
```

## 2. Backend Requirement 선언

모든 어댑터는 클래스 레벨로 요구 스펙을 선언한다:

```python
class Neo4jGraphStore(GraphStore):
    requires = Requires(ram_gb=4.0, services=["neo4j"])

class KuzuGraphStore(GraphStore):
    requires = Requires(ram_gb=0.5)          # embedded

class CrossEncoderReranker(Reranker):
    requires = Requires(vram_gb=1.5)          # GPU 권장, CPU fallback 허용(slow=True)

class NoopReranker(Reranker):
    requires = Requires()                     # 항상 가능
```

## 3. Registry + Resolver

```python
REGISTRY = {
    "graph_store":  [Neo4jGraphStore, FalkorDBGraphStore, KuzuGraphStore, NetworkXGraphStore],
    "vector_store": [QdrantStore, ChromaStore, LanceDBStore, SqliteVecStore],
    "reranker":     [CrossEncoderReranker, LLMReranker, RRFReranker, NoopReranker],
    "embedder":     [APIEmbedder, Qwen3Embedder06B, BgeM3Embedder, MiniLMEmbedder],
    ...
}
```

- 각 slot의 리스트는 **선호 순서** (앞 = 고성능/무거움).
- Resolver는 `config에 명시된 것 > profile 기본값 > capability 매칭 첫 후보` 순으로 선택.
- 요구 미충족인데 config로 강제 지정된 경우: 에러가 아니라
  `CapabilityWarning: Neo4jGraphStore requires service 'neo4j' (not detected). Falling back to KuzuGraphStore.`
  로그 후 강등. `--strict` 모드에서만 에러.

## 4. Profiles

| slot | `lite` (현 PC 기본) | `standard` | `full` (서버/클라우드) |
|---|---|---|---|
| vector_store | SqliteVecStore | LanceDBStore | QdrantStore |
| graph_store | KuzuGraphStore | KuzuGraphStore | Neo4jGraphStore |
| embedder | MiniLM/bge-small (CPU) | Qwen3-Embedding-0.6B (GPU) | API or 대형 embedder |
| reranker | RRFReranker | LLMReranker | CrossEncoderReranker |
| llm.extract | Qwen3-0.6B (로컬) | Qwen3-4B-AWQ | API (gpt-4o-mini급) |
| llm.judge | API | API | API or 로컬 70B |
| write path | async queue | async queue | async queue + workers |

- 프로파일은 시작점일 뿐이며 모든 slot은 TOML config로 오버라이드 가능.
- 동일 실험 코드가 profile 스위칭만으로 lite↔full 재현 가능해야 함 (실험 결과에 사용 profile 기록 필수).

## 5. Config 파일 예시

```toml
[profile]
name = "lite"            # lite | standard | full

[llm.extract]
endpoint = "http://localhost:8000/v1"   # vLLM/llama.cpp/Ollama
model = "qwen3-0.6b"

[llm.judge]
endpoint = "https://api.openai.com/v1"
model = "gpt-4o-2024-08-06"             # LongMemEval judge pin

[override]
reranker = "LLMReranker"                # profile 기본값 무시하고 강제
```
