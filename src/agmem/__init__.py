"""agmem — unified agentic memory library.

Seven memory methodologies (A-Mem, MemoryOS, Nemori, Zep-graph, ACE,
ReasoningBank, G-Memory) behind one capability-gated API.
"""

from agmem.core.ops import MemoryOp, OpType
from agmem.core.types import Episode, MemoryBundle
from agmem.memory import AgenticMemory

__version__ = "0.1.0"

__all__ = [
    "AgenticMemory",
    "Episode",
    "MemoryBundle",
    "MemoryOp",
    "OpType",
    "__version__",
]
