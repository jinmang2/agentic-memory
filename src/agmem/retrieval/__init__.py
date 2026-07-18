"""Public entry point for the recall -> fuse -> rerank -> hydrate pipeline;
see `agmem.retrieval.pipeline` for the stage-by-stage contract."""

from agmem.retrieval.pipeline import RetrievalPipeline

__all__ = ["RetrievalPipeline"]
