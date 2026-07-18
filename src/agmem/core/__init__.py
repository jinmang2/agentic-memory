"""Domain-agnostic building blocks shared by every organizer and store.

`types` holds the episode/derived-item dataclasses (Episode, Note, Fact, ...);
`ops` holds the `MemoryOp` op type and the append-only evolution log contract
that all seven methodologies replay state through.
"""
