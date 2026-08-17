"""Every store implementation must carry the whole surface its Protocol declares.

The `DocStore` and `VectorStore` protocols in `agmem.stores.base` are the contract the rest of the
system codes against. Python does not enforce them — a Protocol is structural, and
`@runtime_checkable` only compares method *names* at `isinstance` time, which nothing here calls — so
a backend can be missing a method for months and only fail when some code path finally reaches it.

**This test exists because that happened.** `list_episodes` lived on `SqliteDocStore` and was used by
three call sites (`hooks/recall.py`, `AgenticMemory`'s episode-mention frontier, and the reproduction
harness's snapshot writer) while `PostgresDocStore` never grew it and the protocol never demanded it.
The snapshot writer guards that call with `getattr`, so instead of raising, every run configured onto
Postgres — the two Nemori arms, via `configs.NEMORI_STORE` — wrote a memory snapshot with no episodic
rows at all. It looked like a complete artifact that simply had no transcript in it, and it was found
much later by a demo trying to read one.

The check is deliberately STATIC — `hasattr` on classes, no instances, no servers, no network. A live
conformance test would skip on any machine without Postgres, Qdrant or Neo4j running, which is every
machine this suite must pass on, and skipping is precisely how the gap survived.

**The rosters below are explicit, and a completeness guard keeps them honest.** Deriving them from
class names looked tidy and was wrong: `SqliteVecStore` does not end in `VectorStore`, so a name
heuristic silently exempted it — the same shape of bug as the one being tested for. So the classes
are listed, and a separate test fails if any store module defines a class no roster mentions.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

from agmem.stores import base

# module name -> class name, per protocol. Listed rather than inferred; see the module docstring.
DOC_STORE_IMPLS = (
    ("sqlite_doc", "SqliteDocStore"),
    ("postgres_doc", "PostgresDocStore"),
)
VECTOR_STORE_IMPLS = (
    ("numpy_vec", "NumpyVectorStore"),
    ("sqlite_vec", "SqliteVecStore"),
    ("lance_vec", "LanceDBVectorStore"),
    ("chroma_vec", "ChromaVectorStore"),
    ("qdrant_vec", "QdrantVectorStore"),
)

# Graph backends exist but `base.py` declares NO GraphStore protocol, so there is no surface to check
# them against. Listed here so the completeness guard does not flag them, and recorded as the reason:
# the missing protocol is a real gap, just not one this test can close by asserting.
UNPROTOCOLED_IMPLS = (
    ("sqlite_graph", "SqliteGraphStore"),
    ("kuzu_graph", "KuzuGraphStore"),
    ("neo4j_graph", "Neo4jGraphStore"),
)

ROSTERS = {"DocStore": DOC_STORE_IMPLS, "VectorStore": VECTOR_STORE_IMPLS}


def _protocol_methods(protocol: type) -> list[str]:
    """Public method names a Protocol declares, excluding typing's own machinery."""
    return sorted(
        name
        for name, value in vars(protocol).items()
        if not name.startswith("_") and inspect.isfunction(value)
    )


def _load(module_name: str, class_name: str) -> type:
    """Import one backend by module path.

    Imported directly rather than through `agmem.stores.__init__` so an optional-dependency failure
    surfaces as an error here instead of quietly shrinking the roster — a test that passes by
    checking fewer things is the failure mode this file was written against.
    """
    module = importlib.import_module(f"agmem.stores.{module_name}")
    return getattr(module, class_name)


@pytest.mark.parametrize("protocol_name", sorted(ROSTERS))
def test_every_implementation_covers_its_protocol(protocol_name):
    """No backend may be missing a method its protocol declares.

    Reported per class with every gap listed, because the useful message is "this backend is missing
    these three", not whichever name happens to fail first.
    """
    protocol = getattr(base, protocol_name)
    required = _protocol_methods(protocol)
    assert required, f"{protocol_name} declares no methods — the test lost its subject"

    missing = {}
    for module_name, class_name in ROSTERS[protocol_name]:
        implementation = _load(module_name, class_name)
        gaps = [name for name in required if not hasattr(implementation, name)]
        if gaps:
            missing[class_name] = gaps

    detail = "\n".join(f"  {name}: missing {gaps}" for name, gaps in sorted(missing.items()))
    assert not missing, f"{protocol_name} implementations with gaps:\n{detail}"


def test_no_store_class_escapes_the_rosters():
    """A new backend must be classified before this suite goes green again.

    Without this, adding a store and forgetting to list it would leave every conformance assertion
    above passing while the new class is checked against nothing.
    """
    rostered = {
        class_name for roster in (*ROSTERS.values(), UNPROTOCOLED_IMPLS) for _, class_name in roster
    }
    found = set()
    for info in pkgutil.iter_modules(base.__spec__.submodule_search_locations or []):
        if info.name == "base":
            continue
        module = importlib.import_module(f"agmem.stores.{info.name}")
        for name, obj in vars(module).items():
            if isinstance(obj, type) and obj.__module__ == module.__name__ and "Store" in name:
                found.add(name)

    unclassified = sorted(found - rostered)
    assert not unclassified, (
        f"store classes not in any roster in this file: {unclassified}. Add each to "
        f"DOC_STORE_IMPLS / VECTOR_STORE_IMPLS, or to UNPROTOCOLED_IMPLS with the reason."
    )


def test_doc_store_protocol_still_requires_list_episodes():
    """Pin the specific method whose absence caused the loss this file documents.

    The parametrized test would keep passing if `list_episodes` were dropped from the protocol —
    nothing would then be required of anyone. This asserts the requirement itself survives.
    """
    assert "list_episodes" in _protocol_methods(base.DocStore)


def test_the_conformance_check_fails_on_a_planted_gap():
    """A class missing declared methods must be reported, not tolerated.

    Guards against a bug in `_protocol_methods` returning an empty list, which would make every
    assertion above vacuous.
    """
    required = _protocol_methods(base.DocStore)

    class HalfDocStore:
        pass

    gaps = [name for name in required if not hasattr(HalfDocStore, name)]
    assert gaps == required, "a class with no methods must be reported as missing all of them"
