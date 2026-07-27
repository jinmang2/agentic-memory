"""Control policies — cross-cutting rules that govern *when* a methodology acts.

The split between this package and ``organizers/`` follows the literature's own
decomposition rather than our convenience. The agentic-memory survey
arXiv:2603.07670 models a memory system as write-manage-read, with a read
operator R(M, x) and an update operator U(M, x, a, o, r) over the operation set
*store / retrieve / update / summarize / discard*, and then separates two levels:

- **mechanisms** (§4) — what the memory system *is*: context compression,
  retrieval stores, reflection, hierarchical memory, learned control. These own a
  memory representation and a retrieval path. In this codebase a mechanism is an
  ``Organizer`` (``organizers/``).
- **control policies** (§3.3) — heuristic, prompted, or learned rules that govern
  *how the core operations execute*. The survey calls this "a cross-cutting
  dimension orthogonal to mechanism choice": the same store can be driven by a
  different policy. That is this package.

The operational test for belonging here: **a policy declares no memory type and
emits no ``MemoryOp``.** It only decides whether, or how, its host's operation
fires. Anything that owns stored state and a way to read it back is a mechanism
and belongs in ``organizers/`` — that is the test that reclassified RecMem
(arXiv:2605.16045) and GRAVITY (arXiv:2605.01688) out of this package while they
were still only survey entries; both turned out to own their memory tiers.

Members, keyed by the survey operation they govern:

| module | operation | papers |
|---|---|---|
| ``admission`` | *store* | A-MAC (2603.04549, implemented), SAGE (2605.30711, candidate) |
| ``retrieval`` | *retrieve* | MemMachine's Retrieval Agent (2604.04853, implemented) |
| *(none yet)* | *discard* / *retrieve* suppression | Memory Worth (2604.12007, candidate) |
| *(none yet)* | *store/update/discard*, learned | Mem-alpha (2509.25911, candidate) |

Mem-alpha is worth singling out because it validates the split from the other
direction: its RL trains a policy over ``memory_insert``/``update``/``delete``
calls and the paper states its "memory architecture is modular and decoupled from
the reinforcement learning framework", with retriever and generator frozen. The
orthogonality is the papers' own, not our imposition. Beyond the operation, a
policy is also classified by *granularity* (LightMem gates tokens, A-MAC gates
turns) and by *kind* (heuristic / prompted / learned, survey §3.3).

A policy is attached to a mechanism by a wrapper, never by a constructor
argument on the mechanism: ``organizers/gated.py::AdmissionGated`` applies any
admission gate to any message-driven organizer, which is what makes the
cross-cutting claim true rather than aspirational. The read side needs no such
adapter module — its seam is a bound ``search`` callable
(``retrieval.QueryContext``), so a strategy reaches any mechanism without either
side importing the other. Same rule, cheaper: the write side had to wrap because
it intercepts a hook the mechanism owns, while a read policy only decides which
searches to run. **No mechanism imports this
package** — inside ``organizers/`` only that one adapter module does, and the
dependency the other way is limited to ``organizers.base.OrganizerContext``. See that module for the verified applicability limits — task-driven
organizers (ACE, G-Memory, ReasoningBank) declare no ``on_message`` and so are
outside *admission*'s reach specifically, though not outside ``policies/``.

There is deliberately no ``base.py`` policy Protocol yet, and the second module
did not change that: ``admission`` gates one candidate at write time and
``retrieval`` plans N searches at read time, so their only common shape would be
the word "policy". A shared base should be forced out by a second member of the
SAME operation (SAGE is another write-admission gate for an A-Mem host, so it
lands inside ``admission``), not by two operations that merely share a package.

Read-path *post-steps* are a different thing and live in ``retrieval/steps.py``:
those are keyed on memory type and belong to whichever mechanism produced the
items (A-Mem's link expansion, Nemori's source attachment), so they are mechanism
parts, not cross-cutting policies. A read-path *policy* — Memory Worth
suppressing what a host retrieved regardless of who produced it — would live
here instead.
"""

from agmem.policies.admission import (
    PAPER_THRESHOLD,
    PAPER_WEIGHTS,
    AdmissionDecision,
    AdmissionFeatures,
    AdmissionGate,
    AdmissionStats,
    TypePriorClassifier,
)
from agmem.policies.retrieval import (
    STRATEGIES,
    ChainOfQuery,
    DirectRetrieval,
    QueryContext,
    QueryResult,
    QueryStrategy,
    SplitQuery,
    ToolSelect,
)

__all__ = [
    "AdmissionDecision",
    "AdmissionFeatures",
    "AdmissionGate",
    "AdmissionStats",
    "ChainOfQuery",
    "DirectRetrieval",
    "PAPER_THRESHOLD",
    "PAPER_WEIGHTS",
    "QueryContext",
    "QueryResult",
    "QueryStrategy",
    "SplitQuery",
    "STRATEGIES",
    "ToolSelect",
    "TypePriorClassifier",
]
