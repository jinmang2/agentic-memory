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
| *(none yet)* | *discard* / *retrieve* suppression | Memory Worth (2604.12007, candidate) |

There is deliberately no ``base.py`` policy Protocol yet: with one implemented
member, any shared interface would be guessed from a single example. The second
member (SAGE is also a write-admission gate for an A-Mem host, so it lands in
``admission``) is what should force the abstraction out.

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

__all__ = [
    "AdmissionDecision",
    "AdmissionFeatures",
    "AdmissionGate",
    "AdmissionStats",
    "PAPER_THRESHOLD",
    "PAPER_WEIGHTS",
    "TypePriorClassifier",
]
