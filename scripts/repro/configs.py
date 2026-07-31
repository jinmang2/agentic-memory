"""Named organizer configs for the repro harness — the runner-facing policy table.
Each entry is (factory, memory_types): the factory builds FRESH organizer instances
per conversation (never share organizer state across convs), and memory_types is
what the eval read path retrieves. Arm variants (e.g. nemori_merge085) live here so
an experiment arm is a --config name, reproducible from the CLI line alone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.nemori import NemoriOrganizer


@dataclass(frozen=True)
class RunnerConfig:
    name: str
    factory: Callable[[], list]
    memory_types: tuple[str, ...]
    # run_ready=False = constructible for counting/tests but NOT wired for a real run yet
    # (exp_locomo_conv0's audited nemori entries also carry per-type k + NEMORI_TEMPS
    # (:350) + NEMORI_STORE (:360) that stage 1 does not thread — Track 1 reconciles
    # all six tuple fields and flips these to True).
    run_ready: bool = True


CONFIGS: dict[str, RunnerConfig] = {
    c.name: c
    for c in (
        RunnerConfig("amem", lambda: [AMemOrganizer()], ("notes",)),
        # factory + memory_types verbatim from exp_locomo_conv0.py:386-393:
        RunnerConfig(
            "nemori_upstream",
            lambda: [NemoriOrganizer(fidelity="upstream")],
            ("episodes", "semantic"),
            run_ready=False,
        ),
        RunnerConfig(
            "nemori_merge085",
            lambda: [NemoriOrganizer(fidelity="upstream", merge_similarity=0.85)],
            ("episodes", "semantic"),
            run_ready=False,
        ),
    )
}


def get_config(name: str) -> RunnerConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(f"unknown runner config {name!r} (known: {sorted(CONFIGS)})") from None
