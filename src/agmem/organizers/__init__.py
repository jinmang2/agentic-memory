"""Organizer registry: maps config/CLI organizer names to their classes.

Consumed by ``AgenticMemory``'s ``organizers`` constructor arg and mcp/server.py's
``--organizers`` flag, both of which accept these string keys interchangeably with
pre-built ``Organizer`` instances."""

from agmem.organizers.ace import ACEOrganizer
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.base import MemoryEvent, Organizer, OrganizerContext, overrides
from agmem.organizers.experience import ExperienceOrganizer
from agmem.organizers.gmemory import GMemoryOrganizer
from agmem.organizers.mem0 import Mem0Organizer
from agmem.organizers.memmachine import MemMachineOrganizer, MemMachineProfileOrganizer
from agmem.organizers.memoryos import MemoryOSOrganizer
from agmem.organizers.nemori import NemoriOrganizer
from agmem.organizers.passthrough import PassthroughOrganizer
from agmem.organizers.reasoning_bank import ReasoningBankOrganizer
from agmem.organizers.zep_graph import ZepGraphOrganizer

ORGANIZERS: dict[str, type[Organizer]] = {
    "passthrough": PassthroughOrganizer,
    "reasoning_bank": ReasoningBankOrganizer,
    "amem": AMemOrganizer,
    "nemori": NemoriOrganizer,
    "mem0": Mem0Organizer,
    "memoryos": MemoryOSOrganizer,
    "memmachine": MemMachineOrganizer,
    "memmachine_profile": MemMachineProfileOrganizer,
    "ace": ACEOrganizer,
    "zep_graph": ZepGraphOrganizer,
    "gmemory": GMemoryOrganizer,
    "experience": ExperienceOrganizer,
}

__all__ = [
    "ORGANIZERS",
    "ACEOrganizer",
    "AMemOrganizer",
    "ExperienceOrganizer",
    "GMemoryOrganizer",
    "Mem0Organizer",
    "MemMachineOrganizer",
    "MemMachineProfileOrganizer",
    "MemoryEvent",
    "MemoryOSOrganizer",
    "NemoriOrganizer",
    "Organizer",
    "OrganizerContext",
    "PassthroughOrganizer",
    "ReasoningBankOrganizer",
    "ZepGraphOrganizer",
    "overrides",
]
