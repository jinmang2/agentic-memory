from agmem.organizers.ace import ACEOrganizer
from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.gmemory import GMemoryOrganizer
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
    "memoryos": MemoryOSOrganizer,
    "ace": ACEOrganizer,
    "zep_graph": ZepGraphOrganizer,
    "gmemory": GMemoryOrganizer,
}

__all__ = ["Organizer", "OrganizerContext", "ORGANIZERS",
           "PassthroughOrganizer", "ReasoningBankOrganizer", "AMemOrganizer",
           "NemoriOrganizer", "MemoryOSOrganizer", "ACEOrganizer",
           "ZepGraphOrganizer", "GMemoryOrganizer"]
