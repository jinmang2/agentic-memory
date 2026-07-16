from agmem.organizers.amem import AMemOrganizer
from agmem.organizers.base import Organizer, OrganizerContext
from agmem.organizers.passthrough import PassthroughOrganizer
from agmem.organizers.reasoning_bank import ReasoningBankOrganizer

ORGANIZERS: dict[str, type[Organizer]] = {
    "passthrough": PassthroughOrganizer,
    "reasoning_bank": ReasoningBankOrganizer,
    "amem": AMemOrganizer,
}

__all__ = ["Organizer", "OrganizerContext", "PassthroughOrganizer",
           "ReasoningBankOrganizer", "AMemOrganizer", "ORGANIZERS"]
