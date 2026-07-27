"""MemMachine (arXiv:2604.04853): whole episodes indexed by mechanical derivatives.

The deployed-code lineage — ``MEMMACHINE_PRESETS`` says which backend a run
reproduces and why the paper's tier names do not appear here."""

from agmem.organizers.memmachine.organizer import MEMMACHINE_PRESETS, MemMachineOrganizer

__all__ = ["MEMMACHINE_PRESETS", "MemMachineOrganizer"]
