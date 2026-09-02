"""The explorer read path: the memory as files, and an agent that greps them.

``export_workspace`` writes the store out under a root (``workspace.py``);
``Explorer`` searches and reads that root with a model in the loop
(``explorer.py``). ``AgenticMemory.research`` joins the two.
"""

from agmem.explore.explorer import Explorer, ResearchResult
from agmem.explore.workspace import WorkspaceStats, export_workspace

__all__ = ["Explorer", "ResearchResult", "WorkspaceStats", "export_workspace"]
