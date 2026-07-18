"""Public facade for host-capability detection and adapter resolution.

Re-exports `detect`/`HostCapabilities` (probe the host once, cache to disk)
and `Requires`/`resolve` (declare and match adapter requirements) — see
`agmem.capabilities.detect` and `.resolver` for the contracts.
"""

from agmem.capabilities.detect import HostCapabilities, detect
from agmem.capabilities.requires import Requires
from agmem.capabilities.resolver import CapabilityWarning, resolve

__all__ = ["HostCapabilities", "Requires", "CapabilityWarning", "detect", "resolve"]
