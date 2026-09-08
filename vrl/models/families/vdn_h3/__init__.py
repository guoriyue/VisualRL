"""VDN-H3 (VideoDeltaNet on MiniMax-H3): hybrid window-softmax + linear attention."""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py); an empty root keeps config
# discovery from importing the torch-backed runtime or the vendored package.
__all__: list[str] = []
