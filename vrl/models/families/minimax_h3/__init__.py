"""MiniMax-H3 (Hailuo 3.0) joint video+audio generator family (T2V policy, flow-matching GRPO RL)."""

from __future__ import annotations

# Deliberately exports nothing. The family registry dispatches by dotted
# submodule path (vrl/models/families/registry.py); keeping this module empty
# is also what stops config discovery from pulling the torch-backed runtime.
__all__: list[str] = []
