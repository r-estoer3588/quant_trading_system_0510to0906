"""Core logic layer.

Provides stable import paths like ``core.system1`` and helper utilities
that are reused by the Streamlit UI as well as offline batch scripts.
"""

from .final_allocation import finalize_allocation

__all__ = ["finalize_allocation"]
