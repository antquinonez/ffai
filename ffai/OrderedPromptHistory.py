# Compatibility shim - module moved to ffai/core/history/ordered.py
from .core.history.ordered import Interaction, OrderedPromptHistory

__all__ = ["Interaction", "OrderedPromptHistory"]
