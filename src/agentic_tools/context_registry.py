"""
context_registry.py

Simple in-memory registry for AnalysisContext objects.
"""

from agentic_tools.context import AnalysisContext


_CONTEXTS = {}


def get_context(source):

    if source not in _CONTEXTS:

        _CONTEXTS[source] = AnalysisContext(
            source
        )

    return _CONTEXTS[source]


def clear_context(source=None):

    if source is None:
        _CONTEXTS.clear()
        return

    if source in _CONTEXTS:
        del _CONTEXTS[source]