"""OWUI orchestrator: a standalone OpenAI-compatible service that replaces the
router_fn filter. OWUI calls it as a model connection; it classifies turn depth,
drafts with the best model per task, verifies grounding, and streams the result.
"""
__version__ = "0.1.0"
