"""The worker: a framework-free Cloud Run Job that executes one gate.

Deliberately re-exports nothing. ``main`` is both this package's entry-point function
and its module name, and re-exporting the function here shadows the module — which
silently breaks ``from gatekeeper.worker import main`` for anyone expecting the module.
Import from :mod:`gatekeeper.worker.main` directly.
"""
