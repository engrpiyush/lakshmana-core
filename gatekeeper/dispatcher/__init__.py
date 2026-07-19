"""The dispatcher: a thin HTTP service that claims gates and launches worker jobs."""

from gatekeeper.dispatcher.app import create_app

__all__ = ["create_app"]
