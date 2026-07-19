"""Neo4j access — read-only, always (D-8).

The graph keeps a single writer (vishwamitra's ASSEMBLE). Lakshmana only hydrates claim
and explanation text, and every Lakshmana write lands in Firestore instead. The RBAC
user enforces this server-side; :class:`Neo4jReader` enforces it client-side too, so a
future gate cannot casually acquire a write session.

Credentials come from Secret Manager in production (``gatekeeper-neo4j-ro``) and from
``GATEKEEPER_NEO4J_*`` locally against the Docker container.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase, Session
from neo4j.api import READ_ACCESS

from gatekeeper.config import Config
from gatekeeper.errors import Neo4jUnavailableError
from gatekeeper.logging import get_logger

__all__ = ["Neo4jReader", "neo4j_driver"]

log = get_logger(__name__)


def neo4j_driver(config: Config) -> Driver:
    """Build a driver from config. Does not connect until first use."""
    return GraphDatabase.driver(
        config.get_str("gatekeeper.neo4j.uri"),
        auth=(
            config.get_str("gatekeeper.neo4j.user"),
            config.get_str("gatekeeper.neo4j.password"),
        ),
    )


class Neo4jReader:
    """A read-only handle on the graph."""

    def __init__(self, driver: Driver, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A session pinned to READ access — write queries fail fast, by construction."""
        session = self._driver.session(database=self._database, default_access_mode=READ_ACCESS)
        try:
            yield session
        finally:
            session.close()

    def run(self, cypher: str, **parameters: Any) -> list[dict[str, Any]]:
        """Execute a read query and materialize the rows.

        Raises:
            Neo4jUnavailableError: wrapping any driver failure, so callers see a
                ``GK_E_NEO4J_UNAVAILABLE`` gate failure rather than a raw driver error.
        """
        try:
            with self.session() as session:
                return [record.data() for record in session.run(cypher, **parameters)]
        except Exception as exc:  # every driver error maps to one GK_E_ code
            raise Neo4jUnavailableError(f"neo4j read failed: {exc}") from exc

    def verify_connectivity(self) -> None:
        """Fail fast at worker startup rather than midway through a gate.

        Raises:
            Neo4jUnavailableError: if the server is unreachable or auth is wrong.
        """
        try:
            self._driver.verify_connectivity()
        except Exception as exc:
            raise Neo4jUnavailableError(f"neo4j connectivity check failed: {exc}") from exc

    def close(self) -> None:
        self._driver.close()
