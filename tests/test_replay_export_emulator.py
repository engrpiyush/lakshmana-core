"""The corpus exporter against a real Firestore (VA-97).

The transformation logic is covered in ``test_replay_corpus.py``; what is left, and what
only a real emulator can prove, is that the query path reads documents back in the shape
the exporter expects.

Scope note: this seeds its own throwaway ``stage3_edges_test_*`` collection, exactly like
every other emulator test in this suite. It is **not** the backup restore — that goes into
a separate emulator instance the operator starts, never the shared dev one on 8082, which
holds live development state.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from google.cloud import firestore

from gatekeeper.replay.corpus import (
    CorpusStats,
    export_corpus,
    inspect_collection,
    stream_edges,
)

pytestmark = pytest.mark.emulator


@pytest.fixture
def edges_collection(emulator_client: firestore.Client) -> Iterator[str]:
    name = f"stage3_edges_test_{uuid.uuid4().hex[:12]}"
    yield name
    for document in emulator_client.collection(name).stream():
        document.reference.delete()


def seed(client: firestore.Client, collection: str, documents: dict[str, dict]) -> None:
    batch = client.batch()
    for doc_id, body in documents.items():
        batch.set(client.collection(collection).document(doc_id), body)
    batch.commit()


def edge(**overrides) -> dict:
    document = {
        "claimAId": "claim-a",
        "claimBId": "claim-b",
        "claimAText": "Revenue grew 12% in Q3.",
        "claimBText": "Q3 revenue was up twelve percent.",
        "verdict": "CORROBORATES",
        "votes": ["CORROBORATES", "CORROBORATES", "NEUTRAL"],
        "golden": False,
        "method": "ENSEMBLE",
        "stage3RunId": "run-1",
    }
    document.update(overrides)
    return document


def test_documents_round_trip_from_firestore_into_records(
    emulator_client: firestore.Client, edges_collection: str
) -> None:
    seed(
        emulator_client,
        edges_collection,
        {
            "pair-1": edge(),
            "pair-2": edge(verdict="NEUTRAL", golden=True),
            "pair-3": edge(method="GK_G1_NLI"),  # the cascade's own output
        },
    )

    stats = CorpusStats()
    records = list(export_corpus(stream_edges(emulator_client, edges_collection), stats=stats))

    assert {record.pair_id for record in records} == {"pair-1", "pair-2"}
    assert stats.skipped_not_ensemble == 1
    assert stats.golden == 1
    assert next(r for r in records if r.pair_id == "pair-1").votes == (
        "CORROBORATES",
        "CORROBORATES",
        "NEUTRAL",
    )


def test_the_export_can_be_scoped_to_one_stage3_run(
    emulator_client: firestore.Client, edges_collection: str
) -> None:
    seed(
        emulator_client,
        edges_collection,
        {
            "pair-1": edge(stage3RunId="run-1"),
            "pair-2": edge(stage3RunId="run-2"),
        },
    )

    records = list(export_corpus(stream_edges(emulator_client, edges_collection, "run-2")))

    assert [record.pair_id for record in records] == ["pair-2"]


def test_inspect_reads_real_documents(
    emulator_client: firestore.Client, edges_collection: str
) -> None:
    """The pass that tells an operator whether the default field map fits their backup."""
    seed(emulator_client, edges_collection, {"pair-1": edge()})

    report = inspect_collection(
        document for _, document in stream_edges(emulator_client, edges_collection)
    )

    assert report["sampled"] == 1
    assert report["resolvedByDefaultMap"]["verdict"] == "1/1"
    assert "claimAId" in report["topLevelKeys"]


def test_an_empty_collection_exports_nothing_without_failing(
    emulator_client: firestore.Client, edges_collection: str
) -> None:
    stats = CorpusStats()
    assert list(export_corpus(stream_edges(emulator_client, edges_collection), stats=stats)) == []
    assert stats.scanned == 0
