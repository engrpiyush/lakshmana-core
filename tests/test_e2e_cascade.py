"""Local end-to-end: a push message drives G1 and the chain publishes G2 (session04 DoD).

This is the wiring test the whole session builds toward. A real Pub/Sub push envelope goes
into the real dispatcher, which runs the real claim transaction against the emulator, which
starts a worker that runs the real G1 gate, which writes real verdicts and publishes the
real next-gate message. The only doubles are the two things a laptop cannot afford: the
graph and the ONNX session.

The `neo4j`-marked test at the bottom is the other half of that promise — it proves the
hydration cypher is valid against a *real* Neo4j rather than only against the fake. It is
strictly read-only: the dev container holds the owner's working state and this suite does
not write to it.
"""

from __future__ import annotations

import base64
import json
import uuid

import pytest

from gatekeeper.clients.pubsub import RecordingPublisher
from gatekeeper.config import load_config
from gatekeeper.contracts.payload import SCHEMA_VERSION, GatekeeperRunRequest
from gatekeeper.dispatcher.app import Dispatcher, create_app
from gatekeeper.dispatcher.oidc import OidcVerifier
from gatekeeper.enums import Gate, GateState, JudgeMode, Method, RunMode, TriggeredBy
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.main import EXIT_OK, run_gate
from gatekeeper.worker.queue import PairQueue
from tests.test_g1_gate import NEUTRAL, UNDECIDED, FakeGraph, StubScorer, _pairs, _texts

pytestmark = pytest.mark.emulator

OWNER = "dispatcher/test/owner"


class InlineWorkerLauncher:
    """Runs the worker in-process instead of starting a Cloud Run Job.

    In production these are two machines; here they are two function calls, which is
    exactly the seam ``JobLauncher`` exists to make substitutable.
    """

    def __init__(self, store, client, config, publisher, graph, scorer):
        self.store = store
        self.client = client
        self.config = config
        self.publisher = publisher
        self.graph = graph
        self.scorer = scorer
        self.exit_codes: list[int] = []

    def execute(self, request: GatekeeperRunRequest, *, lease_owner: str) -> str:
        code = run_gate(
            self.store,
            request.run_request_id,
            request.gate,
            lease_owner,
            context_factory=lambda run: GateContext(
                run=run,
                gate=request.gate,
                config=self.config,
                client=self.client,
                lease_owner=lease_owner,
                reader_factory=lambda: self.graph,
                scorer_factory=lambda binding: self.scorer,
            ),
            publisher=self.publisher,
        )
        self.exit_codes.append(code)
        return f"inline-{request.gate.value}"


def _envelope(request: GatekeeperRunRequest) -> dict:
    return {
        "message": {
            "data": base64.b64encode(request.to_bytes()).decode(),
            "messageId": "e2e-1",
            "publishTime": "2026-07-19T04:15:01.000Z",
        },
        "subscription": "projects/p/subscriptions/gatekeeper-requests-push",
    }


def test_a_push_message_runs_g1_and_publishes_g2(
    store, emulator_client, runs_collection, request_factory=None
) -> None:
    from fastapi.testclient import TestClient

    suffix = uuid.uuid4().hex[:12]
    collections = {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
    }
    config = load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )

    stage3_run_id = f"stage3run-{suffix}"
    subject_id = f"subject-{suffix}"
    emulator_client.collection(collections["runs"]).document(stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )

    # Five pairs: three the gate can settle, two it must hand to G2.
    pairs = _pairs(5)
    texts = _texts(pairs)
    by_premise = {}
    for index, pair in enumerate(pairs):
        scores = NEUTRAL if index < 3 else UNDECIDED
        by_premise[texts[pair.claim_a_id]] = scores
        by_premise[texts[pair.claim_b_id]] = scores

    publisher = RecordingPublisher()
    launcher = InlineWorkerLauncher(
        store,
        emulator_client,
        config,
        publisher,
        FakeGraph(pairs, texts),
        StubScorer(by_premise),
    )

    request = GatekeeperRunRequest(
        schema_version=SCHEMA_VERSION,
        run_request_id=str(uuid.uuid4()),
        intake_id=f"intake-{suffix}",
        stage3_run_id=stage3_run_id,
        gate=Gate.G1_NEUTRAL,
        mode=RunMode.FULL,
        triggered_by=TriggeredBy.OPERATOR,
        request_timestamp="2026-07-19T04:15:00.000Z",
    )

    dispatcher = Dispatcher(
        store=store,
        config=config,
        launcher=launcher,
        judge_mode_resolver=lambda _: JudgeMode.GATEKEEPER,
        lease_owner_factory=lambda: OWNER,
        verifier=OidcVerifier(required=False),
        publisher=publisher,
    )

    with TestClient(create_app(dispatcher), raise_server_exceptions=False) as client:
        response = client.post("/pubsub/push", json=_envelope(request))

    # The push is ACKed fast, and the gate behind it succeeded.
    assert response.status_code == 204
    assert launcher.exit_codes == [EXIT_OK]

    run = store.get(request.run_request_id)
    entry = run.gate(Gate.G1_NEUTRAL)
    assert entry.state is GateState.SUCCEEDED
    assert entry.counters["seen"] == 5
    assert entry.counters["neutral"] == 3
    assert entry.counters["forwarded"] == 2

    # The chain drove itself: G2's message exists and names the same run.
    assert [message.gate for message in publisher.published] == [Gate.G2_CORROBORATION]
    assert publisher.published[0].run_request_id == request.run_request_id

    # Verdicts are durable, and the two undecided pairs are waiting at G2.
    written = list(emulator_client.collection(collections["edges"]).stream())
    assert len(written) == 5
    assert {doc.to_dict()["method"] for doc in written} == {Method.GK_G1_NLI.value}

    queue = PairQueue(emulator_client, collection=collections["queue"])
    assert queue.count_at_gate(stage3_run_id, Gate.G2_CORROBORATION) == 2


# --- the real graph ------------------------------------------------------------------------


@pytest.mark.neo4j
def test_the_hydration_cypher_runs_against_a_real_neo4j() -> None:
    """Read-only smoke: the queries parse and execute on the actual container.

    A cypher typo would pass every test above — the fake never parses it — and fail only in
    production. This asks a real server, with a subject id that matches nothing, so it
    returns an empty list rather than touching the owner's dev data. Nothing is written.
    """
    from gatekeeper.clients.neo4j import Neo4jReader, neo4j_driver
    from gatekeeper.worker.hydrate import claim_texts, explanation_texts, queued_pairs

    config = load_config({})
    reader = Neo4jReader(neo4j_driver(config), database=config.get_str("gatekeeper.neo4j.database"))
    try:
        reader.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 — an absent container is a skip, not a failure
        pytest.skip(
            "neo4j not reachable; start the vishwamitra-neo4j container and set "
            f"GATEKEEPER_NEO4J_PASSWORD (dev default: vishwamitra-dev). Cause: {exc}"
        )

    try:
        absent = f"subject-does-not-exist-{uuid.uuid4().hex}"
        assert queued_pairs(reader, absent) == []
        assert claim_texts(reader, [f"claim-{uuid.uuid4().hex}"]) == {}
        assert explanation_texts(reader, [f"claim-{uuid.uuid4().hex}"]) == {}
    finally:
        reader.close()


def test_the_push_payload_is_ids_only() -> None:
    """Belt and braces on LLD §5: claim text must never reach the wire."""
    request = GatekeeperRunRequest(
        schema_version=SCHEMA_VERSION,
        run_request_id=str(uuid.uuid4()),
        intake_id="intake-1",
        stage3_run_id="stage3run-1",
        gate=Gate.G2_CORROBORATION,
        mode=RunMode.FULL,
        triggered_by=TriggeredBy.SYSTEM,
        request_timestamp="2026-07-19T04:15:00.000Z",
    )

    payload = json.loads(request.to_bytes())

    assert set(payload) == {
        "schemaVersion",
        "runRequestId",
        "intakeId",
        "stage3RunId",
        "gate",
        "mode",
        "triggeredBy",
        "requestTimestamp",
    }
