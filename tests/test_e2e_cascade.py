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


# --- the chain, three gates deep -------------------------------------------------------------


class CascadeScorer:
    """One stub standing in for all three gates' models.

    In production these are three different artifacts; the `scorer_factory` seam hands each
    gate whatever its binding asks for, and here that is this. It answers both interfaces —
    `score_nli` for G1 and G3, `score_grounding` for G2 — so a single double can drive the
    whole chain without any gate knowing it is not talking to ONNX.
    """

    def __init__(self, nli_by_premise, grounding_default):
        self.nli_by_premise = nli_by_premise
        self.grounding_default = grounding_default

    def score_nli(self, pairs):
        return [self.nli_by_premise[premise] for premise, _ in pairs]

    def score_grounding(self, pairs):
        return [self.grounding_default for _ in pairs]


def _cascade_config(suffix: str):
    """Isolated collections for one chain run, and the config that points at them."""
    collections = {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
        "claims": f"claims_test_{suffix}",
    }
    return collections, load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_CLAIMS_COLLECTION": collections["claims"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )


def _cascade_fixture():
    """Six pairs shaped so that all three gates have something to do.

    Pairs 0-1 settle at G1, 2-3 forward into G2, and 4-5 carry G1's contradiction flag —
    which must cross G2 untouched and reach the cross-check. G2's support is deliberately
    below `supportMin`, so its own leftovers fall through to G3 as well.
    """
    from gatekeeper.scoring.encoder import GroundingScore, NliScores

    contradictory = NliScores(entailment=0.03, neutral=0.05, contradiction=0.92)
    pairs = _pairs(6)
    texts = _texts(pairs)
    nli_by_premise = {}
    for index, pair in enumerate(pairs):
        scores = (NEUTRAL, UNDECIDED, contradictory)[min(index // 2, 2)]
        for claim_id in (pair.claim_a_id, pair.claim_b_id):
            nli_by_premise[texts[claim_id]] = scores

    return pairs, texts, CascadeScorer(nli_by_premise, GroundingScore(support=0.40))


def test_the_chain_drives_itself_from_g1_through_g2_to_g3(
    store, emulator_client, runs_collection
) -> None:
    """Session05's integration proof: one push, three gates, no hand-holding.

    Each gate commits and publishes the next, and the *next message is fed back in* rather
    than the next gate being called directly — so this exercises the dispatcher's claim
    transaction three times over, which is where a gate-ordering or lease bug would show up.

    The fixture is built so all three gates have work: pairs that G1 flags reach G3 without
    G2 touching them, pairs G1 forwards get judged by G2, and what G2 cannot settle falls
    through to the cross-check.
    """
    from fastapi.testclient import TestClient

    from gatekeeper.enums import QueueTier
    from gatekeeper.worker.edges import edge_key

    suffix = uuid.uuid4().hex[:12]
    collections, config = _cascade_config(suffix)

    stage3_run_id = f"stage3run-{suffix}"
    subject_id = f"subject-{suffix}"
    emulator_client.collection(collections["runs"]).document(stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )

    pairs, texts, scorer = _cascade_fixture()

    publisher = RecordingPublisher()
    launcher = InlineWorkerLauncher(
        store, emulator_client, config, publisher, FakeGraph(pairs, texts), scorer
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

    with TestClient(create_app(dispatcher), raise_server_exceptions=False) as client:
        pending = [request]
        driven = []
        # Follow the chain the gates publish, up to G3. G4 is VA-102's, and today it is
        # still the no-op runner, so the walk stops where this session's scope does.
        while pending:
            message = pending.pop(0)
            before = len(publisher.published)
            assert client.post("/pubsub/push", json=_envelope(message)).status_code == 204
            driven.append(message.gate)
            pending.extend(
                published
                for published in publisher.published[before:]
                if published.gate is not Gate.G4_ESCALATION
            )

    assert driven == [Gate.G1_NEUTRAL, Gate.G2_CORROBORATION, Gate.G3_CONTRADICTION]
    assert launcher.exit_codes == [EXIT_OK, EXIT_OK, EXIT_OK]

    run = store.get(request.run_request_id)
    for gate in (Gate.G1_NEUTRAL, Gate.G2_CORROBORATION, Gate.G3_CONTRADICTION):
        assert run.gate(gate).state is GateState.SUCCEEDED

    assert run.gate(Gate.G1_NEUTRAL).counters["neutral"] == 2
    assert run.gate(Gate.G1_NEUTRAL).counters["contraFlagged"] == 2

    g2 = run.gate(Gate.G2_CORROBORATION).counters
    assert g2["flaggedPassThrough"] == 2, "flagged pairs must cross G2 unjudged"
    assert g2["seen"] == 2
    assert g2["forwarded"] == 4

    g3 = run.gate(Gate.G3_CONTRADICTION).counters
    assert g3["seen"] == 4
    assert g3["humanRouted"] == 2, "the two flagged pairs agreed across both families"

    # The candidates left the cascade for review, and carry a row the queue can surface.
    queue = PairQueue(emulator_client, collection=collections["queue"])
    human = [pair for pair in queue.all_pairs(stage3_run_id) if pair.tier is QueueTier.HUMAN]
    assert len(human) == 2
    assert all(pair.decided_by is None for pair in human)

    candidate = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(human[0].claim_a_id, human[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert candidate["relation"] == "CONTRADICTS"
    # One row per pair, carrying every gate that looked at it (§7.2's {g1, g2, g3}).
    assert set(candidate["stageScores"]) == {"g1", "g3"}, "G2 never judged a flagged pair"

    forwarded = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(pairs[2].claim_a_id, pairs[2].claim_b_id))
        .get()
        .to_dict()
    )
    assert set(forwarded["stageScores"]) == {"g1", "g2", "g3"}

    # And family A on that row is exactly what G1 wrote — G3 read it, never re-derived it.
    assert forwarded["stageScores"]["g1"]["conFwd"] == pytest.approx(UNDECIDED.contradiction)


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

    # `load_config()` and not `load_config({})`: the empty mapping shuts out the real
    # environment, so `GATEKEEPER_NEO4J_PASSWORD` could never arrive and this test skipped
    # unconditionally — the one test that proves the cypher parses, never running.
    config = load_config()
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
