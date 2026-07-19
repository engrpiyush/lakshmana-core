"""G4_ESCALATION against the emulator (LLD §8, §11, D-1).

Real queue, real edge writes, real routing — with the graph and the *model* stubbed. The
model double is not a convenience here, it is the point: G4 is the only gate that spends
money, and the DoD's fault-injection cases are precisely the ones a real model will not
produce on demand.

Four properties, in the order §11's table names them:

* the cap sends the remainder to a human with ``CAP_EXCEEDED`` and the run survives;
* a Vertex error sends its pair to a human with ``LLM_ERROR`` and the *gate* survives;
* a response nobody can parse is treated as an error, never as a verdict;
* SHADOW makes zero calls and records ``WOULD_ESCALATE_LLM``.

Plus the one that is not in the table but costs the most if wrong: the cap is durable, so a
gate rescued near its ceiling cannot spend the budget a second time.
"""

from __future__ import annotations

import json
import uuid

import pytest

from gatekeeper.clients.vertex import DryRunVertexClient, VertexResponse
from gatekeeper.config import load_config
from gatekeeper.enums import (
    SHADOW_WOULD_ESCALATE_LLM,
    EscalationReason,
    Gate,
    JudgeMode,
    Method,
    QueueTier,
    Verdict,
)
from gatekeeper.errors import Neo4jUnavailableError, VertexError
from gatekeeper.gates.g4 import run_g4
from gatekeeper.worker.edges import edge_key
from gatekeeper.worker.gates import GateContext
from gatekeeper.worker.queue import PairQueue
from tests.test_g1_gate import FakeGraph, _pairs, _texts

pytestmark = pytest.mark.emulator

OWNER = "worker/test/task-0"


# --- doubles ---------------------------------------------------------------------------------


class ScriptedLlm:
    """A Vertex door that answers from a script, so every failure mode is reachable.

    Each entry is either a JSON string (returned as the model's text), or an exception
    instance (raised). ``None`` means "keep using the last entry forever", which is how a
    test asks for N identical answers without writing N of them.
    """

    live = True

    def __init__(self, script, model: str = "test-lite") -> None:
        self._script = list(script)
        self.model = model
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> VertexResponse:
        self.prompts.append(prompt)
        entry = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(entry, Exception):
            raise entry
        return VertexResponse(text=entry, prompt_tokens=100, output_tokens=20)


def _verdict_json(relation: str = "NEUTRAL", confidence: float = 0.7) -> str:
    return json.dumps(
        [
            {
                "i": 1,
                "relation": relation,
                "confidence": confidence,
                "rationale": "Because the two claims describe different episodes.",
                "temporalNote": None,
            }
        ]
    )


# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture
def collections():
    suffix = uuid.uuid4().hex[:12]
    return {
        "queue": f"gatekeeper_pairs_test_{suffix}",
        "edges": f"stage3_edges_test_{suffix}",
        "runs": f"stage3_runs_test_{suffix}",
        "prompts": f"extraction_prompts_test_{suffix}",
    }


@pytest.fixture
def gate_config(collections):
    return load_config(
        {
            "GATEKEEPER_QUEUE_COLLECTION": collections["queue"],
            "GATEKEEPER_STAGE3_EDGES_COLLECTION": collections["edges"],
            "GATEKEEPER_STAGE3_RUNS_COLLECTION": collections["runs"],
            "GATEKEEPER_PROMPTS_COLLECTION": collections["prompts"],
            "GATEKEEPER_QUEUE_BATCH_SIZE": "4",
        }
    )


@pytest.fixture
def subject(emulator_client, collections, claimed):
    subject_id = f"subject-{uuid.uuid4().hex[:8]}"
    emulator_client.collection(collections["runs"]).document(claimed.stage3_run_id).set(
        {"subjectId": subject_id, "paramsSnapshot": '{"judgeMode": "GATEKEEPER"}'}
    )
    return subject_id


def _seed_at_g4(emulator_client, collections, stage3_run_id, intake_id, pairs):
    """Put pairs on the queue already sitting at G4, as G1 or G3 would have left them."""
    queue = PairQueue(emulator_client, collection=collections["queue"])
    queue.seed(stage3_run_id, intake_id, pairs)
    queue.route_all(
        [
            (pair, {"gate": Gate.G4_ESCALATION.value, "tier": QueueTier.LLM_TAIL.value})
            for pair in queue.all_pairs(stage3_run_id)
        ]
    )
    return queue


def _context(emulator_client, gate_config, run, graph, llm, judge_mode=JudgeMode.GATEKEEPER):
    run.judge_mode = judge_mode
    return GateContext(
        run=run,
        gate=Gate.G4_ESCALATION,
        config=gate_config,
        client=emulator_client,
        lease_owner=OWNER,
        reader_factory=lambda: graph,
        llm_factory_override=lambda config, model, budget: llm,
    )


def _run_g4(store, claimed, emulator_client, gate_config, graph, llm, **kwargs):
    return run_g4(
        _context(
            emulator_client,
            gate_config,
            store.get(claimed.run_request_id),
            graph,
            llm,
            **kwargs,
        )
    )


# --- the happy path --------------------------------------------------------------------------


def test_g4_settles_pairs_and_writes_the_prose(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§8: "prose lands exactly where humans will look" — on the edge row."""
    pairs = _pairs(2)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)
    llm = ScriptedLlm([_verdict_json("CORROBORATES", 0.88)])

    counters = _run_g4(
        store, claimed, emulator_client, gate_config, FakeGraph(pairs, _texts(pairs)), llm
    )

    assert counters["seen"] == 2
    assert counters["decided"] == 2
    assert counters["llmCalls"] == 2
    assert counters["capExceeded"] == 0
    assert counters["llmError"] == 0

    # One call per pair (§8), not one call for the batch.
    assert len(llm.prompts) == 2

    row = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert row["relation"] == Verdict.CORROBORATES.value
    assert row["method"] == Method.GK_G4_LLM.value
    assert row["confidence"] == pytest.approx(0.88)
    assert row["rationale"].startswith("Because the two claims")
    # No `stageScores.g4`: an LLM's self-reported confidence is not a probability the
    # calibration corpus can use, so the tail contributes prose rather than numbers.
    assert "g4" not in (row.get("stageScores") or {})

    queued = PairQueue(emulator_client, collection=collections["queue"]).all_pairs(
        claimed.stage3_run_id
    )
    assert {pair.gate for pair in queued} == {None}
    assert {pair.decided_by for pair in queued} == {Method.GK_G4_LLM.value}
    assert {pair.tier for pair in queued} == {QueueTier.LLM_TAIL}


def test_the_spend_counter_tracks_the_tokens_the_model_reported(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """The bill is computed from Google's counts, so the counter has to be too."""
    pairs = _pairs(2)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    counters = _run_g4(
        store,
        claimed,
        emulator_client,
        gate_config,
        FakeGraph(pairs, _texts(pairs)),
        ScriptedLlm([_verdict_json()]),
    )

    assert counters["promptTokens"] == 200
    assert counters["outputTokens"] == 40
    # 200 in at $0.10/M + 40 out at $0.40/M — the shipped default rates.
    assert counters["llmSpendUsd"] == pytest.approx(200 / 1e6 * 0.10 + 40 / 1e6 * 0.40)


# --- fault injection: the cap ------------------------------------------------------------------


def test_over_cap_pairs_go_to_a_human_and_cost_nothing(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§11's ``GK_E_CAP_EXCEEDED`` row: remainder to HUMAN, run still SUCCEEDED."""
    pairs = _pairs(5)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    run = store.get(claimed.run_request_id)
    run.config_snapshot["g4"]["maxLlmPairs"] = 2
    run.judge_mode = JudgeMode.GATEKEEPER
    llm = ScriptedLlm([_verdict_json()])

    counters = run_g4(
        GateContext(
            run=run,
            gate=Gate.G4_ESCALATION,
            config=gate_config,
            client=emulator_client,
            lease_owner=OWNER,
            reader_factory=lambda: FakeGraph(pairs, _texts(pairs)),
            llm_factory_override=lambda config, model, budget: llm,
        )
    )

    assert counters["seen"] == 5
    assert counters["decided"] == 2
    assert counters["capExceeded"] == 3
    # The cap is about spend: three pairs were never sent anywhere.
    assert counters["llmCalls"] == 2
    assert len(llm.prompts) == 2

    queue = PairQueue(emulator_client, collection=collections["queue"])
    queued = queue.all_pairs(claimed.stage3_run_id)
    human = [pair for pair in queued if pair.tier is QueueTier.HUMAN]
    assert len(human) == 3
    assert all(pair.decided_by is None for pair in human)
    # Drained, not stranded — nothing is left sitting at a committed gate.
    assert all(pair.gate is None for pair in queued)

    over = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(human[0].claim_a_id, human[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert over["escalationReason"] == EscalationReason.CAP_EXCEEDED.value
    assert "relation" not in over


def test_the_cap_is_durable_across_a_resumed_gate(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """A rescued gate must not get a fresh budget — the ceiling is per run, not per attempt.

    The first execution spends the whole cap. The second is what a sweeper rescue looks
    like: same run, same queue, a brand-new counter. If the cap were held in that counter,
    this would spend it all over again.
    """
    pairs = _pairs(4)
    queue = _seed_at_g4(
        emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs
    )

    def execute(limit_pairs):
        run = store.get(claimed.run_request_id)
        run.config_snapshot["g4"]["maxLlmPairs"] = 2
        run.judge_mode = JudgeMode.GATEKEEPER
        llm = ScriptedLlm([_verdict_json()])
        counters = run_g4(
            GateContext(
                run=run,
                gate=Gate.G4_ESCALATION,
                config=gate_config,
                client=emulator_client,
                lease_owner=OWNER,
                reader_factory=lambda: FakeGraph(limit_pairs, _texts(limit_pairs)),
                llm_factory_override=lambda config, model, budget: llm,
            )
        )
        return counters, llm

    first, first_llm = execute(pairs)
    assert first["decided"] == 2
    assert len(first_llm.prompts) == 2

    # Put the two over-cap pairs back at G4, as a rescue after a crash would find them.
    queue.route_all(
        [
            (pair, {"gate": Gate.G4_ESCALATION.value, "tier": QueueTier.LLM_TAIL.value})
            for pair in queue.all_pairs(claimed.stage3_run_id)
            if pair.decided_by is None
        ]
    )

    second, second_llm = execute(pairs)
    assert second["decided"] == 0, "the resumed gate spent budget that was already gone"
    assert second["capExceeded"] == 2
    assert second_llm.prompts == [], "no call may be made once the run's cap is reached"


# --- fault injection: the model ----------------------------------------------------------------


def test_a_vertex_error_costs_its_pair_not_the_gate(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§11's partial-tail policy: affected pairs go HUMAN, the gate still SUCCEEDS."""
    pairs = _pairs(3)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    # Middle call fails; the ones on either side must still be judged.
    llm = ScriptedLlm(
        [_verdict_json(), VertexError("429 beyond backoff"), _verdict_json()],
    )
    counters = _run_g4(
        store, claimed, emulator_client, gate_config, FakeGraph(pairs, _texts(pairs)), llm
    )

    assert counters["seen"] == 3
    assert counters["decided"] == 2
    assert counters["llmError"] == 1

    queue = PairQueue(emulator_client, collection=collections["queue"])
    failed = [
        pair for pair in queue.all_pairs(claimed.stage3_run_id) if pair.tier is QueueTier.HUMAN
    ]
    assert len(failed) == 1
    assert failed[0].decided_by is None

    row = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(failed[0].claim_a_id, failed[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert row["escalationReason"] == EscalationReason.LLM_ERROR.value


def test_an_unparseable_answer_is_an_error_never_a_verdict(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """A refusal and a quota failure land in the same place, because the fix is the same."""
    pairs = _pairs(1)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    counters = _run_g4(
        store,
        claimed,
        emulator_client,
        gate_config,
        FakeGraph(pairs, _texts(pairs)),
        ScriptedLlm(["I would rather not judge this pair."]),
    )

    assert counters["decided"] == 0
    assert counters["llmError"] == 1

    row = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert row["escalationReason"] == EscalationReason.LLM_ERROR.value
    assert "relation" not in row, "an unparseable answer must never become a verdict"


def test_a_missing_claim_card_fails_the_gate(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """Without a card there is nothing to ask about; guessing is worse than failing."""
    pairs = _pairs(2)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    texts = _texts(pairs)
    del texts[pairs[1].claim_a_id]

    with pytest.raises(Neo4jUnavailableError, match="no claim card"):
        _run_g4(
            store,
            claimed,
            emulator_client,
            gate_config,
            FakeGraph(pairs, texts),
            ScriptedLlm([_verdict_json()]),
        )


# --- SHADOW ------------------------------------------------------------------------------------


def test_shadow_mode_makes_zero_calls(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """§8/§9: comparing an LLM to an LLM would spend money to learn nothing."""
    pairs = _pairs(3)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)
    llm = ScriptedLlm([_verdict_json()])

    counters = _run_g4(
        store,
        claimed,
        emulator_client,
        gate_config,
        FakeGraph(pairs, _texts(pairs)),
        llm,
        judge_mode=JudgeMode.SHADOW,
    )

    assert llm.prompts == [], "SHADOW must not call the model"
    assert counters["shadowSuppressed"] == 3
    assert counters["llmCalls"] == 0
    assert counters["llmSpendUsd"] == 0

    row = (
        emulator_client.collection(collections["edges"])
        .document(edge_key(pairs[0].claim_a_id, pairs[0].claim_b_id))
        .get()
        .to_dict()
    )
    assert row["shadow"]["verdict"] == SHADOW_WOULD_ESCALATE_LLM
    # The row asserts nothing about the pair — the ensemble is the verdict writer (§9).
    assert "relation" not in row

    queued = PairQueue(emulator_client, collection=collections["queue"]).all_pairs(
        claimed.stage3_run_id
    )
    assert {pair.tier for pair in queued} == {QueueTier.LLM_TAIL}
    assert all(pair.decided_by is None for pair in queued), "nothing decided these pairs"


# --- the default door ---------------------------------------------------------------------------


def test_the_dry_run_double_is_the_default(
    store, claimed, emulator_client, gate_config, collections, subject
) -> None:
    """An unconfigured worker runs the whole tail without reaching Vertex (hard rule 4)."""
    pairs = _pairs(2)
    _seed_at_g4(emulator_client, collections, claimed.stage3_run_id, claimed.intake_id, pairs)

    run = store.get(claimed.run_request_id)
    run.judge_mode = JudgeMode.GATEKEEPER
    # No `llm_factory_override`: this exercises `client_for` and the shipped default.
    counters = run_g4(
        GateContext(
            run=run,
            gate=Gate.G4_ESCALATION,
            config=gate_config,
            client=emulator_client,
            lease_owner=OWNER,
            reader_factory=lambda: FakeGraph(pairs, _texts(pairs)),
        )
    )

    assert counters["decided"] == 2
    # Nothing was spent, because nothing was called.
    assert counters["llmSpendUsd"] == 0
    assert counters["promptTokens"] == 0


def test_the_dry_run_answer_says_it_is_a_dry_run() -> None:
    """A double whose output could be mistaken for a judgement is how one ends up in audit."""
    from gatekeeper.gates.prompts import parse_g4_response

    client = DryRunVertexClient()
    assert client.live is False

    verdict = parse_g4_response(client.generate("anything").text)
    assert verdict.relation is Verdict.NEUTRAL
    assert "Dry-run" in (verdict.rationale or "")
