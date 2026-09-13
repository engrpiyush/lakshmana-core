# Gatekeeper runbook (LK-12 / VA-104)

Operating the Stage 3 gatekeeper cascade. Design authority: LLD page 255688733 §11, §12
(mirror `lakshmana-gatekeeper-lld-wiki.md`).

**`gatekeeper_runs` is the single source of truth.** A stalled run spans Pub/Sub, dispatcher
logs and job logs — but every one of them keys on `runRequestId`, and the run doc already
holds the answer. Read the run doc first, then filter logs by `runRequestId`. Never start
in the logs.

Named database is `vishwakarma-labelling` (not `(default)`). Services: `gatekeeper-dispatcher`
(Cloud Run **service**), `gatekeeper-worker` (Cloud Run **job**). Topic `gatekeeper-requests`,
dead-letter `gatekeeper-requests-dlq` (pull subscription `gatekeeper-requests-dlq-pull`).

---

## 1. First response

1. Open the run doc: Firestore console → database `vishwakarma-labelling` → `gatekeeper_runs` →
   the `runRequestId`. (Or the vishwamitra run page, which polls it.)
2. Read `state`, then the `gates` map. The **first gate not `SUCCEEDED`** is where it stopped.
3. On that gate read `state`, `errorCode`, `errorDetail`, `attempt`, `sweepAttempt`,
   `leaseOwner`, `leaseExpiresAt`.
4. Filter logs by that `runRequestId` (§6). Act per the tables below.

---

## 2. Reading the run doc

| Field | Says |
| --- | --- |
| `state` | `REQUESTED`/`RUNNING` = in flight · `SUCCEEDED`/`SUPERSEDED` = done · `FAILED` = needs an operator |
| `gates.<G>.state` | `PENDING`/`RUNNING`/`SUCCEEDED`/`FAILED`/`SKIPPED` |
| `gates.<G>.errorCode` | the `GK_E_*` on a `FAILED` gate — look it up in §4 |
| `gates.<G>.sweepAttempt` | rescues so far; `3` then FAILED = `GK_E_STUCK` |
| `gates.<G>.leaseExpiresAt` | past + gate `RUNNING` = the worker died or is renewing; the sweeper decides |
| `gates.<G>.counters` | per-gate funnel (`seen`, `neutral`, `forwarded`, `decided`, `capExceeded`, …) |
| `errorCode` / `errorDetail` (run level) | a FINALIZE reconciliation failure — no gate carries it (§4, `GK_E_FIRESTORE_TXN`) |
| `totals` | `pairsSeen`, `decidedByGates`, `escalatedLlm`, `escalatedHuman`, `llmSpendUsd` — written at FINALIZE |

**Reconciliation invariant** (FINALIZE refuses to write `totals` without it):
`decidedByGates + escalatedLlm + escalatedHuman == pairsSeen`, and no pair left at a gate.

---

## 3. Symptom → action

| Symptom | Read | Action |
| --- | --- | --- |
| Run stuck `RUNNING`, a gate `RUNNING`, lease **not** expired | `leaseExpiresAt` | Worker alive (renewing its lease each batch). Wait; watch `gatekeeper/gate_duration`. |
| Run stuck `RUNNING`, a gate `RUNNING`, lease **expired** | `sweepAttempt` | Confirm Cloud Scheduler `gatekeeper-sweep` is firing 2xx (§5). The sweeper rescues on its next pass. |
| Run stuck `RUNNING`, a gate `PENDING`, predecessor `SUCCEEDED` | — | The chain message was lost. The sweeper republishes after the grace window; if not, check the scheduler (§5). |
| Gate `FAILED` | `errorCode` | Fix per §4, then **retrigger FROM_GATE** at that gate from the vishwamitra run page. |
| Run `FAILED`, no gate `FAILED` | run-level `errorCode` | A FINALIZE reconciliation failure — a stranded pair. Inspect `gatekeeper_pairs` for the `stage3RunId`; retrigger FROM_GATE at G4. |
| Verdicts look fabricated (every escalated pair NEUTRAL) | `gates.G4.counters`, edge `judgeModel` | A dry-run tail slipped through. Since VA-158 this FAILS the gate (`GK_E_G4_DRYRUN`); confirm `GATEKEEPER_G4_LIVE_CALLS=true` on the worker, retrigger FROM_GATE at G4. |
| Human queue larger than expected | `totals.escalatedHuman`, `gatekeeper/escalation_tail` | Contradiction candidates + tail failures. Expected ≤5% at the tail; a spike means thresholds drift or a subject that does not fit the cascade. |

---

## 4. `GK_E_*` codes

| Code | Meaning | Retried? | Action |
| --- | --- | --- | --- |
| `GK_E_SCHEMA` | Unparseable payload — refused, dead-lettered | No (→ DLQ) | Inspect the DLQ message (§7). A malformed publisher, not a run to rescue. |
| `GK_E_MODEL_FETCH` | GCS artifact missing or sha256 mismatch | No | Check the model bucket + roster manifest; fix, retrigger FROM_GATE. |
| `GK_E_NEO4J_UNAVAILABLE` | Hydrate failed beyond retries (also: subjectId/claim text absent) | No | Confirm Neo4j reachable over the VPC connector; usually just retrigger FROM_GATE. |
| `GK_E_FIRESTORE_TXN` | Claim/commit contention beyond retries, **or** a FINALIZE reconciliation failure | Dispatcher: 500→redeliver · Worker: gate FAILED | If a gate: retrigger FROM_GATE. If run-level: a stranded pair — inspect `gatekeeper_pairs`, retrigger FROM_GATE at G4. |
| `GK_E_VERTEX` | G4 quota/5xx beyond backoff | Per **pair**, gate SUCCEEDS | The affected pairs went to the human queue with `LLM_ERROR`. Nothing to do unless the rate is high — then check Vertex quota. |
| `GK_E_CAP_EXCEEDED` | G4 hit `maxLlmPairs` | Run SUCCEEDS | Remainder is in the human queue with `CAP_EXCEEDED`. Tail exceeded target; revisit thresholds (VA-160). |
| `GK_E_STUCK` | 3 sweeps exhausted → run FAILED | No | The worker never made progress across 3 rescues. Read the worker logs for the underlying crash, fix, retrigger FROM_GATE. |
| `GK_E_G4_DRYRUN` | G4 reached a GATEKEEPER run with the dry-run double (VA-158) | No | Live calls are off on a deciding run. Set `GATEKEEPER_G4_LIVE_CALLS=true` on the worker (Terraform `g4_live_calls`), redeploy, retrigger FROM_GATE at G4. **No fabricated verdict was written** — the gate refused before writing. |

---

## 5. Alerts → action

| Alert | Fires on | First check |
| --- | --- | --- |
| DLQ is not empty | `num_undelivered_messages > 0` for 5 min | §7 — almost always `GK_E_SCHEMA`. |
| Worker job execution failed | a `gatekeeper-worker` execution result = failed | Run doc → failed gate → `errorCode` (§4). Catches `GK_E_G4_DRYRUN`. |
| Run stuck in RUNNING | `run_overdue` log metric | Is Cloud Scheduler `gatekeeper-sweep` firing 2xx? Then `sweepAttempt` on the run doc. |
| G4 LLM cap exceeded | `cap_exceeded` log metric | Tail volume; revisit thresholds (VA-160). Not a failure. |
| Dispatcher 5xx rate | 5xx over threshold for 5 min | Firestore claim contention (`GK_E_FIRESTORE_TXN`) or a crash loop — **not** normal at-least-once traffic (dups→204, schema→400). |

Sweeper health: `gcloud scheduler jobs describe gatekeeper-sweep --location=<region>` and confirm
recent executions are 2xx. A silent scheduler is the usual cause of a genuinely stuck run.

---

## 6. Log filters

Every line in all three components carries `runRequestId` and `gate` (enforced by
`tests/test_conventions.py::test_every_line_carries_run_request_id_and_gate`). Filter by them.

```
# one run, across the worker
gcloud logging read 'resource.type="cloud_run_job"
  resource.labels.job_name="gatekeeper-worker"
  jsonPayload.runRequestId="<RUN_REQUEST_ID>"' --limit=100 --freshness=1d

# one run, across the dispatcher
gcloud logging read 'resource.type="cloud_run_revision"
  resource.labels.service_name="gatekeeper-dispatcher"
  jsonPayload.runRequestId="<RUN_REQUEST_ID>"' --limit=100 --freshness=1d

# every coded error, most recent first
gcloud logging read 'jsonPayload.errorCode:"GK_E_"' --limit=50 --freshness=1d

# one gate's commit lines (duration + counters)
gcloud logging read 'jsonPayload.event="gate_committed"
  jsonPayload.gate="G1_NEUTRAL"' --limit=20 --freshness=1d
```

Log metrics (Metrics Explorer): `gatekeeper/gate_duration` (per gate p50/p95),
`gatekeeper/gate_error` (by `code`), `gatekeeper/escalation_tail` (paid tail per run),
`gatekeeper/cap_exceeded`, `gatekeeper/run_overdue`.

---

## 7. Dead-letter queue

```
# inspect (does not ack)
gcloud pubsub subscriptions pull gatekeeper-requests-dlq-pull --limit=10 \
  --format='table(message.publishTime, message.data.decode("base64"))'
```

Almost always `GK_E_SCHEMA`: a payload the dispatcher refused to guess at. Fix the publisher.

Replay a message that was dead-lettered for a transient reason (not schema): decode its
`message.data` and re-publish it to the topic, then ack it off the DLQ.

```
gcloud pubsub topics publish gatekeeper-requests --message="$(<decoded-payload.json)"
gcloud pubsub subscriptions pull gatekeeper-requests-dlq-pull --limit=1 --auto-ack
```

The claim transaction makes a replay safe: a duplicate is a no-op the dispatcher answers 204.

---

## 8. Retrigger

- **FROM_GATE** (resume a FAILED/stalled gate on the same `runRequestId`): operator action on
  the vishwamitra run page. Only a `FAILED` gate or a lease-expired `RUNNING` one is eligible.
  The purge is **not** run; earlier gates' output is kept.
- **FROM_START** (fresh `runRequestId`): re-runs the whole cascade. G1's first act purges this
  run's `GK_*` edge rows and resets the queue; any in-flight predecessor is SUPERSEDED. The
  ensemble's own rows are never touched (D-5).

Both are published by vishwamitra, never by a gatekeeper CLI. The gatekeeper only ever reacts
to a message on `gatekeeper-requests`.

---

## 9. Local reproduction

```
# emulator up (named-DB semantics are irrelevant to the emulator)
FIRESTORE_EMULATOR_HOST=127.0.0.1:8082 uv run pytest

# the full cascade end to end against the emulator, with a dry-run tail
uv run pytest tests/test_e2e_cascade.py -q
```

Live calls stay off locally (`GATEKEEPER_G4_LIVE_CALLS` unset). A GATEKEEPER-mode run needs a
live door or it refuses at G4 (`GK_E_G4_DRYRUN`); run SHADOW, or inject a double, to exercise
the tail without Vertex. Neo4j and Vertex are never reached by the test suite.
