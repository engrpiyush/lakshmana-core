# Gatekeeper — Stage 3 Judge Cascade LLD (Lakshmana)

> v1.2 · 2026-07-19 (v1.2: §8 checkpoint ids verified, §12 log-metric contracts, §13 artifact pin chain + local mirror mode, O‑4 closed; v1.1: mermaid diagrams, Figures 1–4) · Status: **design approved for build; numeric thresholds pending the LK‑5 replay report**
> Confluence: child of Stage 3 LLD (249528322) · Mirror: `lakshmana-core/lakshmana-gatekeeper-lld-wiki.md`
> Companions: Stage 3 LLD §11 (JUDGE), cost wiki 253001729 §9, `PLAN-stage3-cost-cut.md` (Workstream C is **superseded** by this document).

## 1. Purpose & scope

Replace the Stage 3 LLM pair-judge (Gemini flash, k=5 self-consistency, ~$49 of a $56 POC run) with **Lakshmana**: a separate gatekeeper service running an LLM-free cascade of encoder **gates** on CPU, with a small pinned-LLM escalation tail and the human contradiction queue as the final backstop.

- **In scope:** the judge leg only — consuming the judge queue written by MATCH, writing verdicts to `stage3_edges`, and signalling judge completion. Plus its own infra (`lakshmana-infra`), trigger protocol, re-run/retrigger machinery, and observability.
- **Out of scope:** SYNC / RESOLVE_ENTITIES / EMBED / MATCH / ASSEMBLE / SCORE (unchanged, in vishwamitra), publish gating, the contradiction-queue UI (vishwamitra), Stage 4 (permanently — see D‑2).
- **Cost target:** judge leg ≈ **$2–5 per 1,000-claim intake** (from $190–730 naive), ~cents on POC re-runs. No GPU anywhere.

## 2. Decision record

| # | Decision | Note |
| --- | --- | --- |
| D‑1 | **G4 (LLM escalation tail) is owned by Lakshmana** | Gatekeeper calls Vertex flash-lite itself; `SUCCEEDED` ⇒ judging fully done. Vishwamitra's ensemble is never invoked in a gatekeeper run. |
| D‑2 | **Scope is the Stage 3 judge only, permanently** | Not a cross-stage judging home; Stage 4 judging stays in vishwamitra. Shrinks the shared-contract surface to ~6 shapes. |
| D‑3 | **Runtime: Python 3.12** | uv + ruff + pytest; ONNX Runtime (CPU) + HF tokenizers. Replay harness and production gates are the same code. |
| D‑4 | **SHADOW is a first-class JudgeMode** | `LLM \| GATEKEEPER \| SHADOW`. Shadow: cascade predicts into `shadow.*`, ensemble decides. |
| D‑5 | **FROM_START purge deletes gatekeeper-written verdicts only** | `ENSEMBLE` cache is immutable (paid-for corpus + replay ground truth). `gatekeeper_runs` docs are immutable history. |
| D‑6 | **Contracts: protobuf payload + golden-fixture tests** | proto3 JSON mapping is lowerCamelCase — same casing on the wire as in Firestore. No shared code artifact between repos. |
| D‑7 | **Dedicated `lakshmana-infra` repo** | Mirrors the vishwamitra-infra convention. State in the existing TF state bucket under a `lakshmana/` prefix. |
| D‑8 | **Neo4j access is read-only** | Dedicated Enterprise-RBAC user; all Lakshmana writes land in Firestore. The graph keeps a single writer (ASSEMBLE). |

## 3. Naming & coding conventions (no technical surprises)

This section is normative for every artifact in `lakshmana-core` and `lakshmana-infra`.

- **Names.** Repos `lakshmana-core` / `lakshmana-infra`; internal name **gatekeeper** everywhere (the pairing mirrors vishwamitra ⇄ "LLM factory"). Gates: `G1_NEUTRAL`, `G2_CORROBORATION`, `G3_CONTRADICTION`, `G4_ESCALATION`, internal `FINALIZE`.
- **Firestore.** Collection names snake_case (`gatekeeper_runs` — matches `stage3_edges`); **field names camelCase** (matches `blockScore`, `humanAsserted`, `withContext` in the factory). Enum values SCREAMING_SNAKE strings.
- **Wire.** Pub/Sub message = proto3 JSON mapping → lowerCamelCase keys, identical casing to Firestore. Topic `gatekeeper-requests`, DLQ `gatekeeper-requests-dlq` (kebab-case resources, matching infra style).
- **Python internals** snake_case per PEP 8; conversion happens only at the serialization boundary (proto / explicit alias maps). No camelCase leaks into Python identifiers, no snake_case leaks into stored fields.
- **Config.** Kebab-case keys mirroring `app.stage3.*` style (e.g. `gatekeeper.gates.g1.neutral-min`); env overrides SCREAMING_SNAKE with `GATEKEEPER_` prefix (e.g. `GATEKEEPER_G1_NEUTRAL_MIN`).
- **Timestamps.** RFC3339 UTC (`Z`), fields named `*Timestamp` (event time carried in messages) or `*At` (row lifecycle, Firestore server time).
- **Standard envelope — required on every gatekeeper-authored document and message:**

```
schemaVersion     int     // starts at 1; unknown version = DLQ, never guess
runRequestId      string  // UUIDv4 minted by the publisher; idempotency root
intakeId          string
stage3RunId       string
requestTimestamp  string  // RFC3339, when the triggering request was minted
triggeredBy       string  // OPERATOR | SYSTEM | SWEEPER
createdAt / updatedAt     // Firestore server timestamps
attempt           int     // per-gate attempt counter where applicable
```

- **Logging.** Structured JSON; every line carries `runRequestId` + `gate`. Error codes are `GK_E_*` (table in §11).

## 4. Architecture overview

**Figure 1 — components & data flow** *(mermaid source — wrap with the diagram plugin):*

```
flowchart TD
    subgraph VW["vishwamitra-core — Kotlin, Cloud Run service"]
        UI["Stage 3 config UI<br/>JudgeMode: LLM / GATEKEEPER / SHADOW"]
        GC["GatekeeperClient<br/>mint runRequestId · publish · poll · retrigger"]
        POLL["Stage 3 poll (existing)<br/>reads gatekeeper_runs, advances Stage3Run"]
    end

    TOPIC(("Pub/Sub topic<br/>gatekeeper-requests"))
    DLQ(("DLQ<br/>gatekeeper-requests-dlq"))
    ALERT["alerting"]
    SCHED["Cloud Scheduler<br/>/sweep every 15 min"]

    subgraph LK["lakshmana — Python"]
        DISP["dispatcher — Cloud Run service, min 0<br/>OIDC + schema check · transactional claim · execute job · fast ACK"]
        WORK["worker — Cloud Run Job, up to 24h<br/>load gate model · hydrate · infer · write"]
    end

    FS[("Firestore<br/>gatekeeper_runs · stage3_edges · judge queue")]
    NEO[("Neo4j read-only<br/>claims · explanations")]
    GCS[("GCS<br/>gatekeeper-models")]
    VX["Vertex flash-lite<br/>(G4 tail only)"]

    UI --> GC
    GC -- "publish G1 {runRequestId}" --> TOPIC
    TOPIC -- "push + OIDC" --> DISP
    TOPIC -. "5 failed deliveries" .-> DLQ
    DLQ -.-> ALERT
    DISP -- "claim PENDING to RUNNING (txn)" --> FS
    DISP -- "execute {runRequestId, gate}" --> WORK
    WORK --> NEO
    WORK --> GCS
    WORK -- "verdicts + gate commit" --> FS
    WORK -- "G4 escalations" --> VX
    WORK -- "publish next gate" --> TOPIC
    SCHED -- "rescue expired leases" --> DISP
    POLL --> FS
```

Three runtime pieces, one repo: **dispatcher** (thin HTTP), **worker** (framework-free `main()`), **sweeper** (a dispatcher endpoint on a schedule). External services only ever publish the G1 message with intake metadata — the chain is self-driving after that (owner requirement #4).

## 5. Pub/Sub protocol

**Payload — `GatekeeperRunRequest` (proto3; JSON on the wire):**

```
message GatekeeperRunRequest {
  int32  schema_version    = 1;  // json: schemaVersion
  string run_request_id    = 2;  // UUIDv4, minted by publisher
  string intake_id         = 3;
  string stage3_run_id     = 4;
  Gate   gate              = 5;  // G1_NEUTRAL | G2_CORROBORATION | G3_CONTRADICTION | G4_ESCALATION
  Mode   mode              = 6;  // FULL | FROM_GATE
  string triggered_by      = 7;  // OPERATOR | SYSTEM | SWEEPER
  string request_timestamp = 8;  // RFC3339
}
```

IDs only — **claim text never transits Pub/Sub** (privacy + size discipline). Config is not in the payload either: it is frozen into `gatekeeper_runs.configSnapshot` when the run doc is created, and gates read the snapshot — a mid-run config edit cannot produce a mixed-calibration run.

**Delivery semantics and how each hazard is neutralized:**

| Pub/Sub reality | Countermeasure |
| --- | --- |
| At-least-once (duplicates) | Every message is only an *attempted transition*; the transactional claim (§6) makes losers no-ops (ACK + `GK_DUP` log) |
| No ordering | Stale-gate messages fail the precondition (prior gate not `SUCCEEDED` / gate already terminal) and are dropped |
| Redelivery after crash | Gate result is committed **before** the next-gate publish; the crash window between commit and publish is rescued by the sweeper |
| Poison messages | 5 delivery attempts → DLQ → alert; `schemaVersion` mismatch goes straight to DLQ (never guessed at) |

**Publishers** (IAM, §13): vishwamitra app SA (G1 + retriggers), lakshmana worker SA (chain), scheduler/sweeper identity.

**Figure 2 — trigger-to-finalize message flow** *(mermaid source — wrap with the diagram plugin):*

```
sequenceDiagram
    autonumber
    participant VW as vishwamitra<br/>(GatekeeperClient)
    participant PS as Pub/Sub topic
    participant DI as dispatcher
    participant FS as Firestore<br/>(gatekeeper_runs)
    participant JB as worker job (gate Gn)

    VW->>PS: publish {schemaVersion, runRequestId, gate: G1, mode}
    PS->>DI: push (OIDC)
    DI->>FS: txn claim: gate PENDING to RUNNING<br/>(first claim creates run doc + freezes configSnapshot)
    alt claim won
        DI->>JB: execute Run Job {runRequestId, gate}
        DI-->>PS: ACK (fast, under 5s)
        JB->>FS: drain queue tier, write verdicts to stage3_edges
        JB->>FS: commit gate SUCCEEDED (always before publish)
        JB->>PS: publish next gate message (chain repeats to FINALIZE)
    else duplicate or stale message
        DI-->>PS: ACK + drop (GK_DUP)
    end
    Note over FS,VW: vishwamitra poll reads gatekeeper_runs — no callbacks anywhere
```

## 6. Run state machine

Run-level: `REQUESTED → RUNNING → SUCCEEDED | FAILED | SUPERSEDED`. Per-gate (map inside the run doc): `PENDING → RUNNING → SUCCEEDED | FAILED | SKIPPED`.

**Figure 3 — run & gate state machines** *(mermaid source — wrap with the diagram plugin):*

```
stateDiagram-v2
    state "Run lifecycle" as Run {
        [*] --> REQUESTED : publish G1 (new runRequestId)
        REQUESTED --> RUNNING : first gate claim
        RUNNING --> SUCCEEDED : FINALIZE (all gates done)
        RUNNING --> FAILED : a gate FAILED
        FAILED --> RUNNING : operator retrigger FROM_GATE
        REQUESTED --> FAILED : sweeps exhausted (GK_E_STUCK)
        REQUESTED --> SUPERSEDED : FROM_START publish (new run)
        RUNNING --> SUPERSEDED : FROM_START publish (new run)
        SUCCEEDED --> [*]
        SUPERSEDED --> [*]
    }
    state "Gate lifecycle (per-gate map inside the run doc)" as Gate {
        state "PENDING" as GP
        state "RUNNING" as GR
        state "SUCCEEDED" as GS
        state "FAILED" as GF
        [*] --> GP
        GP --> GR : dispatcher txn claim (attempt++, lease 90 min)
        GR --> GS : worker commit, then publish next gate
        GR --> GF : GK_E_* error (run goes FAILED)
        GR --> GP : lease expired, sweeper republish (sweepAttempt max 3)
        GF --> GP : operator retrigger FROM_GATE
        GS --> [*]
    }
```

Transition rules (all inside Firestore transactions on the `gatekeeper_runs` doc):

1. **Claim** — dispatcher transitions gate `PENDING → RUNNING` iff run is `REQUESTED|RUNNING`, the prior gate is `SUCCEEDED` (or the message is a valid `FROM_GATE` entry), and no unexpired lease exists. Sets `attempt++`, `leaseOwner`, `leaseExpiresAt = now + 90 min`. Only a successful claim executes the Job.
2. **Duplicate / stale** — precondition fails → ACK, log, drop. This — not Pub/Sub — is the idempotency mechanism (owner requirement #8: runRequestId is the root, the transaction is the enforcement).
3. **Failure** — worker writes gate `FAILED` + `GK_E_*` code and marks the run `FAILED`. **No automatic retry of failures** — the operator retriggers from the vishwamitra UI after fixing data/config (owner requirement #7). The sweeper only rescues *crashed* work (expired leases), never *failed* work.
4. **Supersede** — creating a `FROM_START` run marks any non-terminal predecessor `SUPERSEDED` in the same transaction; superseded runRequestIds refuse all further transitions.

## 7. Data model (Firestore)

### 7.1 `gatekeeper_runs` (new; doc id = `runRequestId`)

```
{ schemaVersion, runRequestId, intakeId, stage3RunId, requestTimestamp, triggeredBy,
  judgeMode: "GATEKEEPER" | "SHADOW",
  mode: "FULL" | "FROM_GATE",
  state: "REQUESTED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "SUPERSEDED",
  supersededBy: runRequestId | null,
  configSnapshot: {
    rosterVersion: "v1",
    g1: { model: "modernbert-base-nli@v1", sha256, neutralMin, repeatMin, contraEscape, maxSeqTokens },
    g2: { model: "minicheck-deberta-l@v1",  sha256, supportMin, supportFloor, groundingMode },
    g3: { model: "deberta-mnli-fever-anli@v1", sha256, contraMin, agreementRule },
    g4: { llmModel: "<pinned lite row>", maxLlmPairs, thinkingBudget }
  },
  gates: { G1_NEUTRAL: { state, attempt, sweepAttempt, leaseOwner, leaseExpiresAt,
                          startedAt, endedAt, counters: {...}, errorCode, errorDetail },
           G2_CORROBORATION: {...}, G3_CONTRADICTION: {...}, G4_ESCALATION: {...} },
  totals: { pairsSeen, decidedByGates, escalatedLlm, escalatedHuman, shadowDisagreed, llmSpendUsd },
  createdAt, updatedAt }
```

Immutable history: never deleted, never purged; one doc per runRequestId forever (audit + the "which run produced this verdict" answer).

### 7.2 `stage3_edges` extensions (additive fields; existing docs untouched)

```
method            "ENSEMBLE"(existing) | "RULE" | "EMBEDDING"
                  | "GK_G1_NLI" | "GK_G2_GROUNDING" | "GK_G3_XCHECK" | "GK_G4_LLM"
judgeModel        artifact id — "modernbert-base-nli@v1" / "<lite pin>" (field exists today)
gkRunRequestId    which run wrote this verdict
stageScores       { g1: {entFwd, entBwd, neuFwd, neuBwd, conFwd, conBwd, ctxDelta?},
                    g2: {supportFwd, supportBwd, supportGrounded?},
                    g3: {conFwdB, conBwdB, agreed} }          // raw probabilities, full precision
escalationReason  "CONTRADICTION_SIGNAL" | "LOW_CONFIDENCE" | "STAGE_DISAGREEMENT"
                  | "CONTEXT_DISAGREEMENT" | "TRUNCATED" | "CAP_EXCEEDED" | "LLM_ERROR"
truncated         bool
shadow            { verdict, method, stageScores, gkRunRequestId }   // SHADOW mode writes ONLY this
```

`stageScores` is deliberately kept at full precision: it enables offline threshold recalibration without re-inference, powers the shadow disagreement report, and is the training corpus for the eventual fine-tune (VA‑79 becomes a weight swap).

### 7.3 Judge queue extensions (collection owned by vishwamitra/MATCH; additive)

```
tier          "CASCADE" | "LLM_TAIL" | "HUMAN"     // decided-tier routing
gate          current owner gate (G1..G4) or null when decided
leaseOwner / leaseExpiresAt                         // batch claiming by workers
decidedBy     method string (mirror of the edge, for queue-side queries)
```

### 7.4 Ownership boundary

Lakshmana **writes**: `gatekeeper_runs`, `stage3_edges` (gatekeeper methods + `shadow.*`), queue routing fields. Lakshmana **reads**: queue, claims/explanations (Neo4j RO). Vishwamitra remains the **sole writer of `Stage3Run`** — its existing poll observes `gatekeeper_runs.state == SUCCEEDED` and advances its own lifecycle; no callback, no notification (owner requirement #6). ASSEMBLE consumes verdicts regardless of `method`.

### 7.5 Composite indexes (land in vishwamitra's `firestore.indexes.json` — one database, one index file)

- queue: `(stage3RunId ASC, tier ASC, gate ASC, leaseExpiresAt ASC)`
- `gatekeeper_runs`: `(intakeId ASC, createdAt DESC)`

## 8. Gates

**v1 roster (off-the-shelf; every slot is model-agnostic — artifact + thresholds come from `configSnapshot`, so the LK‑5 bake-off can swap any row without code):**

| Gate | Model (artifact id) | Size (INT8) | Role |
| --- | --- | --- | --- |
| G1 | `tasksource/ModernBERT-base-nli` → `modernbert-base-nli@v1` | ~150 MB | 3-way NLI, both directions; NEUTRAL gate + REPEATS + contradiction pre-signal |
| G2 | `lytang/MiniCheck-DeBERTa-v3-Large` → `minicheck-deberta-l@v1` | ~435 MB | grounding/support checker → CORROBORATES |
| G3 | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` → `deberta-mnli-fever-anli@v1` | ~185 MB | second-family contradiction cross-check |
| G4 | Vertex **flash-lite** (pinned row, k=1, minimal thinking) | — | escalation tail + rationale prose; hard cap `maxLlmPairs` |

**Bake-off alternates** (mirrored alongside v1 so a NO-GO re-sweeps without another download — LK‑4). Checkpoint ids verified against the Hub 2026-07-19:

| Artifact | Checkpoint | Slot | Note |
| --- | --- | --- | --- |
| `nli-deberta-v3-small@v1` | `cross-encoder/nli-deberta-v3-small` | G1/G3 | cheap end of the curve |
| `ettinx-nli-s@v1` | `dleemiller/EttinX-nli-s` | G1/G3 | the "EttinX-s" row |
| `vitaminc-mnli@v1` | `tals/albert-xlarge-vitaminc-mnli` | G3 | contrastive-evidence training; contradiction recall is its point |
| `factcg-deberta-l@v1` | `yaxili96/FactCG-DeBERTa-v3-Large` | G2 | **checkpoint verified to exist** — closes the "if the checkpoint verifies" condition on O‑1; its scoring behaviour is still what LK‑5 measures |
| `hhem@v1` | `vectara/hallucination_evaluation_model` | G2 | ships as a `trust_remote_code` architecture, so `optimum`'s sequence-classification export cannot produce it. The prep script **refuses** it rather than quantizing the wrong graph; a hand-written export is deferred. Not a blocker — two other candidates cover G2. |

Every slot is model-agnostic, so adopting an alternate is a `configSnapshot` change, not a code change.

Numeric thresholds below are **proposals**; LK‑5 replay sets the real values and the owner signs them.

### G1_NEUTRAL — sees 100% of capped pairs

Hydrate texts by claim id (Neo4j RO); `withContext` pairs (§11.9 dual-eval) also fetch the explanation text and run a contexted variant. Run NLI forward + backward, batch 32.

```
decide_g1(pair, s):                                  # s = stageScores.g1
  if max(s.conFwd, s.conBwd) > contraEscape (0.02):  # THE escape hatch — non-negotiable
      forward(contradictionFlag = true); return      # may NEVER be neutral-discarded
  if withContext and bare_verdict != ctx_verdict:
      escalate(G4, CONTEXT_DISAGREEMENT); return
  if s.neuFwd >= neutralMin (0.95) and s.neuBwd >= neutralMin:
      decide(NEUTRAL, GK_G1_NLI); return
  if s.entFwd >= repeatMin (0.90) and s.entBwd >= repeatMin:
      decide(REPEATS, GK_G1_NLI); return
  forward()                                          # → G2
```

Counters: `seen, neutral, repeats, contraFlagged, forwarded, truncated, ctxDisagreed`.

### G2_CORROBORATION — sees the G1 leftovers (~15–20%)

Support scores in both directions; where the paired claim's source snippet is available, additionally score claim-vs-snippet (**grounding mode** — MiniCheck's native input shape, an evidence mode today's judge doesn't have).

```
decide_g2(pair, s):
  if max(s.supportFwd, s.supportBwd, s.supportGrounded?) >= supportMin (0.90):
      decide(CORROBORATES, GK_G2_GROUNDING,
             direction = argmax, confidence = calibrated(score)); return
  forward()                                          # everything undecided → G3
```

### G3_CONTRADICTION — sees contradiction-flagged + G2 leftovers (a few %)

Second-family NLI both directions; G1's contradiction logits are already in `stageScores` — **cross-architecture agreement replaces the old 5-vote self-consistency**.

```
decide_g3(pair, s1, s3):
  famA = max(s1.conFwd, s1.conBwd); famB = max(s3.conFwdB, s3.conBwdB)
  if famA >= contraMin (0.85) and famB >= contraMin:
      route(HUMAN, "CONTRADICTS-candidate"); return   # cascade NEVER finalizes CONTRADICTS;
                                                      # the human queue confirms, as today
  if famA < neutralConsensus and famB < neutralConsensus and both_neutral_lean:
      decide(NEUTRAL, GK_G3_XCHECK); return
  escalate(G4, STAGE_DISAGREEMENT)
```

### G4_ESCALATION — target ≤ 5%; hard cap `maxLlmPairs` (3,000)

Pinned flash-lite, one call per pair (k=1, minimal thinking), reusing the existing judge prompt family; returns verdict JSON + a 1–2 sentence rationale (prose lands exactly where humans will look). Parse failure / refusal / over-cap → `HUMAN` with `escalationReason`. In **SHADOW** mode G4 makes **no LLM calls** — pairs that would escalate are recorded `shadow.verdict = WOULD_ESCALATE_LLM` (comparing LLM to LLM would spend money to learn nothing).

### FINALIZE

Compute `totals`, set run `SUCCEEDED`, publish nothing. Vishwamitra's poll does the rest.

## 9. JudgeMode semantics

| Mode | Cascade | LLM ensemble | Verdict writer | Purpose |
| --- | --- | --- | --- | --- |
| `LLM` | not invoked | decides (k=3, thin — Workstream A dials) | vishwamitra | legacy / fallback; rollback target |
| `SHADOW` | full run, G4 suppressed | decides | vishwamitra (cascade → `shadow.*` only) | live disagreement report before cutover |
| `GATEKEEPER` | decides | **must not run** | lakshmana (+ human queue) | end state |

The mode is pinned into `Stage3Run.paramsSnapshot` at run start and read from there by **both** services — the split-brain guard: vishwamitra's JUDGE phase asserts `judgeMode == LLM` before consuming the queue and otherwise skips; a mode change never affects an in-flight run.

## 10. Re-runs, purge, retrigger

- **FROM_START** (new `runRequestId`, minted by vishwamitra): G1's first act — before any inference — is the purge: delete `stage3_edges` where `method IN (GK_*)` for this `stage3RunId`, reset queue routing fields to `{tier: CASCADE, gate: G1_NEUTRAL}`, clear leases. Idempotent (safe under redelivery: the purge re-run deletes nothing new). `ENSEMBLE` / `RULE` / `EMBEDDING` rows and all `shadow.*` history are untouched (D‑5).
- **FROM_GATE** (same `runRequestId`): valid only when that gate is `FAILED` (or lease-expired `RUNNING`); the gate returns to `PENDING` and its message is republished. Earlier gates' outputs are retained.
- **Supersede:** §6 rule 4. The vishwamitra run page always operates on `latest gatekeeper_runs by (intakeId, createdAt)`.

## 11. Failure handling & recovery

| Code | Meaning | Handling |
| --- | --- | --- |
| `GK_E_SCHEMA` | unknown `schemaVersion` / malformed payload | NACK → DLQ → alert; never guessed |
| `GK_E_MODEL_FETCH` | GCS artifact missing / sha mismatch | gate FAILED; operator fixes manifest, FROM_GATE |
| `GK_E_NEO4J_UNAVAILABLE` | hydrate failures beyond retry budget | gate FAILED (transient — usually just retrigger) |
| `GK_E_FIRESTORE_TXN` | claim/commit contention beyond retries | gate FAILED |
| `GK_E_VERTEX` | G4 quota/5xx beyond backoff | affected pairs → HUMAN with `LLM_ERROR`; gate still SUCCEEDED (partial-tail policy) |
| `GK_E_CAP_EXCEEDED` | G4 `maxLlmPairs` hit | remainder → HUMAN with `CAP_EXCEEDED`; run SUCCEEDED + warning counter |
| `GK_E_STUCK` | sweeps exhausted | run FAILED; runbook |

**Sweeper** (Cloud Scheduler → dispatcher `/sweep`, every 15 min, OIDC — the VA‑39 pattern): (a) `RUNNING` gates with expired leases → `PENDING` + republish (`sweepAttempt ≤ 3`, then FAILED); (b) `REQUESTED` runs with no claim after 30 min → republish G1; (c) DLQ depth surfaced to the alert channel. **The sweeper rescues crashes, never failures** — failures wait for the operator (owner requirement #7).

**Runbook skeleton** (full doc lands in `lakshmana-core/RUNBOOK.md`, LK‑12): symptom → `gatekeeper_runs` doc (which gate, `errorCode`, `attempt`) → worker logs filtered by `runRequestId` → action = fix data/config → retrigger from UI (FROM_GATE preferred) → DLQ replay command for poisoned messages.

## 12. Observability

- **Counters live on `gatekeeper_runs`** (§7.1) — the vishwamitra gate-progress panel polls this one doc; per-gate counts, durations, totals, `llmSpendUsd`.
- **Alerts** (lakshmana-infra, reusing the existing notification channel): DLQ depth > 0 · worker job execution failed · run in `RUNNING` > 2 h · `GK_E_CAP_EXCEEDED` observed · dispatcher 5xx rate.
- **Logs**: structured JSON with `runRequestId`/`gate` on every line; log-based metrics for gate durations and decided/escalated counts.

Three of the five alerts sit on native GCP metrics (DLQ depth, job execution result, Cloud Run 5xx). The other two are statements about *run state*, which only the gatekeeper knows, so they ride on log-based metrics — which makes these field names a contract the code must keep:

| Log metric | Filter | Emitted by |
| --- | --- | --- |
| `gatekeeper/run_overdue` | `jsonPayload.event="run_overdue"` | the sweeper, when a run has been `RUNNING` past the threshold (2 h) — LK‑11 |
| `gatekeeper/cap_exceeded` | `jsonPayload.errorCode="GK_E_CAP_EXCEEDED"` | G4, when `maxLlmPairs` is hit — LK‑10 |

Both metrics may be applied before the emitting code exists; they simply read zero.

## 13. Infra (`lakshmana-infra`) & IAM

**Inventory (all Terraform):** service accounts (`gatekeeper-worker-sa`, `gatekeeper-dispatcher-sa`, `gatekeeper-push-sa`); topic + push subscription (OIDC auth, 5-attempt DLQ policy) + DLQ topic/pull sub; model bucket `…-gatekeeper-models` (uniform access, versioned objects); Cloud Run **service** `gatekeeper-dispatcher` (1 vCPU / 512 MB, min 0, ingress = internal + push); Cloud Run **job** `gatekeeper-worker` (4 vCPU / 8 GB, task timeout 6 h, maxRetries 1 — verdict writes are idempotent upserts by pair key); Cloud Scheduler `gatekeeper-sweep` (15 min); Secret Manager entry `gatekeeper-neo4j-ro`; VPC-connector reference via vishwamitra-infra remote-state data source; alert policies + channel reuse; TF backend = existing state bucket, prefix `lakshmana/`.

**IAM matrix:**

| Principal | Resource | Role |
| --- | --- | --- |
| vishwamitra app SA | topic `gatekeeper-requests` | `roles/pubsub.publisher` (lands in **vishwamitra-infra** — the VA-side ticket) |
| `gatekeeper-worker-sa` | Firestore | `roles/datastore.user` |
| `gatekeeper-worker-sa` | model bucket | `roles/storage.objectViewer` |
| `gatekeeper-worker-sa` | topic (chain) | `roles/pubsub.publisher` |
| `gatekeeper-worker-sa` | Vertex (G4) | `roles/aiplatform.user` |
| `gatekeeper-worker-sa` | `gatekeeper-neo4j-ro` secret | `roles/secretmanager.secretAccessor` |
| `gatekeeper-dispatcher-sa` | worker job | `roles/run.developer` (executes job; scoped to the job) |
| `gatekeeper-dispatcher-sa` | Firestore | `roles/datastore.user` (claim transactions) |
| `gatekeeper-push-sa` / scheduler SA | dispatcher service | `roles/run.invoker` |

Resource names as built: push subscription `gatekeeper-requests-push`, DLQ inspection subscription `gatekeeper-requests-dlq-pull` (pull, not push — a poisoned message is replayed by an operator from the runbook, never automatically). The Pub/Sub **service agent** additionally holds `pubsub.publisher` on the DLQ topic and `pubsub.subscriber` on the push subscription; without both the dead-letter policy is configured but inert. The dispatcher also holds `iam.serviceAccountUser` on `gatekeeper-worker-sa`, because starting a job means acting as the job's identity.

**Artifact pin chain.** The `sha256` in each gate's config is the digest of the artifact's **`manifest.json` canonical bytes** (`json.dumps(sort_keys=True, separators=(",",":"))` + `\n`), and the manifest in turn carries a `sha256` + byte count for each of `model.onnx` / `tokenizer.json` / `config.json`. One short value per gate therefore covers hundreds of megabytes transitively, and it is small enough to live in `configSnapshot` — so an auditor reading a six-month-old run doc can prove which bytes produced its verdicts. The manifest also pins `sourceCheckpoint` + `sourceRevision` (the upstream commit sha, never a branch name). Any mismatch at any link is `GK_E_MODEL_FETCH`. An empty config `sha256` means *unpinned*: the artifact still loads (config ships blank until LK‑4 has mirrored anything) but every load logs a WARNING carrying the observed digest.

**Loader source modes.** `GATEKEEPER_MODELS_BUCKET` is the deployed path. `GATEKEEPER_MODELS_LOCAL_DIR` points the same loader at a directory laid out identically to the bucket, and wins when set — this is how the LK‑5 replay bake-off runs the *production* loader before anything has been mirrored to GCS. Verification is identical in both modes; a mode that skipped digest checks would be measuring code that never ships. Terraform leaves the local override empty, so a deployed worker cannot silently fall back to a directory that happens to exist.

Neo4j side (manual, owner): create role `gatekeeper_ro` (MATCH read on claims/explanations, zero writes) + user bound to the SM secret — cypher provided in LK‑2's description and in `lakshmana-infra/README.md`.

## 14. Integration contract (vishwamitra side — single VA‑27 ticket)

1. **IAM**: publisher binding above (vishwamitra-infra).
2. **Config/UI**: `JudgeMode` selector on the Stage 3 configuration surface (`LLM | GATEKEEPER | SHADOW`), pinned into `paramsSnapshot` at run start; JUDGE-phase split-brain guard (§9).
3. **`GatekeeperClient` (Kotlin)** — design sketch, signatures only:

```
interface GatekeeperClient {
  fun trigger(intakeId, stage3RunId, mode: FULL, judgeMode): RunRequestId   // mints UUID, publishes G1,
                                                                            // creates gatekeeper_runs doc? NO —
                                                                            // doc is created by dispatcher on first claim;
                                                                            // client persists {runRequestId, publishState} on Stage3Run
  fun retrigger(runRequestId, fromGate: Gate?): RunRequestId               // null = FROM_START (new UUID)
  fun status(intakeId): GatekeeperRunView                                   // poll gatekeeper_runs (latest by intakeId)
}
```

Publish failures: bounded retry with backoff; terminal publish failure surfaces on the run page as `PUBLISH_FAILED` with a manual retry button (no silent loss). 4. **Run-page gate panel**: per-gate chips (state/counters/durations from `gatekeeper_runs`), failure card (`errorCode` + detail), retrigger buttons (FROM_START / failed-gate), shadow-disagreement summary line in SHADOW mode.

## 15. Replay & cutover plan

1. **LK‑5 replay (zero LLM spend)** — export the 12,208 ensemble-labeled pairs (+ votes, golden pairs) from the Firestore backup via the emulator-restore flow into JSONL; run every roster candidate + alternates per gate through the *production gate code*; sweep thresholds. Report: per-verdict P/R, NEUTRAL precision/coverage, **CONTRADICTS recall vs golden pairs**, calibration (ECE), projected G4 volume, wall-clock.
2. **GO bar (owner signs at the report):** NEUTRAL precision ≥ 0.95 at ≥ 60% coverage · CONTRADICTS recall ≥ 95% of the ensemble's own recall on golden pairs · projected G4 ≤ 5%. NO-GO → swap roster rows and re-run replay (architecture unchanged).
3. **SHADOW** on the next real run → disagreement report must match replay-predicted rates.
4. **Flip** to `GATEKEEPER`. Rollback = set `JudgeMode = LLM` (one config value; the ensemble path is untouched by this entire design).

## 16. Worked example — the 209-claim reference subject (12,208 pairs)

| Step | In | Decided here | Routed on |
| --- | --- | --- | --- |
| G1_NEUTRAL | 12,208 | 10,450 NEUTRAL + 160 REPEATS | 1,598 → G2 (140 of them contradiction-flagged) |
| G2_CORROBORATION | 1,598 | 1,050 CORROBORATES | 548 → G3 |
| G3_CONTRADICTION | 548 | 210 NEUTRAL (two-family consensus) · 60 CONTRADICTS-candidates → **HUMAN** | 278 → G4 |
| G4_ESCALATION | 278 | 265 by flash-lite (~$0.30) | 13 → HUMAN |
| **Totals** | | **encoders 97.2% · LLM 2.2% · human 0.6% (73 pairs)** | |

Cost ≈ **$0.30 LLM + ~$0.50 CPU ≈ $0.80** vs $48.83 today (61×). Wall-clock ≈ 25–35 min (G1 ~10 min at ~40 pairs/s dual-pass, G2 ~5 min, G3+G4+finalize ~10 min). A 1,000-claim intake under the B2 cap (~35–40k pairs) scales linearly: ~$2–4, ~1.5–2.5 h single worker — far inside the 24 h job ceiling, parallelizable via N job tasks if ever needed.

**Figure 4 — the cascade funnel on the reference subject** *(mermaid source — wrap with the diagram plugin):*

```
flowchart TD
    Q["judge queue<br/>12,208 pairs — 209-claim reference subject"] --> G1{"G1_NEUTRAL<br/>ModernBERT-base-nli, both directions"}
    G1 -- "10,450 NEUTRAL + 160 REPEATS" --> D1["decided — GK_G1_NLI"]
    G1 -- "1,598 forward<br/>(140 contradiction-flagged: escape hatch)" --> G2{"G2_CORROBORATION<br/>MiniCheck-DeBERTa-L"}
    G2 -- "1,050 CORROBORATES" --> D2["decided — GK_G2_GROUNDING"]
    G2 -- "548 forward" --> G3{"G3_CONTRADICTION<br/>two-family cross-check"}
    G3 -- "210 NEUTRAL (two-family consensus)" --> D3["decided — GK_G3_XCHECK"]
    G3 -- "60 CONTRADICTS-candidates" --> H["human queue<br/>confirm/dismiss, as today"]
    G3 -- "278 disagreement" --> G4{"G4_ESCALATION<br/>flash-lite, k=1, cap 3,000"}
    G4 -- "265 verdicts, about $0.30" --> D4["decided — GK_G4_LLM"]
    G4 -- "13 unresolvable" --> H
    D1 & D2 & D3 & D4 --> FIN["FINALIZE<br/>encoders 97.2% · LLM 2.2% · human 0.6%<br/>about $0.80 vs $48.83 today"]
    H --> FIN
```

## 17. Open items

| # | Item | Owner | Lands |
| --- | --- | --- | --- |
| O‑1 | Real thresholds + final roster (~~may swap G2 → FactCG if checkpoint verifies~~ — **checkpoint verified 2026-07-19**, `yaxili96/FactCG-DeBERTa-v3-Large`; the swap is now purely an LK‑5 numbers question) | LK‑5 report → owner sign-off | configSnapshot defaults |
| O‑2 | CONTRADICTS-recall tolerance number | owner | §15 GO bar |
| O‑3 | G4 prompt: exact reuse/trim of the existing judge prompt family | LK‑10 | prompt registry |
| O‑4 | ~~Region pin + notification-channel id~~ **CLOSED 2026-07-19** — region `asia-southeast1` (vishwamitra-infra's default, and where Firestore and the buckets already are); **no channel minted**, the alert policies read `notification_channel_ids` out of vishwamitra-infra's state via the remote-state data source | owner (infra vars) | lakshmana-infra tfvars |
| O‑5 | `PLAN-stage3-cost-cut.md` Workstream C: mark superseded by this LLD | ~~next vishwamitra session~~ **DONE 2026-07-19** (plan doc + cost wiki v1.3) | plan doc |

## References

Stage 3 LLD (249528322) §11 · Cost wiki (253001729) §9–10 · `PLAN-stage3-cost-cut.md` · LLM-AggreFact leaderboard · MiniCheck (Liyan06/MiniCheck) · tasksource/ModernBERT-base-nli · MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli · Vectara HHEM (alternate) · FactCG (arXiv 2501.17144, G2 alternate)
