# session07 — vishwamitra integration (VA-106)

> ⚠️ **REPO: vishwamitra-core** — executed there with its own handoff ritual; file lives here for plan completeness.
> Blocked by: session01 (contract fixtures). **May be pulled forward** any time after session01; must complete before session08.
> LLD: §14 integration contract · §7.5 indexes · §9 split-brain guard — page 255688733.

## Scope

- **VA-106** — [BE][FE] Gatekeeper integration — publish IAM, JudgeMode config UI, GatekeeperClient + run-page gate panel

## Objectives

1. **vishwamitra-infra**: `roles/pubsub.publisher` for the app SA on `gatekeeper-requests` (owner applies).
2. `JudgeMode` selector (`LLM | GATEKEEPER | SHADOW`) on the Stage 3 config surface; pinned into `Stage3Run.paramsSnapshot` at run start; **JUDGE-phase split-brain guard**: the LLM judge loop asserts `judgeMode == LLM`, otherwise skips the queue entirely.
3. `GatekeeperClient` (Kotlin): mint runRequestId (UUIDv4), build payload (schemaVersion 1, camelCase JSON), **contract test pinned to lakshmana's golden fixtures byte-for-byte**, publish with bounded retry/backoff, terminal failure → visible `PUBLISH_FAILED` + manual retry; status poll = latest `gatekeeper_runs` by intakeId; retrigger FROM_START (new UUID) / FROM_GATE.
4. Run-page gate panel: per-gate chips (state/counters/durations), failure card (`errorCode` + detail), retrigger buttons, shadow-disagreement summary line.
5. `firestore.indexes.json`: queue `(stage3RunId, tier, gate, leaseExpiresAt)` + `gatekeeper_runs (intakeId, createdAt DESC)` composites.
6. Stage 3 LLD §11: add the JudgeMode/gatekeeper note via the REST-splice pipeline (the deferred doc item from 2026-07-19).

## Definition of done

- [ ] Contract test green against the pinned fixtures
- [ ] `judgeMode != LLM` ⇒ ensemble provably never consumes the queue (test)
- [ ] Panel renders a full lifecycle incl. FAILED gate + successful FROM_GATE retrigger
- [ ] VA-106 → In Review; Stage 3 LLD spliced; handoff ritual in vishwamitra-core; delete this file
