# session04 — dispatcher + G1 (VA-98, VA-99)

> Repo: lakshmana-core · Blocked by: nothing. ~~session03 GO (thresholds signed)~~ — **the GO gate was removed by owner directive 2026-07-19**; build against §8's v1 roster defaults, which are already the shipped config. Calibration lands later as a config change (DEFERRED-LIVE 17–19). session02 infra apply is deferred-live and does not block local work.
> **Read the model from the run, not from config:** `GateBinding.from_snapshot(run.config_snapshot, gate)` — see the freeze invariant in LLD §8 and `tests/test_freeze_invariant.py`.
> LLD: §4 architecture · §5 protocol · §6 claim rules · §8 G1 · §10 purge — page 255688733.

## Scope

- **VA-98** — [BE] Dispatcher — push endpoint, OIDC, transactional claim, Run Job execution, /sweep
- **VA-99** — [BE] G1_NEUTRAL gate — hydrate, dual-direction NLI, decide/forward, chain publish

## Objectives

1. Dispatcher `/pubsub/push`: OIDC verify, schema validate (unknown → DLQ), transactional claim (session01 machinery), Run Jobs API execution `{runRequestId, gate}`, fast ACK; claim losers ACK+drop with `GK_DUP`.
2. First-claim path creates the run doc + freezes `configSnapshot`.
3. `/sweep`: expired-lease rescue (≤3 sweeps → `GK_E_STUCK`), unclaimed-run republish, DLQ depth check. Crashes only, never FAILED gates. `/healthz`.
4. G1 worker loop: queue batch claim (lease), Neo4j RO hydrate (withContext pairs + explanation text), ModernBERT dual-direction NLI (batch 32), decide per LLD §8 — **contradiction escape-hatch is non-negotiable**; withContext bare-vs-ctx disagreement → G4 `CONTEXT_DISAGREEMENT`.
5. Writes: `stage3_edges` (method `GK_G1_NLI`, `stageScores.g1` full precision, `gkRunRequestId`, `truncated`), queue routing, counters; SHADOW variant (`shadow.*` only); FROM_START purge as G1's first act (idempotent); commit-then-publish G2.
6. Local e2e: Pub/Sub emulator (or direct dispatcher POST) + Firestore emulator + local Neo4j — G1 end-to-end, G2 message observed.

## Definition of done

- [ ] Duplicate/stale no-op integration tests; sweep rescues a synthetic expired lease exactly once
- [ ] Escape-hatch property test: no contradiction-signal pair is ever neutral-discarded
- [ ] Replay parity: gate-loop output ≡ the VA-97 harness (`gatekeeper/replay/`) on the same corpus + thresholds — both must drive `gatekeeper/gates/decisions.py`, so parity is a wiring check, not a numbers check; crash mid-batch resumes cleanly
- [ ] VA-98, VA-99 → In Review; handoff ritual; delete this file
