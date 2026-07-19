# session06 — G4 + FINALIZE + purge/retrigger (VA-102, VA-103)

> Repo: lakshmana-core · Blocked by: session05.
> LLD: §8 G4/FINALIZE · §10 re-runs · §11 partial-tail policy — page 255688733. Decision D-1: the tail lives here.

## Scope

- **VA-102** — [BE] G4_ESCALATION + FINALIZE — pinned flash-lite tail, caps, run completion
- **VA-103** — [BE] Purge & retrigger semantics — FROM_START / FROM_GATE / supersede

## Objectives

1. G4: Vertex flash-lite client (ADC, region + model row from `configSnapshot.g4`), one call per pair, k=1, minimal thinking, existing judge prompt family (LLD O-3 — resolve the exact prompt reuse here), JSON verdict + 1–2 sentence rationale onto the edge doc.
2. Caps + failure policy: `maxLlmPairs` (3000) → over-cap to HUMAN `CAP_EXCEEDED` (run still SUCCEEDED + warning counter); Vertex errors beyond backoff → HUMAN `LLM_ERROR`; `llmSpendUsd` accounting.
3. SHADOW: **zero LLM calls** — `shadow.verdict = WOULD_ESCALATE_LLM`.
4. FINALIZE: totals reconciliation invariant (decided + escalated + human == pairsSeen), run → SUCCEEDED, publish nothing.
5. Purge/retrigger hardening: FROM_START purge idempotency under redelivery; FROM_GATE precondition (FAILED or lease-expired only); superseded-run guard; **ENSEMBLE-cache-untouched invariant test** (D-5).
6. Full chain e2e on emulator corpus with a dry-run LLM double: G1 → FINALIZE green, three consecutive re-runs leave history intact.

## Definition of done

- [ ] Fault-injection tests for cap/error paths with correct `escalationReason`
- [ ] Spend counter within 5% of actual billing on a small real test batch (owner-confirmed live calls)
- [ ] Double FROM_START, rejected FROM_GATE-on-healthy-gate, redelivered retrigger — all covered by tests
- [ ] VA-102, VA-103 → In Review; handoff ritual; delete this file

## Owner follow-ups

- Deploy dispatcher + worker images; verify the sweep scheduler fires in the live project
