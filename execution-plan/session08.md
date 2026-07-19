# session08 — observability + SHADOW run + cutover (VA-104, VA-105)

> Repo: lakshmana-core (+ live GCP) · Blocked by: session06 + session07 + owner deploys done.
> Owner-heavy session: live shadow run needs per-instance confirmation; the flip is an owner sign-off.
> LLD: §11 recovery · §12 observability · §15 cutover — page 255688733.

## Scope

- **VA-104** — [BE][DO] Observability & recovery — counters, alert wiring, runbook
- **VA-105** — [VERIFY] Shadow run + cutover — disagreement report, flip checklist

## Objectives

1. Counter completeness across all gates; per-gate durations; FINALIZE totals invariant on the full emulator corpus.
2. Log-based metrics (gate duration, decided/escalated, `GK_E_*`); fire all four alerts once via fault injection (DLQ depth, job failure, stuck run, cap exceeded).
3. `RUNBOOK.md`: symptom → run-doc field → log filter → action tables; DLQ inspect/replay commands. Terse imperative.
4. **SHADOW run on a real intake** (owner-confirmed; ensemble decides ≈ $10–30; cascade writes `shadow.*`, G4 suppressed) → disagreement report per verdict class vs session03 replay-predicted rates; CONTRADICTS misses on present golden pairs must be 0.
5. Cutover checklist: replay GO ✓ → shadow agreement ✓ → **owner signs** → `JudgeMode = GATEKEEPER` next run → first gatekeeper-decided run reviewed (scores within tolerance, publish contract unaffected) → rollback rehearsal (`JudgeMode = LLM`).

## Definition of done

- [ ] All four alerts observed firing; runbook walked against a deliberately broken run
- [ ] Shadow disagreement ≤ replay-predicted; report archived on the run doc / eval page
- [ ] Cutover + rollback both demonstrated; owner signed the flip
- [ ] VA-104, VA-105 → In Review; LLD §15 marked executed (page + mirror); handoff ritual; delete this file
- [ ] Epic VA-92 ready for owner close-out; VA-79 note: fine-tune is now a weight swap on this component
