# session00 — Stage 3 Workstream A+B dials (VA-77)

> ⚠️ **REPO: vishwamitra-core** — executed there with its own handoff ritual; this file lives here only so the plan is complete.
> Blocked by: nothing. Executes before (or in parallel with) everything else — it cuts real spend now and B2's cap controls pair volume for the gatekeeper too.
> Authority: `vishwamitra-core/PLAN-stage3-cost-cut.md` Workstreams A + B (still live; only Workstream C is superseded).

## Scope

- **VA-77** — Stage 3 judge cost dials (Workstreams A + B).

## Objectives

1. **A-offline validation (zero spend, first):** simulate k=3 by subsampling the cached 5-vote maps → verdict-flip rate; count the 0.90–0.93 sim band + spot-check 20 pairs.
2. **A config flip** (`ensemble-k` 5→3, `sim-auto-repeat` 0.93→0.90) — only on the owner's go at the validation numbers.
3. **B0 offline cap replay** (acceptance gate): cap=40 must retain ≥95% of CORROBORATES-majority pairs and 100% of published-fact contributing pairs.
4. **B PR (one PR):** B1 thinking budget → `app.stage3.judge-thinking-budget` (default 512, env `STAGE3_JUDGE_THINKING_BUDGET`) · B2 per-claim candidate cap 40 in `ClaimMatcher` (human-asserted + CONTRADICTS-lane quota exempt) · B3 entity-IDF gate on co-mention · counters `pairsCapped`, `pairsIdfGated`.

## Not in scope

VA-77's "adaptive ensemble early-exit" dial — dropped by design (k=3 covers it); note on the ticket at completion.

## Definition of done

- [ ] Offline validation + B0 replay reports delivered; owner signed the config flip
- [ ] B PR complete, golden-pair eval shows no F1 regression; VA-77 → In Review
- [ ] Expected end state: Stage 3 ≈ $45–60 per 1,000-claim intake (from $190–730)
- [ ] Handoff ritual in vishwamitra-core; delete this file
