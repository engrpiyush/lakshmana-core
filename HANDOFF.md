# HANDOFF — after session06 (VA-102 G4_ESCALATION + FINALIZE · VA-103 purge & retrigger)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: mirror at **v1.7**, Confluence page 255688733 still at **v1.6**
> Session file: `execution-plan/session06.md` — **deleted**, the session is complete.
> VA-102 and VA-103 → **NOT transitioned** (see "Two things the owner has to do" below).
> Mode: **FINISH-ALL-CODE**. Nothing is parked. Five new DEFERRED-LIVE items (28–32), one correction (27).
> Tests: **437 passed, 1 skipped** (was 375 · +62).

## The short version

**The cascade is complete.** One Pub/Sub push now drives G1 → G2 → G3 → G4 → FINALIZE, and the
run doc ends `SUCCEEDED` with totals that reconcile. `test_the_full_chain_reaches_finalize_and_the_totals_reconcile`
feeds every published message back through the real dispatcher, so four claim transactions are
exercised, and the walk terminates on its own because G4 publishes nothing — that termination is
part of what the test asserts.

**G4** is the only thing in this system that spends money, and nearly every design choice in it
follows from that. It runs one call per pair against a pinned flash-lite row, reusing
vishwamitra's judge prompt with the rubric resolved live from the shared `extraction_prompts`
registry. Its cap is durable across executions; a Vertex error costs its pair and never the gate;
an answer nobody can parse is an error and never a verdict; and **the live door is off by
default**, so the whole cascade runs locally, cap and error paths included, without spending
anything.

**FINALIZE** recomputes `totals` from the queue rather than summing gate counters, and refuses to
mark a run `SUCCEEDED` if any pair is stranded at a committed gate — the one failure mode that
would otherwise tell vishwamitra to consume a verdict set with a hole in it, silently.

**VA-103** proves the operator stories end to end: double FROM_START, FROM_GATE refused on a
healthy gate in each of its three shapes, redelivered retrigger, and D‑5 — which turned out to be
stronger than the LLD claimed.

## Two things the owner has to do

1. **Push the LLD.** The mirror is at v1.7; page 255688733 is still at v1.6. This session ran
   headless with the Atlassian MCP unauthorized, so the whole-body push could not be made.
   CLAUDE.md says the page and the mirror move together — they are apart *now*, and the mirror is
   ahead. A banner at the top of `lakshmana-gatekeeper-lld-wiki.md` says so; delete it after the
   push. **Until then, read the file, not the page.**
2. **Transition VA-102 and VA-103 to In Review.** Same reason. Both are code-complete and green.

Authorize the connector (claude.ai connector settings, or `/mcp` in an interactive session) and a
short rerun clears both.

## What was built

### VA-102 — G4_ESCALATION + FINALIZE

| File | What it is |
| --- | --- |
| `gatekeeper/gates/prompts.py` | The G4 prompt (**closes O‑3**), rubric resolution, hostile parser |
| `gatekeeper/clients/vertex.py` | ADC + regional `generateContent`, bounded backoff, spend, dry-run double |
| `gatekeeper/gates/g4.py` | The gate: cap, per-pair error policy, SHADOW suppression, counters |
| `gatekeeper/worker/finalize.py` | The tally, the reconciliation invariant, `totals` |

**O‑3, resolved.** The prompt is a *trim* of `Judging.kt`'s `judgePrompt`, not a rewrite. Kept
verbatim: the task line, the §14 data-hardening clause, the card shape and field order, the
strict-JSON contract keyed by 1-based index, the relation vocabulary, `temporalNote`, and the
`explanationRelevant` question. The **rubric is resolved live** from `extraction_prompts/STAGE3_JUDGE`
(read-only) with a vendored fallback — mirroring `ExtractionPromptService.resolveKey`, because in
SHADOW mode the two judges must be answering the same question and a rubric living only in
lakshmana's source would drift the instant an admin edited the shared one.

Three trims, each with a reason:

- **batching** → one pair, since §8 pins one call per pair;
- **`flipPresentation`** → dropped. It is the ensemble's position-bias control *across k samples*,
  and k=1 has nothing to alternate over — flipping would relabel the bias, not cancel it. Worth
  saying plainly: **this is a real accuracy difference between G4 and today's ensemble**, accepted
  because G4 sees ~5% of pairs and the alternative is paying k=3 for the whole tail;
- **`sharedEntities`** → dropped, because lakshmana's queue read does not carry it. A hint, not
  evidence; omitting costs a shortcut, faking would be a lie. Restoring it is a hydrate change.

**The cap is durable.** `maxLlmPairs` is checked against how many pairs the *run* has already
settled with `GK_G4_LLM`, counted off `gatekeeper_pairs` — not a counter that restarts at zero on
every sweeper rescue. A gate that crashed at 2,999 calls and resumed on a fresh counter would
spend the budget twice, and the run doc would only reveal it to someone adding two attempts up by
hand. `test_the_cap_is_durable_across_a_resumed_gate` is the regression.

**FINALIZE recomputes from the queue.** Counters are per-execution; a rescued gate writes a fresh
set, so summing them across a swept run double-counts exactly the pairs the sweep touched. The
queue has one row per pair by construction, which is what makes
`decidedByGates + escalatedLlm + escalatedHuman == pairsSeen` a real check rather than a circular
one. Classification is **tier first, method second** — a G3 candidate and a G4 failure both leave
`decidedBy` unset, and reading it first would put them in no bucket at all.

A reconciliation failure fails the **run**, not the gate: the gate genuinely succeeded, and what
broke is a run-level invariant. That needed a new `store.fail_run` and two new run-doc fields
(`errorCode`/`errorDetail`) — pushing G4 back to FAILED would misreport which step broke *and*
make a FROM_GATE retrigger re-run the whole expensive tail to fix a bookkeeping problem.

### VA-103 — purge & retrigger

`tests/test_retrigger.py` (14 tests). Double FROM_START with the purge in between; purge and queue
reset each idempotent on their own; FROM_GATE refused on a `PENDING` gate, on a live-leased
`RUNNING` gate, and (earliest of all) on a gate whose predecessor never succeeded; FROM_GATE
*admitted* on a lease-expired gate; redelivered retrigger losing to its own first copy; superseded
run refusing every mode.

**D‑5 is stronger than the LLD claimed.** The purge's `stage3RunId` filter and its `method`
re-check were documented as belt and braces. Reading vishwamitra's `Stage3EdgeVerdict` shows the
first is absolute: an ensemble row carries **neither `stage3RunId` nor `method`**, so the purge
query cannot return one *even in principle* — the field it filters on does not exist on those
documents. Both guards are now tested, the second against a row built to sit inside the query's
reach. LLD §10 updated to say so.

## Decisions worth knowing

- **`llmModel` default is now `gemini-2.5-flash-lite`**, replacing `pending-lite-pin`, which no
  live call could resolve. Not a new product decision — §8 had already chosen flash-lite; this
  makes the row concrete. Vishwamitra's `providers/stage3-judge` pin is deliberately **not** read:
  the judges are meant to be independently pinnable, and a SHADOW comparison where one silently
  follows the other's pin measures nothing. Owner confirms both in DEFERRED 29.
- **`shadowDisagreed` ships as 0 and VA-105 owns the real number** (DEFERRED 32). In GATEKEEPER
  that is correct — the cascade's verdict *is* the verdict. In SHADOW the honest number cannot be
  computed at finalize time: the ensemble runs concurrently and may not have judged a pair yet, so
  any count taken then races it with a systematic bias toward zero. Do not read the zero as
  evidence of agreement.
- **No `stageScores.g4`, and no empty slot either.** §7.2's map is the encoders' calibration
  corpus; an LLM's self-reported confidence is not a comparable number. The tail contributes prose
  (`rationale`, `temporalNote`). `EdgeVerdict` now omits the slot entirely when a gate computed
  nothing — same reasoning as G2 declining to write a support score for a pair it never judged.
- **`GATEKEEPER_G4_LIVE_CALLS` defaults false.** Hard rule 4 in code: `client_for` returns the
  dry-run double unless told otherwise, so no deployment reaches Vertex by accident and the full
  cascade is exercisable on a laptop.
- **`thinkingBudget` vs `thinkingLevel`** — Gemini 3.x swapped the knob and *silently ignores the
  wrong one*, so a pin bump to a 3.x row would quietly restore full thinking and multiply the
  bill. `thinking_config()` ports `GeminiThinking.config`. Unit-covered, never seen a real 3.x
  response (DEFERRED 30).

## State

- **Tests: 437 passed, 1 skipped.** The skip is the real-Neo4j cypher test (DEFERRED 27).
- **Lint/format clean**; both images build; contract fixtures byte-identical.
- Local-stack smoke: the real `worker.main()` drove **G4 + FINALIZE** against the emulator →
  `exit=0`, run `SUCCEEDED`. G1–G3 were settled through the store for that smoke on purpose —
  each of them ends by publishing to *real* Pub/Sub, and a live publish is owner-gated.
- `gatekeeper.gates.g4.live-calls` has never been on. **Vertex has never been called.**

**DEFERRED-LIVE 27 corrected.** It told the owner to put the Neo4j password in
`.claude/settings.local.json` "(gitignored)". That file is **tracked** — the fix as written would
commit a container credential. Gitignore it first, or export the variable from the shell. I
declined to write it and verified the path another way: a scratch smoke drove `worker.main()`
against the live container read-only with the variable set, and G1's hydrate cypher ran clean.

## What is next

`execution-plan/session07.md` (**REPO: vishwamitra-core** — VA-106 integration, executed over
there) and `execution-plan/session08.md` (VA-104 alerts/runbook, VA-105 SHADOW + cutover). The
`/next-session` guard says session08 must not start while session07 exists.

All lakshmana *code* for the cascade is now written. What remains here is session08's operational
work; everything else is the single DEFERRED-LIVE round.
