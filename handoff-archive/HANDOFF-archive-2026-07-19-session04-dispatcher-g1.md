# HANDOFF — after session04 (VA-98 dispatcher · VA-99 G1_NEUTRAL)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: page 255688733 **v1.5** (mirror synced, page version 6)
> Session file: `execution-plan/session04.md` — **deleted**, the session is complete.
> VA-98 and VA-99 → **In Review**.
> Mode: **FINISH-ALL-CODE**. Nothing is parked. Five new DEFERRED-LIVE items (20–24).
> Tests: **354 passed, 1 skipped** (the skip is the `neo4j`-marked cypher smoke; see below).

## The short version

The cascade now runs end to end for its first gate. A Pub/Sub push reaches the dispatcher,
which authenticates the caller, wins the claim transaction, and starts a worker; the worker
purges, seeds its queue from the graph, scores both NLI directions in batches, applies
`decide_g1`, writes verdicts, and publishes G2. A scheduled `/sweep` rescues anything that
crashes on the way. All of it is proven against the Firestore emulator with the graph and
the ONNX session stubbed — the two things a laptop cannot afford and the two things a
deferred-live item covers.

**The one thing to read before anything else is the §7.3 correction below.** It is a real
divergence between the LLD and the system it describes, I resolved it in the only way the
hard rules allow, and it is the piece most worth your disagreement if you have any.

## The §7.3 correction — the judge queue is not where the LLD said it was

LLD §7.3 described the judge queue as a **Firestore collection owned by vishwamitra/MATCH**
that lakshmana would add `tier` / `gate` / `leaseOwner` / `decidedBy` fields to, and §7.4
listed "queue routing fields" among lakshmana's writes.

It is not a Firestore collection. In vishwamitra the judge queue is a **Neo4j
relationship** — `(:Claim)-[q:JUDGE_QUEUED {rank, blockScore, sources, humanAsserted,
withContext, status}]->(:Claim)` — written wholesale by MATCH in
`Stage3GraphRepository.applyMatchOutcome`, with `q.status` as the JUDGE phase's own cursor.

That collides head-on with **D-8**: lakshmana's graph access is read-only by construction
(RBAC user, `READ_ACCESS` sessions). The routing fields §7.3 asks for cannot be written
where §7.3 puts them. Two constraints, both load-bearing, and only one design satisfies
both.

**What I did:** the routing lives in `gatekeeper_pairs`, a collection lakshmana owns
outright, seeded from the graph as G1's second act. Doc id `{stage3RunId}|{low}|{high}`,
so seeding is an idempotent upsert. Every property §7.3 actually wanted survives — per-pair
tier, current gate, lease-based batch claiming, `decidedBy`, plus `contradictionFlag` so
G1's escape hatch reaches G3 across two separately-scheduled executions. Two things the
literal reading would have broken are preserved: the graph keeps its single writer, and
vishwamitra's queue stays exactly as MATCH left it — so a rollback to `JudgeMode = LLM`
needs nothing undone in the graph.

I treated this as an implementation decision forced by the LLD's own invariants rather than
a product decision, so I did not park it. **If you disagree, this is the thing to say so
about** — it is one collection and one module (`gatekeeper/worker/queue.py`), and the gates
above it do not care where routing lives. LLD §7.3/§7.4/§7.5 are rewritten, O‑8 is filed,
and DEFERRED-LIVE 23 flags that a new collection now exists in the shared database.

## What was built

**VA-98 — dispatcher**

- `dispatcher/oidc.py` — verifies the push/scheduler bearer token: signature via
  `google.oauth2.id_token`, then the service-account allow-list. **Fails closed** (on by
  default), rejects with **401** rather than 400 — a well-formed message from an
  unacceptable caller is not a schema problem, and `GK_E_SCHEMA`/400 stays reserved for
  payloads we cannot parse. An empty allow-list accepts any Google identity and warns on
  every request; that is the pre-Terraform default, never the deployed one (DEFERRED-LIVE 21).
- `dispatcher/jobs.py` — `CloudRunJobLauncher` (`run_v2.JobsClient.run_job`) passing the run
  id, gate and lease owner as **container overrides**. Opt-in via
  `gatekeeper.worker.execute-jobs` so a local dispatcher cannot reach the Run Admin API by
  accident. Started and not awaited: a gate takes minutes, the push wants seconds.
- `dispatcher/sweep.py` + `/sweep` — three passes: expired leases → `PENDING` + republish
  (`sweepAttempt ≤ 3`, then `GK_E_STUCK`); idle runs → republish the actionable gate; DLQ
  depth. Emits the `event="run_overdue"` line LK‑11's log metric filters on.
- `clients/pubsub.py` — the chain publisher (synchronous: a worker that exited with the next
  gate still in a client-side batch would stall the run until a sweep noticed).
- `runs/store.py` — `active_runs()` and `rescue_gate()`, the transactional half of the sweep.

**VA-99 — G1_NEUTRAL**

- `worker/hydrate.py` — read-only cypher: the queue, claim text, explanation text. Reads the
  **whole** queue rather than filtering on `q.status = 'QUEUED'`: that flag is vishwamitra's
  JUDGE cursor, and in GATEKEEPER mode its JUDGE phase never runs, so it would be whatever
  the last LLM run left behind. Lakshmana keeps its own cursor.
- `worker/queue.py` — `gatekeeper_pairs`: seed, reset, batch-lease, route.
- `worker/edges.py` — `stage3_edges` writes and the FROM_START purge.
- `gates/g1.py` — the gate: purge → seed → batch loop (hydrate, dual-direction NLI, decide,
  write, route) → counters. SHADOW writes `shadow.*` only.
- `worker/main.py` — publishes the next gate after committing, never before.

## Decisions worth knowing about

**One `stage3_edges` row per pair, not per gate.** §7.2's `stageScores: {g1, g2, g3}` only
makes sense as a single document each gate merges its slot into. Doc id
`{low}|{high}|{bare|ctx}|GK` — vishwamitra's own `(pair, variant, promptStamp)` shape with
`GK` in the stamp position, so a gatekeeper row can never collide with an ensemble one.
Writes are `merge=True` upserts, which is what makes `maxRetries 1` safe and what lets a
crashed gate resume without duplicating a row.

**A forwarded pair still gets a row.** Not a verdict — `relation` is left untouched — but its
`stageScores.g1`, because §7.2 wants full precision retained for recalibration and because
G3's cross-check reads G1's contradiction logits back out of it.

**`attempt == 1` is FROM_START.** The run doc's `mode` field cannot distinguish a fresh run
from a FROM_GATE retrigger: only a FULL/G1 message creates a doc, and nothing rewrites
`mode` afterwards. But a new `runRequestId` is the *only* thing that creates a run doc, and
creation claims G1 at attempt 1 — so first attempt means new run, purge; any later G1
execution is resuming output the purge would destroy. The purge stays idempotent regardless.

**Sweeper pass (b) was describing an unreachable state.** §11 said "`REQUESTED` runs with no
claim after 30 min". A `REQUESTED` doc cannot exist — the dispatcher creates the doc *by*
claiming, which sets it `RUNNING`. The reachable version of that hazard is the commit-then-
publish crash window, so pass (b) now republishes the first `PENDING` gate whose predecessor
`SUCCEEDED`. Skipped entirely when pass (a) rescued something on the same run, so one gate
never gets two messages from one sweep. LLD §11 updated.

**The purge has two guards, not one.** It is the only method in the codebase that deletes
another service's data. Scoped to `stage3RunId` (a field only lakshmana writes on its own
rows) *and* every candidate's `method` is re-checked against the `GK_*` set before the
delete is queued. A test plants an `ENSEMBLE` row with our `stage3RunId` and asserts it
survives.

## What is proven, and what is not

Proven against the emulator, in `tests/test_g1_gate.py` and `tests/test_e2e_cascade.py`:

- **The escape hatch holds through the wiring**, not just inside `decide_g1` — 60 pairs with
  hostile score shapes (overwhelming neutral beside a small contradiction), and not one
  flagged pair is decided at G1.
- **A crash mid-batch resumes cleanly** — the first batch's work stays durable, only the
  remainder is re-scored, no pair is duplicated, one edge row per pair.
- **Replay parity** — the gate loop and the VA-97 harness reach the same disposition from the
  same scores and the same `configSnapshot`. A wiring check by construction, exactly as the
  DoD framed it: both drive `gates/decisions.py`.
- **The full push → claim → G1 → publish-G2 path**, through a real envelope and a real claim
  transaction.

Not proven, and honestly so:

- The **graph is a fake** in every gate test. I did verify the three hydration cyphers parse
  and execute against the real `vishwamitra-neo4j` 5.26 container, read-only, with a subject
  id that matches nothing — that is the `neo4j`-marked test, and it **skips** unless
  `GATEKEEPER_NEO4J_PASSWORD` is set (dev default `vishwamitra-dev`). Nothing was written to
  your dev graph.
- **No model has ever run through this gate.** The scorer is stubbed everywhere. Per CLAUDE.md
  rule 0 I did not run inference; the scorer itself was exercised in session02/03.
- The queue's index-free query is emulator-proven only. If production disagrees the fix is one
  composite index — DEFERRED-LIVE 24.

## New DEFERRED-LIVE items

| # | Item |
| --- | --- |
| 20 | `gatekeeper-dispatcher-sa` needs `roles/monitoring.viewer` for the DLQ-depth read (best-effort; the rescue passes work without it) |
| 21 | Configure the OIDC allow-list + audience |
| 22 | Live spot-check the Run Jobs launcher — **and keep the three env vars out of the job's own env** |
| 23 | `gatekeeper_pairs` is a new collection in the shared database; no index required |
| 24 | Confirm the index-free queue query in production |

## What is next

`execution-plan/session05.md` — **VA-100 (G2_CORROBORATION) + VA-101 (G3_CONTRADICTION)**.
Most of the hard parts are done: the gate registry, the context, the queue, the edge writer
and the chain all exist and are gate-agnostic. G2 and G3 should be `gates/g2.py` and
`gates/g3.py` plus registration, following `gates/g1.py`'s shape.

Two things session05 must not get wrong, both already written down in the LLD:

- **G2 may not finalize CORROBORATES on a contradiction-flagged pair** (§8, "Gate ordering for
  flagged pairs"). The flag is on `gatekeeper_pairs.contradictionFlag`; `run_cascade` in the
  replay harness already implements the skip and is the reference.
- **G3 needs G1's contradiction logits**, which live in `stage3_edges.stageScores.g1` — that is
  why a forwarded pair still gets a row. Read them back; do not re-score with the G1 model.

Commits are yours — the message is in the session log. Nothing is parked and nothing is
waiting on you before session05 can run.
