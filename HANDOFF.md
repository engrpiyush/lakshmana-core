# HANDOFF — after session05 (VA-100 G2_CORROBORATION · VA-101 G3_CONTRADICTION)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: page 255688733 **v1.6** (mirror synced, page version 7)
> Session file: `execution-plan/session05.md` — **deleted**, the session is complete.
> VA-100 and VA-101 → **In Review**.
> Mode: **FINISH-ALL-CODE**. Nothing is parked. Three new DEFERRED-LIVE items (25–27).
> Tests: **375 passed, 1 skipped** — and the skip is now *fixable locally*, see item 27.

## The short version

Three gates of the cascade now run end to end. One Pub/Sub push drives G1 → G2 → G3, each
gate claiming its own lease, judging its own slice, committing, and publishing the next —
`tests/test_e2e_cascade.py::test_the_chain_drives_itself_from_g1_through_g2_to_g3` feeds each
published message back through the real dispatcher rather than calling the next gate directly,
so the claim transaction is exercised three times over.

**G2** scores support in both claim-vs-claim directions and, where the paired claim has a
source excerpt, additionally scores claim-vs-evidence — MiniCheck's native `(document, claim)`
shape, and the one thing this cascade can do that the LLM judge could not. **G3** cross-checks
contradiction across two model families, reading family A's logits back off the edge row rather
than re-inferring them, and routes agreed candidates to the human queue without ever finalizing
a CONTRADICTS verdict.

## What was built

- `gatekeeper/gates/g2.py` — claim batch → hydrate → score (both directions + grounding) →
  `decide_g2` → edge write → route. Counters `seen, corroborates, forwarded, grounded,
  flaggedPassThrough, truncated`.
- `gatekeeper/gates/g3.py` — claim batch → read family A off `stageScores.g1` → score family B →
  `decide_g3` → edge write → route. Counters `seen, neutral, humanRouted, escalated, truncated`.
- `gatekeeper/integration.py::source_excerpts` — the `claims.sourceExcerpt` read G2's grounding
  mode needs, chunked through `getAll`.
- `gatekeeper/worker/edges.py` — `g2_stage_scores`, `g3_stage_scores`,
  `g1_scores_from_stage_scores`, and `EdgeWriter.read_stage_scores` (direct gets by
  deterministic key; no index).
- `decisions.py` — `decide_g3` now sets `confidence` on its ROUTE_HUMAN decision. Added to the
  *shared* function rather than the gate so replay parity holds by construction.
- Config: `gatekeeper.integration.claims-collection` (env `GATEKEEPER_CLAIMS_COLLECTION`).
- Tests: `test_g2_gate.py` (10), `test_g3_gate.py` (9), the three-gate chain, and a registry
  regression test. LLD v1.6 on both the mirror and Confluence.

## Three decisions worth knowing about

**1. The source excerpt is in Firestore, not the graph.** The LLD said "the paired claim's
source snippet" without saying where it lives. It is not in Neo4j at all —
`Stage3GraphRepository.mergeEvidence` writes `text`/`basis`/`sourceClass` onto `:Claim` but
never `sourceExcerpt`, and `:Source` carries no text. A graph read would have returned nothing
for every pair and failed *silently*. It is read from vishwamitra's `claims` collection instead,
read-only. New open item **O‑9** on the LLD, and DEFERRED-LIVE 26 asks for a coverage count on
real data — if few claims carry an excerpt, grounding mode is decoration.

**2. G3's edge row does carry `relation = CONTRADICTS`, and this is not a violation of "the
cascade never finalizes CONTRADICTS".** They are different layers, and getting this wrong in
either direction is expensive. On the **queue** row the pair leaves with `tier = HUMAN` and no
`decidedBy` — that field is what asserts the cascade settled a pair, and G3 never sets it. On
the **edge** row `relation` is a pair-level *candidate*: vishwamitra's `FactAssembler` lifts
CORROBORATES/CONTRADICTS pair rows into Fact edges and stamps every contradiction
`reviewStatus = PROPOSED`, which is exactly what the §11.10 human queue reads. Writing nothing
would have made confirmed contradictions **invisible** to the queue built to review them. That
is the DoD's "visible to the existing contradiction-queue reader", and it is tested against the
fields the assembler actually consumes.

**3. The candidate's confidence is `min(famA, famB)`, the weaker family.** §8 gives ROUTE_HUMAN
no confidence, but the §11.10 queue orders and floors on one. A two-family agreement is only as
strong as the family least convinced by it; `max` would let one confident model push a pair the
other barely flagged to the top of a human's worklist — the self-consistency failure this
cross-check exists to replace.

## Two bugs found and fixed on the way

**The gate registry silently disabled G2 and G3.** `_load_implementations` guarded on
`if _RUNNERS:` — but each gate module registers itself on import, so anything importing
`gatekeeper.gates.g1` directly (the replay harness, half the test suite) left the registry
non-empty and the guard concluded "already loaded". Every other gate then resolved to
`no_op_gate`, **which commits successfully**: a full cascade would have run G1 for real, committed
zeroes for G2 and G3, and reported SUCCEEDED having skipped two thirds of the judging. Now a
dedicated `_LOADED` flag, with a regression test that reproduces the trigger.

**The real-Neo4j test could never run.** `test_the_hydration_cypher_runs_against_a_real_neo4j`
called `load_config({})` — an empty env map — so `GATEKEEPER_NEO4J_PASSWORD` could never arrive
and it skipped unconditionally, including in session04's "0 skipped" claim. Now `load_config()`.
Verified passing against the running container. It still skips in this repo because the password
is not in the settings env — see DEFERRED-LIVE 27, a one-line owner fix.

## State

- **375 passed, 1 skipped.** The skip is the `neo4j` cypher smoke; set
  `GATEKEEPER_NEO4J_PASSWORD=vishwamitra-dev` and it passes (0 skipped, verified).
- Both Docker images build. Ruff clean and formatted.
- Nothing parked. No live GCP touched, no infra mutated, no commits made.

## What's next — session06 (VA-102 G4_ESCALATION · VA-103 FINALIZE)

`execution-plan/session06.md` is the next file. G4 is the first gate that spends money: pinned
flash-lite, k=1, `maxLlmPairs` cap, `WOULD_ESCALATE_LLM` in SHADOW, and a dry-run double locally
per the FINISH-ALL-CODE rule (the live spot-check is DEFERRED-LIVE 5). It plugs into exactly the
same seams G2 and G3 used — `gates/g4.py` + `register()`, `GateContext.scorer_factory` swapped
for a Vertex client — and `runner_for(G4_ESCALATION)` still resolves to `no_op_gate`, which is
the signal that it is genuinely not built yet. FINALIZE computes `totals` and sets the run
SUCCEEDED; it publishes nothing.

Watch for: G4's counters are the last input to `totals`, and `run_gate` currently logs
"last gate committed; FINALIZE is pending VA-103" instead of finalizing.
