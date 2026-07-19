# HANDOFF — after session03 (VA-97, the replay bake-off — rescoped and closed)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: page 255688733 **v1.4** (mirror synced, page version 5)
> Session file: `execution-plan/session03.md` — **deleted**, the session is complete.
> VA-97 → **In Review** with scope notes on the ticket.
> Mode: **FINISH-ALL-CODE**. Nothing is parked. The previous handoff parked the bake-off;
> that parking is resolved — not by running it, but by the owner removing it as a gate.

## The short version

**Session03 no longer decides anything about models.** The owner stopped the full-corpus
evaluation mid-run (it monopolised the machine) and rescoped the session: the GO bar is not
a gate on sessions 04-06, model choice and threshold calibration happen later, and the
build proceeds on the v1 roster defaults.

So this closeout did three things: made model selection a pure config concern with the v1
roster as the default, **closed a real hole in the freeze invariant**, and told the truth in
the LLD and on the ticket about what was and was not measured. The harness, the 12,208-pair
corpus and the 4-of-7 score cache are deliverables as they stand.

## The bug worth knowing about

Thresholds were read from the run's frozen `configSnapshot`; **the model was read from live
config.** `artifact_for_gate(loader, config, gate)` went straight to
`gatekeeper.gates.g1.model`. An operator editing config between G1 and G3 — which is exactly
what will happen once calibration resumes — would have split a run across two checkpoints
while its thresholds stayed put. The run doc would have looked clean; the verdicts would
have been unreproducible.

Fixed by making the snapshot the only source for both:

- `gatekeeper.config.GateBinding.from_snapshot(snapshot, gate)` — model, sha256 pin, and
  token budget, resolved from `configSnapshot.g{1,2,3}`. The artifact-side companion to
  `Thresholds.from_snapshot` on the numbers side. It lives in `config.py` because that plus
  `enums.py` is the base layer — anywhere else and loader/encoder/decisions import-cycle.
- `artifact_for_binding()` / `scorer_for_binding()` are the worker path. The old
  `artifact_for_gate()` / `scorer_for_gate()` survive for tooling with no run doc (the
  bake-off, the label probe) but now resolve **through** `config.snapshot()`, so they cannot
  drift from what a real run would freeze.
- Token budget is frozen too — truncation changes scores, so a run that started at 512
  tokens must not finish at 256.
- `tests/test_freeze_invariant.py` (10 tests) proves it from three sides: resolution beats
  live config, the loader fetches the frozen artifact, and a FROM_GATE resume keeps the
  original snapshot while FROM_START picks up the new one.

The store side was already correct (`claim_gate` ignores the passed snapshot on an existing
run) and had a test; it was only the resolution half that leaked.

## Also delivered this session

- **Per-gate model selection by config alone.** `configSnapshot.g{1,2,3}.model` defaults to
  the v1 roster, thresholds to the LLD §8 proposals. Adopting a bake-off winner is a config
  change, never a code change — that was the owner's explicit requirement.
- **LLD v1.4**, mirror and page 255688733 together: §8 gained the config-selection rule and
  the freeze invariant; §15 gained the decoupling notice and the honest statement about the
  unmeasurable contradiction criterion; O-1/O-2 rescoped as open-but-not-blocking; **O-7
  closed** (the corpus was exported — access was never the remaining problem, calibration
  is).

## What was deliberately NOT done

- **No roster chosen, no thresholds calibrated.** O-1 open. Do not read the v1 defaults as
  a decision — they are the LLD's proposals, unchanged.
- **The remaining 3 artifacts were not scored.** The session file says so explicitly.
  `var/replay/scores/` holds minicheck-L, nli-deberta-small, deberta-mnli-fever-anli and
  ettinx; missing are modernbert, vitaminc, factcg. The grid search after them is seconds.
- **The GO bar's CONTRADICTS-recall criterion is unmeasurable on this corpus** — zero golden
  pairs, zero CONTRADICTS verdicts, and `withContext=0` (that vishwamitra run judged every
  pair bare). It is reported `goBar.unmeasurable`, never as a pass. Closing O-2 needs a
  second corpus from an intake whose ensemble actually contradicted something.

## Findings carried forward from the earlier part of this session

1. **INT8 is unusable as exported.** All 7 checkpoints failed: mean |Δp| 0.015-0.155, max
   0.23-0.94, agreement 84-98%. Everything ships fp32. Artifacts are 2.5-4× larger than
   §8's table and §16's wall-clock/memory numbers are optimistic by about the same. **O-6.**
   Worth flagging: the worker job's 8 GB ceiling wants re-checking against the 1.66 GB
   artifacts before the first real run.
2. **`contraEscape = 0.02` is probably wrong by an order of magnitude.** On 14 hand-written
   probe pairs it flagged 7 of 8 genuinely-neutral pairs while true contradictions scored
   0.985-0.9997. Fourteen pairs is not evidence and no value is proposed — but it is why the
   harness has a joint `--grid` search: at 0.02 the escape hatch fires before `neutralMin`
   is consulted, so one-axis sweeps return a flat curve.
3. **`ettinx-nli-s` ships unnamed `LABEL_0/1/2`** with a self-contradictory model card.
   Resolved empirically (98% agreement, 62-point margin) and recorded on the roster entry.
4. **FactCG separates weakly** — 0.44-0.85 on probe pairs where MiniCheck gives 0.011-0.977.
   A real signal about it as a G2 candidate, for the bake-off to quantify.

## State

- **306 tests green** (was 289), 52 of them emulator-backed, 0 skipped. Ruff check + format
  clean. Both images build.
- Emulator 8082 and Neo4j both up. The backup was **never** imported into 8082 — the export
  used a throwaway instance, since torn down.
- `var/` is gitignored: `var/models/` (~7.7 GB, both precisions), `var/model-prep/` (scratch
  exports, retained — they make a re-mirror free), `var/model-cache/`, `var/replay/`
  (corpus + the 4 cached score files).
- The `neo4j` pytest marker currently selects 0 tests; the Neo4j reader is covered by fakes.
  Not a regression, just worth knowing the marker proves nothing today.
- Nothing committed. Commit message printed in chat.

## What is next

1. **session04** (VA-98/VA-99, dispatcher + G1). It is no longer gated on anything — build
   against the v1 defaults. Wrap `gatekeeper/gates/decisions.py`, do not reimplement it;
   read the gate's artifact via `GateBinding.from_snapshot(run.config_snapshot, gate)`, never
   from live config.
2. **Calibration, whenever the owner wants the machine for it** — DEFERRED-LIVE 17-19.
   Item 19 resumes the bake-off from the score cache; item 17 exports the second corpus that
   O-2 needs; item 18 mirrors every artifact to the bucket.
3. **Pin the artifact shas** (DEFERRED-LIVE 16) once a roster is final — not before, since a
   demotion or re-quantization changes the manifest digest. Until then every load logs
   "artifact is not pinned".

Session07 (VA-106, vishwamitra-side integration) remains independent and can be pulled
forward at any time.
