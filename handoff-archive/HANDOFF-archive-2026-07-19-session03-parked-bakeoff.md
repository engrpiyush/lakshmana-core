# HANDOFF — after session03 (VA-97, the replay bake-off)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: page 255688733 **v1.3** (mirror synced, page version 4)
> Session file: `execution-plan/session03.md` — **retained**, because the session is parked.
> Read first: `execution-plan/AWAITING-OWNER-session03.md`.
> Mode: **FINISH-ALL-CODE**. All code is written, tested and locally verified. One thing is
> parked, and it is not code.

## The short version

The bake-off harness is finished and works end-to-end on real ONNX artifacts. It has never
seen the real corpus, because `vishwamitra-core` was outside this session's permitted
working directories and the Firestore backup could not be read. **So VA-97's harness is
done and VA-97's go/no-go is unanswered.** Two commands close it; both are written out in
the AWAITING file.

Parking the numbers rather than substituting a synthetic corpus is deliberate. The GO bar
is a claim about how the cascade behaves on vishwamitra's own ensemble labels; numbers from
invented pairs would look like an answer without being one, and session04 is explicitly
gated on this passing.

## What was built

**Artifact policy (the session file's FIRST TASK).** Manifest `schemaVersion` 2: an
artifact mirrors fp32 *and* INT8 and names one as the shipping precision; the loader fetches
only that one. INT8 earns the slot by clearing the same sanity gate as before — the change
is that failing it now *demotes* the artifact instead of refusing it, which is what left
`var/models/` empty last time. Failing numbers ride along in the manifest, and the digest
covers them, so a demotion stays auditable.

All 7 exportable checkpoints re-mirrored from the retained scratch exports (no re-download).
**All 7 ship fp32** — see the findings below.

**`gatekeeper/scoring/`** — ONNX Runtime + `tokenizers`, no torch, no transformers. Refuses
a head whose `id2label` cannot be mapped rather than guessing column order.

**`gatekeeper/gates/decisions.py`** — LLD §8 as pure functions. This is the code the
bake-off scores through *and* the code sessions 04-06 wrap in worker plumbing, so the two
cannot drift. Sessions 04-06 should wrap these, not reimplement them.

**`gatekeeper/replay/`** — corpus exporter, bake-off, metrics. Score once, cache the raw
probabilities, replay them through the decision functions for as many threshold
combinations as wanted.

**`scripts/`** — `export_replay_corpus.py`, `replay_bakeoff.py`, `probe_label_order.py`.

## Findings worth the owner's attention

**1. INT8 is unusable as exported.** Every checkpoint failed: mean |Δp| 0.015-0.155, max
0.23-0.94, label agreement 84-98%. Dynamic per-channel quantization is not value-preserving
on these architectures. Artifacts are consequently 2.5-4× larger than LLD §8's table, and
§16's wall-clock and memory numbers are optimistic by about the same factor. Opened as
**O-6**; nothing is blocked on it.

**2. `contraEscape = 0.02` is probably wrong by an order of magnitude.** On 14 hand-written
probe pairs it flagged 7 of 8 genuinely-neutral pairs, while true contradictions scored
0.985-0.9997 — the populations separate cleanly and the threshold sits far below the gap.
Fourteen pairs is not evidence and no value is proposed. It is why the harness grew a joint
`--grid` search: at 0.02 the escape hatch flags pairs before `neutralMin` is consulted, so
one-axis sweeps return a flat curve and say nothing. Expect this to be the number that moves
the GO bar most.

**3. `ettinx-nli-s` ships unnamed `LABEL_0/1/2`.** Its model card is self-contradictory
about the order. Resolved empirically against the sanity fixture (98% agreement, 62-point
margin over the runner-up permutation) and recorded on the roster entry. The probe also
independently confirmed the other four NLI rows — two of which genuinely disagree on column
order, which is why positional guessing is refused outright.

**4. FactCG separates weakly.** Its polarity is correct (column 1 = supported, same as
MiniCheck, verified against known pairs) but its scores on probe pairs sit in a narrow
0.44-0.85 band where MiniCheck gives 0.011-0.977. Not a bug — a real signal about it as a G2
candidate, which the bake-off will quantify properly.

## LLD drift, all synced (v1.3, page version 4 + mirror)

- `neutralConsensus` was named in §8's `decide_g3` pseudocode with no key and no value —
  now `gatekeeper.gates.g3.neutral-consensus`, default 0.10, and in `configSnapshot.g3`.
- `both_neutral_lean` was undefined — now *neutral is the argmax in both directions of both
  families*, the strictest reading.
- Flagged-pair ordering: a contradiction-flagged pair reaches G3 whatever G2 thinks, per
  §8's own G3 heading. G2 may not finalize CORROBORATES on a flagged pair.
- §13 gained the per-precision bucket layout; §15 gained the metric definitions, the grid
  search rationale and the escape-hatch finding; §16 gained the fp32 caveat.
- O-6 (INT8) and O-7 (corpus unreachable) opened.

## State

- **289 tests green** (was 184). Ruff clean, format clean. Both images build.
- Emulator 8082 and Neo4j both up and used. Nothing was written to 8082 beyond the usual
  throwaway `*_test_*` collections; the backup was **not** imported into it.
- `var/` is gitignored: `var/models/` (~7.7 GB, both precisions), `var/model-prep/`
  (scratch exports, retained — they make a re-mirror free), `var/model-cache/`,
  `var/replay/`.
- Nothing committed. Diffs are ready for review.

## What is next

1. **You:** run the two commands in `AWAITING-OWNER-session03.md` — export the corpus, run
   the bake-off. Budget hours for the first inference pass; everything after it is cached.
2. **Then:** rerun `/next-session`. It will adopt the winning roster + thresholds into
   `gatekeeper/config.py` (closes O-1), record the CONTRADICTS tolerance (O-2), update LLD
   §8 from proposals to signed values, and delete `session03.md`.
3. **Only then** session04 (VA-98/VA-99, dispatcher + G1). Nothing gate-shaped should be
   built before the bar passes — that is the whole point of this gate.

Session07 (VA-106, vishwamitra-side integration) is independent and can be pulled forward
at any time; it does not need the bake-off.
