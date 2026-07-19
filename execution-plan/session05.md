# session05 — G2 + G3 gates (VA-100, VA-101)

> Repo: lakshmana-core · Blocked by: session04 (worker gate pattern established).
> LLD: §8 G2/G3 — page 255688733.

## Scope

- **VA-100** — [BE] G2_CORROBORATION gate — grounding checker, direction + calibrated confidence
- **VA-101** — [BE] G3_CONTRADICTION gate — two-family cross-check, human-queue routing

## Objectives

1. G2: MiniCheck-DeBERTa-L on G1 leftovers — `supportFwd`/`supportBwd`, plus **grounding mode** (claim vs paired claim's source snippet) where available; decide CORROBORATES at `supportMin` (direction = argmax, confidence = session03 calibration); everything undecided → G3; contradictionFlag pairs pass through untouched.
2. G3: DeBERTa-mnli-fever-anli dual-direction; reuse G1 contradiction logits from `stageScores` (no re-inference); two-family agreement ≥ `contraMin` → **CONTRADICTS-candidate → HUMAN queue** (cascade never finalizes CONTRADICTS); two-family neutral consensus → NEUTRAL (`GK_G3_XCHECK`); disagreement → G4 `STAGE_DISAGREEMENT`.
3. Both: SHADOW variants, counters, commit-then-publish next gate.

## Definition of done

- [ ] Replay parity with the session03 harness for both gates
- [ ] Property test: G3 never writes CONTRADICTS as a final verdict
- [ ] HUMAN-routed pairs visible to the existing vishwamitra contradiction-queue reader (emulator check)
- [ ] Calibrated confidence spot-checked against the calibration table
- [ ] VA-100, VA-101 → In Review; handoff ritual; delete this file
