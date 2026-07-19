# session03 — REPLAY BAKE-OFF, the go/no-go (VA-97)

> Repo: lakshmana-core · Blocked by: session01 + session02 (code only — FINISH-ALL-CODE mode substitutes: models from local `var/models/` cache; corpus from the newest `../vishwamitra-core/var/firestore-backups/` snapshot restored into a SEPARATE emulator you start (e.g. 8092) — never the running 8082).
> **This is the gate for everything downstream. Zero LLM spend. No gate code before its numbers clear.**
> GO handling: apply LLD §15's pre-agreed bar mechanically — adopt the best passing roster+thresholds, archive the full report for the owner's async review; park only if NO roster passes.
> LLD: §15 replay & cutover — page 255688733.

## Scope

- **VA-97** — [BE] Replay bake-off harness — thresholds + contradiction-recall report

## Objectives

1. Exporter: Firestore backup → emulator restore (existing vishwamitra `scripts/` flow) → JSONL of the 12,208 pairs {claim texts, withContext, ensemble verdict, votes, golden flags}.
2. Harness: run any gate-config (artifact + thresholds) over the corpus **through the production gate code**, INT8 ONNX artifacts (not fp32); threshold sweeps.
3. Report per candidate roster: per-verdict P/R, NEUTRAL precision/coverage curve, **CONTRADICTS recall vs golden pairs**, calibration (ECE), projected G4 volume, wall-clock.
4. Evaluate the GO bar explicitly: NEUTRAL precision ≥ 0.95 at ≥ 60% coverage · CONTRADICTS recall ≥ 95% of the ensemble's own on golden pairs · projected G4 ≤ 5%.
5. NO-GO path: swap roster rows (alternates already mirrored), re-sweep, re-report — architecture unchanged.

## Definition of done

- [ ] One command produces the full report for a named roster version
- [ ] **Owner sign-off recorded**: chosen roster + thresholds become `configSnapshot` defaults (closes LLD O-1) + the CONTRADICTS tolerance number (closes O-2)
- [ ] LLD §8 threshold proposals updated to the signed values (page + mirror)
- [ ] VA-97 → In Review; handoff ritual; delete this file
