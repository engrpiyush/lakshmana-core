# Golden contract fixtures

These files are the contract between `lakshmana-core` (Python) and `vishwamitra-core`
(Kotlin). D-6 rules out a shared code artifact, so both sides hand-write their mapping
and pin it here: `tests/test_payload_contract.py` on this side, VA-106's contract test on
the other, both asserting against **these exact bytes**.

| File | What it pins |
| --- | --- |
| `gatekeeper_run_request.g1_full.json` | The message that starts a run (FULL / G1_NEUTRAL) |
| `gatekeeper_run_request.g3_from_gate.json` | An operator retrigger of one gate (FROM_GATE / G3) |
| `gatekeeper_runs.doc.json` | A run doc mid-flight: G1 committed, G2 leased, G3/G4 pending |

Canonical form: proto3 JSON mapping (lowerCamelCase), keys in **proto field-number
order**, two-space indent, trailing newline, RFC3339 UTC timestamps at millisecond
precision.

## Regenerating

Never hand-edit these. Run:

```
uv run python scripts/generate_fixtures.py
```

then read the diff. The point of the generator is that "the serializer changed" and "the
contract changed" produce the same diff, and you have to decide which one it was. A diff
here that you did not intend is a contract break, and the Kotlin side breaks with it.

## Values that are still placeholders

`gatekeeper_runs.doc.json` embeds `configSnapshot` built from the shipped defaults, and
two groups of those are known-provisional:

- **Model `sha256` values are empty** until LK-4 (VA-96) mirrors the artifacts to GCS.
- **`g4.llmModel` is `pending-lite-pin`** until O-3/LK-10 pins the flash-lite row.
- **Every numeric threshold is a proposal** from LLD §8; LK-5's replay report (VA-97)
  sets the real values and the owner signs them (O-1).

Each of those lands as a deliberate fixture regeneration on its own ticket.
