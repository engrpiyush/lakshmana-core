---
description: Standard Gatekeeper verification pass — lint, tests, build, local-stack smoke
---

Run the standard verification pass for lakshmana-core and report a single results table. Do not skip a step because it "obviously passes"; run it.

1. **Lint/format**: `uv run ruff check .` and `uv run ruff format --check .`
2. **Tests**: `uv run pytest -q` (unit + emulator-backed integration; `FIRESTORE_EMULATOR_HOST=127.0.0.1:8082`). Report count passed/failed/skipped.
3. **Build**: `docker build` both images (dispatcher, worker) if Dockerfiles exist at this stage of the plan.
4. **Local-stack smoke** (when the relevant code exists): emulator reachable (`curl -s http://127.0.0.1:8082/` → Ok), Neo4j bolt reachable, worker executes a no-op/dry-run gate against the emulator without error.
5. **Contract fixtures**: if `contracts/fixtures/` exists, round-trip serialization test passes byte-identically.
6. **Conventions spot-check**: no snake_case leaking into stored Firestore field names, no camelCase in Python identifiers (ruff + a grep over serialization modules).

Output: a table of step → status → evidence (one line each). Any failure: fix it and re-run before reporting the pass. Live-GCP checks are NOT part of this command — those are owner-gated.
