# lakshmana-core — Gatekeeper (Stage 3 judge cascade)

Lakshmana is the **gatekeeper**: an LLM-free encoder gate cascade (G1_NEUTRAL → G2_CORROBORATION → G3_CONTRADICTION → G4_ESCALATION) replacing vishwamitra's Stage 3 LLM pair-judge. Python 3.12 · uv · ruff · pytest · ONNX Runtime CPU. The design authority is `lakshmana-gatekeeper-lld-wiki.md` (mirror of Confluence page 255688733); read its §3 conventions before writing any code.

## HARD RULES

1. **NEVER run `git commit` or `git push`.** End-of-session: print the commit message in chat (no double quotes; subject + one-line bullets). The owner commits. Sessions may build on uncommitted work from prior sessions — never revert, stash, or `checkout --` anything you didn't write this session.
2. **NEVER run infra-mutating commands**: no `terraform apply/destroy/import`, no `gcloud`/`gsutil` mutations, no deploys. `terraform fmt`/`validate` are fine; `terraform plan` only with explicit owner approval in-session. The owner runs all applies and deploys.
3. **NEVER modify `vishwamitra-core`.** It is read-only reference (code patterns, scripts, runbooks). Vishwamitra-side work (session00, session07) happens in separate sessions over there.
4. **Live/paid tests are owner-gated**: anything hitting real GCP (Vertex calls, live Firestore, GCS writes) needs explicit per-instance owner confirmation, every time. Local tests (emulator, Docker Neo4j, ONNX on disk) are always free to run.
5. **Model artifacts**: runtime never fetches from Hugging Face. HF is allowed only inside the model-prep script (LK-4), run by the owner.

## The session drill

Work arrives as `execution-plan/sessionNN.md` files — see `execution-plan/README.md` for the map and rules. `/next-session` runs the drill. A session ends with: tests green → tickets to In Review → LLD synced if drifted → handoff ritual (archive to `handoff-archive/`, fresh `HANDOFF.md`, commit message printed, **delete the completed session file**).

## Gate protocol — FINISH-ALL-CODE mode (owner directive 2026-07-19: do not interrupt him)

Write and locally verify ALL code. Anything needing live GCP or the owner's hands is **DEFERRED, never parked**:

- Append every deferred item to `execution-plan/DEFERRED-LIVE.md` (what · why · the exact command/check for the final live round) and keep going.
- **Cloud substitutes:** model artifacts → local cache `var/models/` (HF download locally in the prep script is fine; GCS upload deferred; the loader supports both gcs and local-dir modes). Terraform → `fmt` + `validate` only (plan/apply deferred). Vertex live calls → dry-run doubles (one live spot-check deferred).
- **Replay corpus:** restore the vishwamitra Firestore backup (`../vishwamitra-core/var/firestore-backups/`, newest snapshot) into a **separate emulator instance you start yourself** (e.g. `127.0.0.1:8092`) — **NEVER import into the running 8082 emulator**; it holds the owner's live dev state.
- **Session03 GO bar:** apply LLD §15's pre-agreed criteria mechanically — adopt the best passing roster + thresholds, archive the full report (the owner reviews it later, asynchronously). Park ONLY if no roster combination passes.
- Parking (`AWAITING-OWNER-sessionNN.md`) remains only for: no local path AND no deferral possible, or a genuine product decision the LLD doesn't already answer.
- Commits stay orchestrator-side: build sessions still never run git commit/push.

## Local environment

- Firestore emulator: `FIRESTORE_EMULATOR_HOST=127.0.0.1:8082` (plaintext channel; prefer IPv4). Check it's up before relying on it: `curl -s http://127.0.0.1:8082/` returns Ok.
- Neo4j: Docker container `vishwamitra-neo4j` at `bolt://localhost:7687` (HTTP 7474).
- If either is down, report it and park (machine-action gate) rather than starting heavyweight services unasked.

## Conventions (LLD §3 digest)

Python internals snake_case; **Firestore field names camelCase**, collection names snake_case (`gatekeeper_runs`); Pub/Sub payload = proto3 lowerCamelCase JSON; config keys kebab-case, env overrides `GATEKEEPER_*`; timestamps RFC3339 UTC; every gatekeeper-authored doc/message carries the standard envelope (`schemaVersion`, `runRequestId`, `intakeId`, `stage3RunId`, `requestTimestamp`, `triggeredBy`, `createdAt`/`updatedAt`, `attempt`); structured JSON logs with `runRequestId` + `gate` on every line; error codes `GK_E_*`.

## Jira / Confluence

Site `vishx.atlassian.net`, project VA, epic **VA-92** (tickets VA-93…VA-105; LK codes in descriptions). Transitioning code-complete tickets to **In Review** is delegated to Claude; Done stays with the owner. LLD page 255688733 is updated whole-body via MCP markdown together with its mirror file — never one without the other.

## Infra repo

`/Users/piyushvishwakarma/workspace/vishwakarma-ai/lakshmana-infra` — Terraform edits happen there (session02); same hard rules apply (fmt/validate only; owner applies).
