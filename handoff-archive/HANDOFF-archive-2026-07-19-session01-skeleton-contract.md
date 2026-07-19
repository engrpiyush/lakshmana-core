# HANDOFF — after session01 (VA-93, VA-95)

> Date: 2026-07-19 · Repo: lakshmana-core · LLD: page 255688733 (mirror `lakshmana-gatekeeper-lld-wiki.md`)
> Next session file: `execution-plan/session02.md` (VA-94 · VA-96 — infra TF + model prep)

## What was built

The lakshmana-core skeleton and the idempotency backbone. Everything below is
**uncommitted working-tree state** — the owner commits (see the commit message printed
at the end of the session).

**VA-93 — skeleton (LK-1)**

- Python 3.12 / uv / ruff / pytest project; package `gatekeeper/` with `dispatcher/`
  (FastAPI push handler), `worker/` (framework-free `main()`), `clients/`, `contracts/`,
  `runs/`.
- `Dockerfile.dispatcher` + `Dockerfile.worker`; both build clean locally.
- Firestore client (ADC, emulator-aware) and a **read-only-by-construction** Neo4j
  reader (`Neo4jReader.session()` pins `READ_ACCESS`, D-8) — creds from
  `GATEKEEPER_NEO4J_*` locally, Secret Manager (`gatekeeper-neo4j-ro`) in prod.
- Structured JSON logging with `runRequestId` + `gate` on every line via a context
  manager; `GK_E_*` error-code enum with one subclass per code.
- Config layer: kebab-case keys, `GATEKEEPER_*` env overrides, secrets masked when
  logged, `snapshot()` producing the LLD §7.1 `configSnapshot` shape.
- GitHub Actions CI: ruff, pytest **against a real Firestore emulator container**
  (`GATEKEEPER_REQUIRE_EMULATOR=1` turns an unreachable emulator into a failure, not a
  silent skip), a fixtures-are-current job, and both image builds.

**VA-95 — contract + state machine (LK-3)**

- `contracts/gatekeeper_run_request.proto` is the normative schema; `contracts/payload.py`
  is the hand-written Python mapping (D-6 rules out a shared artifact), and
  `tests/test_payload_contract.py` re-reads the .proto and fails on drift.
- Strict parsing both directions: every field required, unknown keys refused, unknown
  `schemaVersion` diagnosed *before* unknown fields → `GK_E_SCHEMA` → 400 → DLQ.
- `gatekeeper_runs` doc model matching LLD §7.1 field-for-field; snake_case in Python,
  camelCase only at `to_firestore()` / `to_wire()`.
- `GatekeeperRunStore` — all transitions are Firestore transactions: claim
  (PENDING→RUNNING iff prior gate SUCCEEDED, 90-min lease, `attempt++`), gate commit,
  gate/run failure, FINALIZE, and FROM_START supersede in the *same* transaction as run
  creation.
- Golden fixtures in `contracts/fixtures/` (payload ×2 + run doc), byte-identical
  round-trip — these are what VA-106's Kotlin contract test will pin against.

## Decisions worth carrying forward

1. **Transaction rounds, not just client retries.** The Firestore client retries an
   aborted transaction with *no backoff* and then raises a bare `ValueError`. Contenders
   therefore re-collide in lockstep and several dispatchers racing one message could all
   exhaust their budget, leaving a gate unclaimed until redelivery. `_commit()` wraps
   each attempt in a fresh transaction with jittered backoff and maps exhaustion onto
   `GK_E_FIRESTORE_TXN`. Safety never depended on this — the transaction admits one
   winner regardless — liveness did.
2. **Only FULL/G1 may open a run.** A chained or FROM_GATE message for a nonexistent run
   is stale by definition and is dropped (`INVALID_ENTRY`) rather than creating a doc.
   Run docs are never created by the client (LLD §14).
3. **FAILED is not terminal, SUPERSEDED is.** A redelivery cannot restart failed work;
   only an operator FROM_GATE retrigger revives it (LLD §6 rule 3). A FAILED predecessor
   keeps its state through a supersede — it is the audit record of what went wrong.
4. **Supersede is filtered on `stage3RunId` alone** and narrowed in Python. Adding the
   state predicate to the query would demand a composite index for a handful of docs;
   LLD §7.5 lists only the two indexes we actually need.
5. **Only the lease holder settles a gate.** A worker whose lease expired and was swept
   cannot overwrite the rescuer's work — `_settle_gate` checks `leaseOwner` inside the
   transaction.
6. **Duplicates answer 204, not 5xx.** Every duplicate and stale message is *expected*
   under at-least-once delivery; NACKing them would manufacture a DLQ backlog out of
   normal operation.
7. **Commit-then-publish.** The gate result is durable before any next-gate message
   exists, so a crash in that window costs a sweeper rescue rather than a double-run.

## State

| Check | Result |
| --- | --- |
| `ruff check .` / `ruff format --check .` | clean · 30 files |
| `pytest` (emulator-backed) | **125 passed, 0 skipped** |
| Docker images | dispatcher + worker both build |
| Local-stack smoke | Neo4j bolt RO read OK · claim → no-op gate → commit OK · `main()` refuses bad env with exit 2 |
| Golden fixtures | regeneration is a no-op; round-trip byte-identical |
| LLD drift | **none** — run doc, enums, and FROM_START semantics match §5/§6/§7.1 as written |
| VA-93, VA-95 | **In Review** |

Nothing is committed yet: the whole tree is untracked on top of `afe921c Initial commit`.

## What's next — session02 (VA-94 · VA-96)

Infra Terraform in `lakshmana-infra` + model prep. Before it can run, the session01 file
listed two **owner follow-ups**:

- [ ] **Create the `lakshmana-infra` GitHub repo.** The local directory exists and is a
      git repo (CLAUDE.md, LICENSE, README only) — it needs its remote.
- [ ] **Decide region + notification-channel tfvars values** (LLD O-4).

Session02 will check these concretely and park an `AWAITING-OWNER-session02.md` if either
is unmet. Note also that model artifacts are fetched from Hugging Face **only** inside the
LK-4 prep script, run by the owner — the runtime never fetches.

Gates still stubbed: `gatekeeper/worker/gates.py` has a registry and a `no_op_gate` that
commits `{"seen": 0, "forwarded": 0, "noOp": true}`. The zero `seen` counter on a run doc
is the signal that no judging actually happened. Real gates land on VA-99 (G1),
VA-100/VA-101 (G2/G3), VA-102 (G4) — and **nothing gate-shaped is built before the
session03 replay bake-off returns GO**.
