# HANDOFF — after session02 (VA-94, VA-96)

> Date: 2026-07-19 · Repos: lakshmana-infra + lakshmana-core · LLD: page 255688733 **v1.2** (mirror synced)
> Next session file: `execution-plan/session03.md` (VA-97 — the replay bake-off, the go/no-go)
> Mode: **FINISH-ALL-CODE** (owner directive in CLAUDE.md, 2026-07-19). Nothing was parked;
> everything needing live GCP went to `execution-plan/DEFERRED-LIVE.md` (items 9–14 added).

## What was built

**VA-94 — lakshmana-infra Terraform (LK-2)**

A single flat root at `lakshmana-infra/terraform/` — there is not enough here to justify
modules, and flat keeps the IAM matrix readable in one pass. Per LLD §13: topic +
`gatekeeper-requests-push` (OIDC, 5-attempt DLQ) + DLQ topic and a `-dlq-pull` inspection
subscription; three SAs and the full IAM matrix; versioned model bucket
(`prevent_destroy`); dispatcher service (1 vCPU/512Mi, min 0, internal ingress) and
worker job (4 vCPU/8Gi, 6 h, maxRetries 1, VPC connector for Neo4j); 15-minute sweep
schedule; `gatekeeper-neo4j-ro` secret shell; 2 log-based metrics + 5 alert policies.
README carries the `gatekeeper_ro` cypher and the stand-up runbook.

**VA-96 — model prep + loader (LK-4)**

`scripts/prepare_models.py`: HF → optimum ONNX fp32 → dynamic INT8 → 50-pair sanity diff
→ local mirror → optional GCS. `gatekeeper/models/`: `roster.py` (v1 + 5 alternates),
`manifest.py`, `loader.py`. Heavy deps sit in a `model-prep` extra so neither the runtime
image nor CI carries torch.

## Decisions worth carrying forward

1. **The sha256 pin is a two-link chain.** Config's per-gate `sha256` pins the
   *manifest's canonical bytes*; the manifest pins each file by digest and size. One
   short value per gate therefore covers hundreds of MB transitively and still fits in
   `configSnapshot` — so an auditor reading a six-month-old run doc can prove which bytes
   produced its verdicts. Verifying the manifest *without* pinning it would be theatre:
   anyone who can rewrite the model can rewrite the manifest beside it.
2. **`sourceRevision` is a commit sha, never a branch.** `main` today and `main` next
   month are different weights; a manifest that says "main" does not make a run
   reproducible.
3. **The loader's local-dir mode is the same code path, not a test seam.** LK-5 has to
   measure the *production* loader before anything is in GCS. Digest verification is
   identical in both modes; Terraform leaves the local override empty so a deployed
   worker cannot silently fall back to a directory that happens to exist.
4. **Cache hits are re-verified, not trusted via a stamp file.** Hashing half a gigabyte
   costs about a second; being wrong about which weights produced a verdict costs a
   re-run of the whole corpus. Downloads also stage into a temp dir and rename only after
   every digest matches, so a killed task leaves no plausible-looking cache entry.
5. **HHEM is refused, deliberately.** It ships as a `trust_remote_code` architecture that
   optimum's sequence-classification path cannot export. A wrong-architecture INT8 model
   does not crash — it returns confident, plausible, wrong probabilities, which here means
   pairs silently discarded as NEUTRAL. The script stops instead. MiniCheck and FactCG
   both cover G2, so VA-97 is unaffected.
6. **The INT8 sanity diff gates on the mean, not just labels.** Label agreement hides
   recalibration, and the gates read *probabilities* against thresholds, not argmax.
   Tolerances: mean |Δp| ≤ 0.02, max |Δp| ≤ 0.15, label agreement ≥ 0.98.
7. **Dead-lettering needs the Pub/Sub service agent bound on both sides** —
   `pubsub.publisher` on the DLQ topic *and* `pubsub.subscriber` on the push subscription.
   Miss either and the policy looks right in the console and never fires. Likewise the
   dispatcher needs `iam.serviceAccountUser` on the worker SA to start the job at all.
8. **Alert policies fail the plan when no channel resolves** (a `lifecycle.precondition`).
   An alert nobody hears is the most likely silent failure of the remote-state wiring.
9. **Images are `ignore_changes`d.** CI rolls them; without this every `terraform apply`
   would revert the running revision to whatever tfvars last said.

## State

| Check | Result |
| --- | --- |
| `ruff check .` / `ruff format --check .` | clean · 37 files |
| `pytest` (emulator-backed) | **184 passed, 0 skipped** (was 125; +59) |
| Docker images | dispatcher + worker both build |
| Local-stack smoke | loader local-dir → G1 artifact · memoization · disk cache · tampered file → `GK_E_MODEL_FETCH` · wrong pin refused · Neo4j 7687 + emulator 8082 reachable |
| Golden fixtures | regeneration is a no-op |
| `terraform fmt -check -recursive` | clean |
| `terraform validate` | **NOT RUN** — needs `terraform init`, which is not allow-listed (see below) |
| LLD | synced to **v1.2**, mirror + Confluence together |
| VA-94, VA-96 | **In Review**, each with a comment listing the un-met acceptance criteria |

Still uncommitted on top of `afe921c`. Note: the session01 files were `git add`-ed at some
point outside this session — left exactly as found, nothing staged or unstaged by me.

## The one thing worth fixing before session03

**`terraform validate` never ran.** `terraform init` (provider download — not
infra-mutating) is absent from `.claude/settings.json`'s allow-list, so a headless session
cannot self-verify the TF root; its schema correctness is currently reviewed-by-eye.
Recommend adding `Bash(terraform init:*)` — `apply`/`destroy`/`import` stay denied.

## What's next — session03 (VA-97), the go/no-go

The bake-off. Two substitutions the FINISH-ALL-CODE directive already pins:

- **Models** come from the local mirror, not GCS. Run
  `uv run --extra model-prep python scripts/prepare_models.py --roster all` first — it
  populates `var/models/` and prints the manifest shas to pin. Then point the harness at
  it with `GATEKEEPER_MODELS_LOCAL_DIR=var/models`.
- **Corpus** comes from the newest `../vishwamitra-core/var/firestore-backups/` snapshot
  restored into a **separate emulator you start yourself** (e.g. 8092). **Never import
  into the running 8082 emulator** — it holds the owner's live dev state.
- **GO handling:** apply LLD §15's bar mechanically, adopt the best passing
  roster + thresholds, archive the report for async review. Park only if *no* roster passes.

Gates are still stubbed (`no_op_gate`, zero `seen` counter). Nothing gate-shaped gets
built before session03's numbers clear.
