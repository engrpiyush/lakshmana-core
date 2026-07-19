# DEFERRED-LIVE — the single final round (build/deploy/live verification)

> Running ledger. Sessions append; nothing here blocks code. Executed as ONE round after all
> lakshmana code sessions complete (+ the vishwamitra sessions 00/07 the owner scoped separately).

| # | Item | Owner action / exact command | Added by |
| --- | --- | --- | --- |
| 1 | Terraform apply (lakshmana-infra) | `terraform init && terraform apply` after review; plan was deferred | setup |
| 2 | Model artifacts → GCS | rerun the LK-4 mirror script with the GCS destination (local `var/models/` cache is the interim source of truth) | setup |
| 3 | Neo4j `gatekeeper_ro` role + user + SM secret | cypher in lakshmana-infra README; store creds in Secret Manager | setup |
| 4 | Dispatcher + worker image push & deploy | build already verified locally; push to Artifact Registry, deploy Cloud Run service + job | setup |
| 5 | G4 live spot-check | small real flash-lite batch, spend counter vs billing (session06 DoD item, dry-run double used locally) | setup |
| 6 | Alert fault-injection | fire DLQ/job-failure/stuck-run/cap alerts once in the live project (session08/VA-104) | setup |
| 7 | vishwamitra sessions 00 + 07 | VA-77 dials · VA-106 integration (separate sessions, owner-scoped) | setup |
| 8 | SHADOW run + cutover | session08/VA-105 — needs 1–4 + 7 done | setup |
| 9 | `terraform init` + `validate` on the new TF root | `cd lakshmana-infra/terraform && cp backend.hcl.example backend.hcl && cp terraform.tfvars.example terraform.tfvars` (fill both) `&& terraform init -backend-config=backend.hcl && terraform validate`. **`fmt` is clean and the HCL parses; `validate` never ran** — `terraform init` is not in `.claude/settings.json`'s allow-list, so a headless session cannot download the provider. Recommend adding `Bash(terraform init:*)` (it is not infra-mutating; `apply`/`destroy`/`import` stay denied) so future sessions can self-verify. | session02 |
| 10 | Fill the tfvars the plan needs | `project_id`, `vishwamitra_state_bucket`, `dispatcher_image`, `worker_image`, `neo4j_uri` have no defaults and must be supplied. Confirm vishwamitra-infra's state really exports **`vpc_connector_id`** and **`notification_channel_ids`** — if the names differ, set the `vpc_connector_id` / `notification_channel_ids` variables instead of editing the root. | session02 |
| 11 | Images must exist before the first apply | Cloud Run will not create a service pointing at a missing image: push `gatekeeper-dispatcher` and `gatekeeper-worker` to Artifact Registry, *then* apply. Both build clean locally (verified session01 + session02). | session02 |
| 12 | Mirror the roster to `var/models/`, then pin the shas | `uv run --extra model-prep python scripts/prepare_models.py --roster all` (downloads from HF, exports ONNX INT8, runs the 50-pair sanity diff, writes `var/models/<name>/<version>/`). It prints one `manifest sha256` per artifact — put those in `gatekeeper.gates.g{1,2,3}.sha256`. Until then the loader logs "artifact is not pinned" and loads unverified. GCS upload is item 2 (add `--bucket`). | session02 |
| 13 | `hhem@v1` needs a hand-written export | It ships as a `trust_remote_code` architecture, so `optimum`'s sequence-classification path cannot export it; the prep script refuses it by design rather than quantizing the wrong graph. Not a blocker for VA-97 — `minicheck-deberta-l` and `factcg-deberta-l` cover the G2 slot. | session02 |
| 14 | Two alerts need log lines that do not exist yet | `gatekeeper/run_overdue` reads `jsonPayload.event="run_overdue"` (sweeper, VA-103) and `gatekeeper/cap_exceeded` reads `jsonPayload.errorCode="GK_E_CAP_EXCEEDED"` (G4, VA-102). Both metrics apply fine now and simply read zero; confirm they fire during the item-6 fault injection. | session02 |
