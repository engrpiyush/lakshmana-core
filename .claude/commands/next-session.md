---
description: Run the next Gatekeeper build session from the execution plan (the session drill)
---

ultracode

Run the next Gatekeeper build session. Follow CLAUDE.md's hard rules throughout — especially: never commit/push, never run infra-mutating commands, never touch vishwamitra-core.

## 0. Preflight

- Confirm the local stack: `curl -s http://127.0.0.1:8082/` (Firestore emulator → `Ok`) and `docker ps` shows `vishwamitra-neo4j` healthy. Either down → this is a machine-action gate (step 2's park procedure), unless the session's scope doesn't need it.
- `git status` — note (do not touch) any uncommitted work from prior sessions; you build on top of it.

## 1. Pick the session

- Choose the **lowest-numbered** `execution-plan/sessionNN.md`.
- If it contains `REPO: vishwamitra-core`, it is NOT yours: print a loud skip notice naming it, and pick the next non-vishwamitra file.
- Guard: never start `session08.md` while `session07.md` still exists.
- If an `execution-plan/AWAITING-OWNER-*.md` exists for your chosen session, verify the owner has resolved it (the file describes how to check); resolved → delete the AWAITING file and proceed; unresolved → stop and repeat its ask.
- No runnable session files → report the plan complete and stop.

## 2. Verify prerequisites

Check the session file's "Blocked by" and prerequisites concretely (files, infra, artifacts — not assumptions). **FINISH-ALL-CODE mode is active (see CLAUDE.md):** an unmet prerequisite that has a local substitute (local model cache, separate emulator, dry-run double, validate-only Terraform) is NOT a blocker — use the substitute, log the live counterpart in `execution-plan/DEFERRED-LIVE.md`, and proceed. Park via `AWAITING-OWNER-sessionNN.md` only when no local path or deferral exists.

## 3. Implement the scope

Work the ticket scopes as specified in the session file and the LLD (`lakshmana-gatekeeper-lld-wiki.md`) — the LLD is authoritative on shapes, names, and thresholds. Test-first where practical. Build on any existing uncommitted work from a prior interrupted run of the same session — verify it against the scope and continue; never redo or revert it. Any **decision** the session file or DoD reserves for the owner: you are headless — do not ask and wait, do not guess, do not skip; park it (write `execution-plan/AWAITING-OWNER-sessionNN.md` with the options + your recommendation, leave the session file in place, end cleanly). If you completed everything except such a decision, say so in the AWAITING file — the rerun after resolution will be short.

## 4. Verify

Run `/verify-session` and make it green. Anything red gets fixed before proceeding — never report done with failing checks.

## 5. Close out

1. LLD drift check: if the implementation diverged from the LLD, update **both** the mirror and Confluence page 255688733 (whole-body MCP markdown), version-bumped.
2. Transition the session's tickets to **In Review** (Jira MCP, site vishx.atlassian.net).
3. Handoff ritual: archive the current `HANDOFF.md` to `handoff-archive/HANDOFF-archive-<date>-sessionNN-<slug>.md` (skip if none exists yet), write a fresh `HANDOFF.md` (what was built, decisions, state, what's next), **print the commit message in chat** (no double quotes; subject + one-line bullets), and **delete the completed session file**.
4. Send a push notification: session done, N tickets In Review, commit message ready.

If the session file says the DoD includes an owner sign-off you could not obtain (asked, no answer), treat it as parked: AWAITING-OWNER file, session file stays, notify.
