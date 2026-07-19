# Gatekeeper execution plan — session map

> Epic **VA-92** (+ VA-77 precursor, VA-106 in VA-27) · LLD Confluence **255688733** (mirror `../lakshmana-gatekeeper-lld-wiki.md`)
> Drill rules: work the **lowest-numbered session file** present; a session ends with the handoff ritual (archive → fresh `HANDOFF.md` → commit message printed in chat → **delete the completed session file**). Owner tasks between sessions are listed in each file — do not start a session whose prerequisites are unmet.

| Session | Tickets | Repo | Gate / note |
| --- | --- | --- | --- |
| [session00](session00.md) | VA-77 | vishwamitra-core | Workstream A+B dials — executes FIRST; independent of Lakshmana |
| [session01](session01.md) | VA-93 · VA-95 | lakshmana-core | Skeleton + contract/state machine (emulator only) |
| [session02](session02.md) | VA-94 · VA-96 | lakshmana-infra + lakshmana-core | Infra TF + model prep; heavy owner follow-ups |
| [session03](session03.md) | VA-97 | lakshmana-core | **REPLAY BAKE-OFF — the go/no-go.** Nothing gate-shaped before this passes |
| [session04](session04.md) | VA-98 · VA-99 | lakshmana-core | Dispatcher + G1; requires session03 GO |
| [session05](session05.md) | VA-100 · VA-101 | lakshmana-core | G2 + G3 gates |
| [session06](session06.md) | VA-102 · VA-103 | lakshmana-core | G4 + FINALIZE + purge/retrigger; full chain green on emulator |
| [session07](session07.md) | VA-106 | **vishwamitra-core** | Integration; may be pulled forward any time after session01 |
| [session08](session08.md) | VA-104 · VA-105 | lakshmana-core (+ live GCP) | Observability + SHADOW run + cutover; owner-heavy |

Critical path: 00 ∥ (01 → 02 → **03 GO** → 04 → 05 → 06) + 07 → 08.

## Run modes (automated spawning)

Sessions are spawned by the standardized `/next-session` command (`.claude/commands/next-session.md`); `CLAUDE.md` carries the hard rules; `.claude/settings.json` pre-approves the local dev loop and denies commits/applies. Vishwamitra-repo sessions (00, 07) are **never** run from here — the tooling skips them loudly, and session08 is blocked while session07.md exists.

- **THE mode — headless driver:** `scripts/run-next-session.sh` runs one session; `--loop` grinds until the plan is complete, a session parks (`AWAITING-OWNER-*.md` written, session file retained), or session08 is blocked on session07. The script self-wraps in `caffeinate -is` so the machine cannot sleep mid-run, and tees every session's output to `~/Library/Logs/lakshmana-driver.log`. Headless cannot answer prompts — anything not pre-approved in settings fails visibly instead of hanging.
- **Remote Control mode: RETIRED (2026-07-19).** Server crashes mint new environments (`environment_deleted` on the phone) and kill in-flight sessions — more churn than value here. `scripts/remote-control.sh` + the launchd plist remain for reference only.

**Gate protocol (headless):** every owner gate — decision or machine action — parks: `AWAITING-OWNER-sessionNN.md` with what's needed (+ options and a recommendation for decisions), session file retained, clean stop. Resolve the file, rerun the driver. **Commits:** batch mode — sessions never commit; review diffs and commit with the printed messages when at the Mac.

Owner setup (one-time, done): `claude mcp` (Atlassian auth) in this repo · workspace trust accepted.
