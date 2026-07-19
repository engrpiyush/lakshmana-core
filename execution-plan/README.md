# Gatekeeper execution plan — session map

> Epic **VA-92** (+ VA-77 precursor, VA-106 in VA-27) · LLD Confluence **255688733** (mirror `../lakshmana-gatekeeper-lld-wiki.md`)
> Drill rules: work the **lowest-numbered session file** present; a session ends with the handoff ritual (archive → fresh `HANDOFF.md` → commit message printed in chat → **delete the completed session file**). Owner tasks between sessions are listed in each file — do not start a session whose prerequisites are unmet.

| Session | Tickets | Repo | Gate / note |
| --- | --- | --- | --- |
| [session00](session00.md) | VA-77 | vishwamitra-core | Workstream A+B dials — executes FIRST; independent of Lakshmana |
| [session01](session01.md) | VA-93 · VA-95 | lakshmana-core | Skeleton + contract/state machine (emulator only) |
| [session02](session02.md) | VA-94 · VA-96 | lakshmana-infra + lakshmana-core | Infra TF + model prep; heavy owner follow-ups |
| ~~session03~~ | VA-97 | lakshmana-core | ~~Replay bake-off — the go/no-go~~ **DONE 2026-07-19.** Rescoped: harness + corpus + config surface delivered; the GO gate was removed by owner directive and calibration moved to DEFERRED-LIVE 17–19 |
| ~~session04~~ | VA-98 · VA-99 | lakshmana-core | ~~Dispatcher + G1~~ **DONE 2026-07-19.** Dispatcher (OIDC, Run Jobs launcher, `/sweep`) + G1_NEUTRAL end to end on the emulator; chain publishes G2. LLD v1.5 corrects §7.3 — the judge queue is a Neo4j relationship, so routing lives in the lakshmana-owned `gatekeeper_pairs` |
| ~~session05~~ | VA-100 · VA-101 | lakshmana-core | ~~G2 + G3 gates~~ **DONE 2026-07-19.** G2 (grounding mode off `claims.sourceExcerpt`, flagged pairs pass through uninferred) + G3 (two-family cross-check, family A read back from `stageScores.g1`, candidates to `tier: HUMAN` with `relation: CONTRADICTS` and no `decidedBy`); G1→G2→G3 chain green on the emulator. LLD v1.6; new open item O‑9 |
| [session06](session06.md) | VA-102 · VA-103 | lakshmana-core | G4 + FINALIZE + purge/retrigger; full chain green on emulator |
| [session07](session07.md) | VA-106 | **vishwamitra-core** | Integration; may be pulled forward any time after session01 |
| [session08](session08.md) | VA-104 · VA-105 | lakshmana-core (+ live GCP) | Observability + SHADOW run + cutover; owner-heavy |

Critical path: 00 ∥ (01 → 02 → 03 → 04 → 05 → 06) + 07 → 08. *(2026-07-19: 05 done — G4 + FINALIZE in session06 plug into the same seams: `gates/g4.py` + `register()`, with `GateContext.scorer_factory` swapped for a Vertex client. `runner_for(G4_ESCALATION)` still resolves to `no_op_gate`, which is how "not built yet" stays visible.)* *(2026-07-19: 04 done — G2/G3 in session05 plug into the registry, queue, edge writer and chain that 04 built; both are `gates/gN.py` + `register()`.)* *(2026-07-19: the 03 GO gate is removed by owner directive — models are configurable per gate with defaults, per-run frozen via configSnapshot; calibration deferred to DEFERRED-LIVE 18–19.)*

## Run modes (automated spawning)

Sessions are spawned by the standardized `/next-session` command (`.claude/commands/next-session.md`); `CLAUDE.md` carries the hard rules; `.claude/settings.json` pre-approves the local dev loop and denies commits/applies. Vishwamitra-repo sessions (00, 07) are **never** run from here — the tooling skips them loudly, and session08 is blocked while session07.md exists.

- **THE mode — headless driver:** `scripts/run-next-session.sh` runs one session; `--loop` grinds until the plan is complete, a session parks (`AWAITING-OWNER-*.md` written, session file retained), or session08 is blocked on session07. The script self-wraps in `caffeinate -is` so the machine cannot sleep mid-run, and tees every session's output to `~/Library/Logs/lakshmana-driver.log`. Headless cannot answer prompts — anything not pre-approved in settings fails visibly instead of hanging.
- **Remote Control mode: RETIRED (2026-07-19).** Server crashes mint new environments (`environment_deleted` on the phone) and kill in-flight sessions — more churn than value here. `scripts/remote-control.sh` + the launchd plist remain for reference only.

**Gate protocol (headless):** every owner gate — decision or machine action — parks: `AWAITING-OWNER-sessionNN.md` with what's needed (+ options and a recommendation for decisions), session file retained, clean stop. Resolve the file, rerun the driver. **Commits:** batch mode — sessions never commit; review diffs and commit with the printed messages when at the Mac.

Owner setup (one-time, done): `claude mcp` (Atlassian auth) in this repo · workspace trust accepted.
