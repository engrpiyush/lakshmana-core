#!/usr/bin/env bash
# THE driver: runs the next Gatekeeper session headlessly (claude -p).
# One session per invocation by default; --loop grinds until parked/blocked/complete.
# Remote Control mode is RETIRED (environment churn on crashes); gates park via
# AWAITING-OWNER files — resolve them, rerun this script.
set -euo pipefail

# Keep the machine awake for the whole run (idle + system sleep).
if [[ -z "${LAKSHMANA_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  export LAKSHMANA_CAFFEINATED=1
  exec caffeinate -is "$0" "$@"
fi

cd "$(dirname "$0")/.."
PLAN=execution-plan
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
DRIVER_LOG="$HOME/Library/Logs/lakshmana-driver.log"

run_one() {
  local f pick=""
  for f in $(ls "$PLAN"/session*.md 2>/dev/null | sort); do
    if grep -q "REPO: vishwamitra-core" "$f"; then
      echo "SKIP (vishwamitra-core session — run it over there): $f"
      continue
    fi
    pick="$f"
    break
  done

  if [[ -z "$pick" ]]; then
    echo "No runnable session files left — lakshmana plan complete (vishwamitra sessions may remain)."
    return 10
  fi
  if [[ "$pick" == */session08.md && -f "$PLAN/session07.md" ]]; then
    echo "BLOCKED: session08 requires session07 (vishwamitra integration) to complete first."
    return 11
  fi

  echo "=== $(date '+%F %T') running $(basename "$pick") ===" | tee -a "$DRIVER_LOG"
  "$CLAUDE_BIN" -p "/next-session" 2>&1 | tee -a "$DRIVER_LOG" || echo "claude exited non-zero (continuing to state check)" | tee -a "$DRIVER_LOG"

  if [[ -f "$pick" ]]; then
    echo "PARKED: $(basename "$pick") still present — owner action needed."
    for r in "$PLAN"/AWAITING-OWNER-*.md; do
      [[ -f "$r" ]] || continue
      echo "--- $r ---"
      cat "$r"
    done
    return 12
  fi

  echo "=== $(basename "$pick") completed (session deleted its file) ==="
  echo "Reminder: review the diff and commit with the printed message before the next session if you want clean per-session commits."
  return 0
}

if [[ "${1:-}" == "--loop" ]]; then
  while run_one; do :; done
else
  run_one
fi
