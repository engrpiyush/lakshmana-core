#!/usr/bin/env bash
# Phone-driven mode: persistent Remote Control server named "Lakshmana" in tmux.
# Sessions started from the Claude mobile app / claude.ai/code land in this directory
# (and under the Lakshmana group); kick each one off with: /next-session
# The inner loop self-restarts the server (survives crashes and the first-run
# workspace-trust error: accept trust once via interactive `claude` here, and the
# server comes up on the next retry). Boot persistence: scripts/launchd plist.
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=lakshmana-rc
CLAUDE_BIN="${CLAUDE_BIN:-/Users/piyushvishwakarma/.local/bin/claude}"
LOG="$HOME/Library/Logs/lakshmana-rc.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Remote-control server already running — attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$(pwd)" \
  "sh -c 'while true; do echo \"[\$(date)] starting claude remote-control\" | tee -a $LOG; $CLAUDE_BIN remote-control --name Lakshmana --spawn same-dir 2>&1 | tee -a $LOG; echo \"[\$(date)] server exited — retrying in 15s\" | tee -a $LOG; sleep 15; done'"

echo "Remote-control server 'Lakshmana' started (tmux session: $SESSION, log: $LOG)."
echo "Status:  tmux capture-pane -pt $SESSION | tail -12"
echo "Phone:   Claude app / claude.ai/code -> Lakshmana -> new session -> /next-session"
