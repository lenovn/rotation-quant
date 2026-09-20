#!/usr/bin/env bash
set -euo pipefail
handoff_dir=/home/dongpeiyan/projects/rotation-quant/runs/phase3/handoff-20260914.NByJvb
cd /home/dongpeiyan/projects/rotation-quant
echo "$BASHPID" > "$handoff_dir/launcher.pid"
set +e
/home/dongpeiyan/.vscode-server/extensions/openai.chatgpt-26.908.40401/bin/linux-x86_64/codex -a on-request exec \
  --sandbox workspace-write --skip-git-repo-check \
  -C /home/dongpeiyan/projects/rotation-quant \
  -m gpt-6-astra -c 'model_reasoning_effort="xhigh"' \
  -c 'approvals_reviewer="auto_review"' \
  --json --color never \
  --output-last-message "$handoff_dir/latest_final.txt" \
  resume 01a09bd8-a78a-7702-9e4b-57442618928c - < "$handoff_dir/resource_update_prompt.txt" \
  > "$handoff_dir/events_resource_update.jsonl" 2> "$handoff_dir/codex_resource_update.stderr.log"
codex_exit_status=$?
echo "$codex_exit_status" > "$handoff_dir/codex_exit_status_resource_update.txt"
exit "$codex_exit_status"
