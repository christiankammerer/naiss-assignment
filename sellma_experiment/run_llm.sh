#!/usr/bin/env bash
# Step 5: run Claude Code headless on every model folder, one fresh session per model.
#
# Usage:   ./run_llm.sh <runs_dir> <options|open> [parallel_sessions]
# Example: ./run_llm.sh ~/llm_runs options 3
#          ./run_llm.sh ~/llm_dev  options 1        # test on the dev model first
#
# <runs_dir> must contain one sub-folder per model (weights.npz + prompt_*.md), copied OUTSIDE
# the project folder. Each session may only run python and write files; web tools are disabled.
# Outputs per folder: transcript_<cond>.jsonl (full session), answer_<cond>.md (final answer).
# Already answered folders are skipped, so the script can simply be restarted.
# Override the model with:  MODEL=claude-opus-5-5 ./run_llm.sh ...
set -euo pipefail
RUNS="$1"; COND="$2"; PAR="${3:-3}"; MODEL="${MODEL:-claude-opus-5-5}"

run_one() {
  local d="$1" cond="$2" model="$3"
  cd "$d"
  if [ -s "answer_${cond}.md" ]; then echo "skip   $(basename "$d") ($cond)"; return; fi
  echo "start  $(basename "$d") ($cond)"
  claude -p --model "$model" ${EFFORT:+--effort $EFFORT} \
    --output-format stream-json --verbose \
    --allowedTools "Bash(python3:*)" "Bash(python:*)" "Write" \
    --disallowedTools "WebSearch" "WebFetch" \
    --append-system-prompt "Write any scripts or temporary files only inside the current folder, never in /tmp." \
    < "prompt_${cond}.md" > "transcript_${cond}.jsonl" 2> "stderr_${cond}.log" || true
  python3 - "$cond" <<'EOF'
import json, sys
cond = sys.argv[1]
result = None
for line in open(f"transcript_{cond}.jsonl", encoding="utf-8"):
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if event.get("type") == "result":
        result = event.get("result")
open(f"answer_{cond}.md", "w", encoding="utf-8").write(result or "")
EOF
  echo "done   $(basename "$d") ($cond)"
}
export -f run_one

find "$RUNS" -mindepth 1 -maxdepth 1 -type d | sort |
  xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} "$COND" "$MODEL"
