#!/usr/bin/env bash
# Steps 4-7 for one batch of (seed task, evolved graph) pairs.
# Usage: OPENAI_API_KEY=... ./run_stages4to7.sh <pairs.json> <run_dir>
# Exit code 75 = paused (API unavailable / quota); rerun the same command to continue from the checkpoint.
set -u
stage() { "$@"; rc=$?; if [ $rc -eq 75 ]; then echo "== PAUSED $(date) - rerun to continue"; exit 75; fi; return 0; }
HERE=$(cd "$(dirname "$0")" && pwd); PAIRS=$(realpath "$1"); R=$(realpath -m "$2"); mkdir -p "$R"
export TOKEN_LOG="$R/tokens.jsonl"; W=${WORKERS:-16}
exec > >(tee -a "$R/run.log") 2>&1
cd "$HERE"
export LLM_CONFIG=$HERE/configs/stages4to6.json
echo "== step 4 $(date)"; stage python3 step4_instantiation/instantiate.py --pairs "$PAIRS" --output "$R/4_instantiation" --workers $W
echo "== step 5 $(date)"; stage python3 step5_sessions/partition.py --input "$R/4_instantiation/accepted" --output "$R/5_sessions" --workers $W
echo "== step 6 $(date)"; stage python3 step6_dialogue/generate_dialogue.py --input "$R/5_sessions" --output "$R/6_dialogue" --workers $W
export LLM_CONFIG=$HERE/configs/stage7.json
RES=; [ -d "$R/7_qa" ] && RES=--resume
echo "== step 7 $(date)"; (cd step7_qa && stage python3 -B run_qa.py --input "$R/6_dialogue" --output "$R/7_qa" --workers $W $RES); [ $? -eq 75 ] && exit 75
echo "== done $(date)"
