#!/usr/bin/env bash
# Run LExGraph on LExMem and score it (token F1 + LLM-judged ACC).
#
#   ./evaluate.sh <retrieval|full_dialogue> <output_dir> [extra run.py options]
#
#   retrieval      the reading LLM sees only the retrieved memory
#   full_dialogue  the reading LLM sees the full dialogue plus the retrieved memory
#
# Environment: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (reading/writing LLM), EMBEDDING_MODEL (local path of
# gte-Qwen2-1.5B-instruct), JUDGE_API_KEY (OpenAI key for the gpt-4o-mini judge).
# Optional: QA_FILE, a question subset from ../benchmark/data/sample_subset.py (default: the full qa.json).
# Both settings share the memory graphs in <output_dir>/graphs/, so the second setting only re-runs answering.
set -euo pipefail
cd "$(dirname "$0")"

setting=${1:?usage: ./evaluate.sh <retrieval|full_dialogue> <output_dir>}
out=${2:?usage: ./evaluate.sh <retrieval|full_dialogue> <output_dir>}
shift 2
case "$setting" in
  retrieval)     flag=();                       answers="$out/answers" ;;
  full_dialogue) flag=(--with-full-dialogue);   answers="$out/answers_full_dialogue" ;;
  *) echo "unknown setting: $setting" >&2; exit 1 ;;
esac

ls ../benchmark/data/dialogues/*/dialogue.json >/dev/null 2>&1 || python ../benchmark/data/make_dialogues.py
qa=(); [ -n "${QA_FILE:-}" ] && qa=(--qa "$QA_FILE")

python run.py --output "$out" --base-url "$LLM_BASE_URL" --model "$LLM_MODEL" \
    --embedding-model "$EMBEDDING_MODEL" ${flag[@]+"${flag[@]}"} ${qa[@]+"${qa[@]}"} "$@"

python ../benchmark/evaluation/scoring.py --predictions "$answers/results.json" ${qa[@]+"${qa[@]}"}
python ../benchmark/evaluation/llm_judge.py --predictions "$answers/results.json" --output "$answers/judge" \
    ${qa[@]+"${qa[@]}"} --base-url https://api.openai.com/v1 --model-name gpt-4o-mini --json-mode
