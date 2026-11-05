# Data generation

| Step | Entry point | Prompts |
|---|---|---|
| 1 Initial graphs | `step1_initial_graphs/generate_graphs.py`, `screen_batch.py` | `graph_generation_prompt.txt`, `graph_check_prompt.txt` |
| 2 Graph evolution | `step2_evolution/run_batch.py` | `evolution_rule_prompts.txt` |
| 3 Seed tasks | `step3_seed_tasks/generate_tasks.py` | `task_prompt.txt` |
| 4 Graph instantiation | `step4_instantiation/make_pairs.py`, `instantiate.py` | `node_instantiation_prompt.txt`, `node_relation_check_prompt.txt`, `graph_instantiation_check_prompt.txt` |
| 5 Session partition | `step5_sessions/partition.py` | `session_partition_prompt.txt` |
| 6 Dialogue expansion | `step6_dialogue/generate_dialogue.py` | `node_dialogue_prompt.txt`, `node_dialogue_check_prompt.txt`, `full_dialogue_check_prompt.txt` |
| 7 QA generation | `step7_qa/run_qa.py` | `step7_qa/prompts/` |

## Requirements

- Python 3.10+ and `pip install -r requirements.txt`.
- All steps use GPT-6 Sol (`gpt-6-sol`).
- **Steps 1–3** call it through the [Codex CLI](https://github.com/openai/codex) (`codex exec`); change it with `--model`.
- **Steps 4–7** call an OpenAI-compatible API through `llm.py`. Model settings are in `configs/`, and the API key is read from `OPENAI_API_KEY`.

## Running

```bash
python3 step1_initial_graphs/generate_graphs.py --count 500 --output runs/initial
python3 step1_initial_graphs/screen_batch.py --input runs/initial --output runs/screen
python3 step2_evolution/run_batch.py --input runs/screen/accepted --output runs/evolved
python3 step3_seed_tasks/generate_tasks.py --assignments <assignments.json> --output runs/tasks
python3 step4_instantiation/make_pairs.py --tasks runs/tasks/<id>/tasks.json --graphs runs/evolved/accepted --output runs/pairs.json
OPENAI_API_KEY=... ./run_stages4to7.sh runs/pairs.json runs/batch1
```

- `assignments.json` lists `{"task_type", "topic"}` entries, one seed task per entry; `topic_assignments.json` is an example.
- `make_pairs.py` pairs each seed task with 4 evolved graphs.
- Questions are written to `runs/batch1/7_qa/qa.json`, and dialogues to `runs/batch1/6_dialogue/<job>/`.
- All steps resume from their on-disk cache.
