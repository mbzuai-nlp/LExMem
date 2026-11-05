# LExMem

```
data/          dialogues, graphs, seed tasks and questions
evaluation/    token F1 and LLM-judged ACC
generation/    data-generation pipeline and prompts
```

## Data

```
data/
  qa.json                       2,018 questions
  metadata.json                 per dialogue: task type, seed task, evolved graph, sizes
  seed_tasks.json               150 seed tasks
  evolved_graphs/graph_N.json   150 evolved graphs (typed nodes and edges, no content)
  make_dialogues.py             creates dialogues/<id>/dialogue.json
  sample_subset.py              samples a random subset of dialogues with all their questions (qa.json format)
  dialogues/<dialogue_id>/      224 dialogues
    annotated_dialogue.json     messages grouped by source graph node
    graph.json                  instantiated graph and its seed task
    partition.json              session partition
```

First run `python data/make_dialogues.py`. It writes the plain dialogue `dialogues/<id>/dialogue.json`:

```json
{"sessions": [{"session_id": "S01", "messages": [{"role": "user", "content": "..."}, ...]}, ...]}
```

A system under evaluation reads only:
- `dialogue.json`;
- the `question_id`, `dialogue_id` and `question` fields of `qa.json`.

Message IDs are `<dialogue_id>:<session_id>:M<NNNN>`, numbered from 1 within each session.

Question fields:

| Field | Meaning |
|---|---|
| `method` | question type: `dense`, `sparse` or `same_type` |
| `answer_mode` | `named_slots` or `unordered_set` |
| `answer` | reference answer, one value per slot |
| `answer_keywords` | per slot: `canonical` answer and `aliases`, used for scoring |
| `evidence` | supporting message IDs, per slot |
| `evidence_node_ids` | supporting graph nodes, per slot |

## Evaluation

Predictions are a JSON list of `{"question_id": ..., "prediction": {"answer_items": [...]}}`.

```bash
python evaluation/scoring.py --predictions results.json                   # token F1
JUDGE_API_KEY=... python evaluation/llm_judge.py --predictions results.json --output judge_out \
    --base-url https://api.openai.com/v1 --model-name gpt-4o-mini --json-mode   # ACC
```

- **Subset**: to score on a subset from `data/sample_subset.py`, add `--qa <subset file>` to both commands.
- **ACC**: the judge prompt is `judge_prompt.txt`. Each question is graded yes / partial / no, scored 1 / 0.5 / 0.
- **Answer instruction**: `answer_prompt.txt` is the instruction for answering from the full dialogue.

## Generation

See `generation/README.md`.
