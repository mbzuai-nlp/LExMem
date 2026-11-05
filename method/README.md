# LExGraph

| File | Content |
|---|---|
| `graph_build.py` | Memory graph construction: segmentation, key generation, edge construction with the right frontier |
| `retrieve.py` | Node retrieval: cosine + BM25 fused by RRF, primary nodes and neighbor nodes under token budgets |
| `answer.py` | Context assembly and answer generation |
| `run.py` | Builds graphs for all dialogues and answers all questions |
| `evaluate.sh` | `run.py` followed by token F1 and LLM-judged ACC |
| `prompts/` | `segmentation.txt`, `key_generation.txt`, `edge_building.txt`, `answer.txt`, `answer_with_full_dialogue.txt` |

## Evaluation settings

LExGraph can be evaluated in two settings. They differ only in what the reading LLM receives:

| Setting | Context given to the reading LLM | Prompt |
|---|---|---|
| `retrieval` | retrieved memory only | `prompts/answer.txt` |
| `full_dialogue` | full dialogue + retrieved memory | `prompts/answer_with_full_dialogue.txt` |

The retrieved memory is the same in both: primary nodes with their keys and original turns, plus neighbor nodes with their keys and the relations between them.

## Running

```bash
pip install -r requirements.txt
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<model> LLM_API_KEY=...
export EMBEDDING_MODEL=/path/to/gte-Qwen2-1.5B-instruct JUDGE_API_KEY=...

./evaluate.sh retrieval     runs/lexgraph
./evaluate.sh full_dialogue runs/lexgraph
```

- `LLM_BASE_URL` can be any OpenAI-compatible endpoint.
- **Subset.** The `full_dialogue` setting puts each whole dialogue into every answer call. On the full dataset this can be costly and slow, so you may run it on a random subset instead. `sample_subset.py` samples whole dialogues together with all their questions. Pass the resulting file through `QA_FILE` and every run uses the same subset:

  ```bash
  python ../benchmark/data/sample_subset.py --output subset_qa.json   # default: 40 dialogues, seed 0
  QA_FILE=subset_qa.json ./evaluate.sh full_dialogue runs/lexgraph
  ```
- The two settings share the memory graphs in `runs/lexgraph/graphs/`. Whichever runs second only repeats answering.
- Extra arguments after the output directory are passed to `run.py`, e.g. `--n 20` or `--token-budget 4096`.
- **Defaults**: `--n 20 --similarity-threshold 0.75 --rrf-k 10 --token-budget 4096 --neighbor-token-budget 4096 --neighbor-hops 1`.

## Outputs

| Path | Content |
|---|---|
| `runs/lexgraph/graphs/<dialogue_id>/graph.json` | memory graph |
| `runs/lexgraph/answers/` | `retrieval` setting |
| `runs/lexgraph/answers_full_dialogue/` | `full_dialogue` setting |

Each answers folder contains:
- `results.json`: predictions;
- `usage.json`: token usage and latency;
- `judge/`: ACC.

Token F1 is printed to the terminal. Every LLM call is cached, so runs resume where they stopped.
