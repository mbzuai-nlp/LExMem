<div align="center">

# Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2611.06414-b31b1b.svg)](https://arxiv.org/abs/2611.06414)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

</div>

## 📖 Overview

Long-term problem solving, such as scientific research and software development, moves along an evolving trial-and-error trajectory toward a central goal. We term these **exploration tasks**. To handle them, a model must remember facts, and also how solutions were tested against intermediate outcomes and how earlier attempts were revised or revived.

This repository contains:

- **LExMem**, a graph-guided benchmark for memory in long-term exploration tasks. It contains 224 dialogues and 2,018 question–answer instances derived from exploration graphs.
- **LExGraph**, a graph-based memory framework. It incrementally organizes a dialogue into interconnected memory nodes and retrieves evidence through hybrid retrieval and neighbor expansion.

<p align="center">
  <img src="assets/example.png" width="60%">
  <br>
  <em>An exploration task modeled as a graph. Two branches are highlighted; their steps are connected by relations.</em>
</p>

## 🧩 LExMem

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

LExMem is built in three stages:

1. **Graph construction.** LLMs generate exploration graphs of typed nodes and relations. An evolutionary mechanism then adds realistic twists, such as contradictory evidence, refuted ideas being reconsidered, and revised hypotheses.
2. **Graph-guided dialogue synthesis.** Each graph is instantiated with a research or software-design task. Every node is then expanded into a block of multi-turn dialogue.
3. **QA generation.** Questions are generated from sampled structures (dense subgraphs, sparse paths and same-type node sets), followed by quality control.

## 🕸️ LExGraph

LExGraph builds its memory graph from the raw dialogue only, without access to the ground-truth graphs.

- **Segmentation.** After every *n* new user turns, an LLM detects topic boundaries. Each segment becomes a memory node, with a key (summary and salient facts) and a value (the original turns).
- **Edges.** A new node attaches to a parent on the *right frontier*, i.e. the path from the root to the latest node. It is also linked to similar nodes on other branches through cross-branch edges.
- **Retrieval.** Nodes are ranked by cosine similarity and BM25, fused with reciprocal rank fusion. The top nodes and their graph neighbors are selected under token budgets and given to the LLM to answer.


## 📁 Repository Structure

```
.
├── benchmark/
│   ├── data/            LExMem: dialogues, graphs, seed tasks, questions
│   ├── evaluation/      token F1 and LLM-judged ACC
│   └── generation/      7-step data-generation pipeline and prompts
├── method/              LExGraph
└── assets/              figures
```

More details:
- [`benchmark/README.md`](benchmark/README.md): data format and evaluation.
- [`benchmark/generation/README.md`](benchmark/generation/README.md): regenerating the data.
- [`method/README.md`](method/README.md): LExGraph options and outputs.

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/mbzuai-nlp/LExMem.git
cd LExMem
conda create -n lexmem python=3.10 -y && conda activate lexmem
pip install -r method/requirements.txt
```

LExGraph uses [gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) as its embedding model. Download it locally first.

### 2. Prepare the dialogues

To save space, each dialogue is stored once, annotated with the graph node of every message. The following command writes the plain dialogue `benchmark/data/dialogues/<id>/dialogue.json`, which is the only input an evaluated system may read:

```bash
python benchmark/data/make_dialogues.py
```

### 3. Run and evaluate LExGraph

`evaluate.sh` builds the memory graphs, answers all questions, and reports token F1 and ACC. The judge is gpt-4o-mini. There are two settings for what the answering LLM sees:

- `retrieval`: the retrieved memory only.
- `full_dialogue`: the full dialogue plus the retrieved memory.

```bash
cd method
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<model> LLM_API_KEY=...   # any OpenAI-compatible endpoint
export EMBEDDING_MODEL=/path/to/gte-Qwen2-1.5B-instruct JUDGE_API_KEY=...

./evaluate.sh retrieval     runs/lexgraph
./evaluate.sh full_dialogue runs/lexgraph     # reuses the graphs built above
```

> [!TIP]
> Running `full_dialogue` on the whole benchmark can be costly and slow, because every answer call reads an entire dialogue. You can sample a fixed random subset of dialogues (e.g, 40 dialogues often contain ~400 questions) and run on it instead:
> ```bash
> python ../benchmark/data/sample_subset.py --output subset_qa.json
> QA_FILE=subset_qa.json ./evaluate.sh full_dialogue runs/lexgraph
> ```

### 4. Evaluate your own system

Your system may read only `dialogue.json` and the `question_id`, `dialogue_id` and `question` fields of `benchmark/data/qa.json`. Write predictions as:

```json
[{"question_id": "...", "prediction": {"answer_items": ["...", "..."]}}]
```

Then score them:

```bash
python benchmark/evaluation/scoring.py --predictions results.json                          # token F1
JUDGE_API_KEY=... python benchmark/evaluation/llm_judge.py --predictions results.json --output judge \
    --base-url https://api.openai.com/v1 --model-name gpt-4o-mini --json-mode               # ACC
```

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{dai2026twiststurnsgraphbasedbenchmark,
      title={Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks}, 
      author={Chongyuan Dai and Yaling Shen and Zirui Song and Junfeng Liao and Jinpeng Hu and Xiuying Chen},
      year={2026},
      eprint={2611.06414},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.06414}, 
}
```
