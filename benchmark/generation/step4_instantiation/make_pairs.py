#!/usr/bin/env python3
"""Pair seed tasks (step 3) with evolved graphs (step 2): --rounds independent random one-to-one pairings,
so each task is paired with --rounds graphs (150 tasks x 4 rounds = 600 candidates). Output: the --pairs input of instantiate.py."""
import argparse
import json
from pathlib import Path
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', type=Path, required=True, help='JSON list of seed tasks')
    ap.add_argument('--graphs', type=Path, required=True, help='accepted/ directory of step 2 (graph_*.json)')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    tasks = json.loads(args.tasks.read_text())
    graphs = sorted((p for p in args.graphs.glob('graph_*.json') if not p.name.endswith('.input.json')),
                    key=lambda p: int(p.stem.split('_')[1]))
    n = min(len(tasks), len(graphs))
    rng = random.Random(args.seed)
    pairs = []
    for r in range(1, args.rounds + 1):
        for t, g in zip(rng.sample(tasks, n), rng.sample(graphs, n)):
            pairs.append({'job': f'r{r}_{len(pairs) + 1:03d}_{g.stem}', 'task': t, 'graph': str(g.resolve())})
    args.output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(pairs)} pairs -> {args.output}')


if __name__ == '__main__':
    main()
