#!/usr/bin/env python3
"""Sample a random subset of LExMem: whole dialogues with all of their questions.

    python sample_subset.py --output subset_qa.json            # 40 dialogues, seed 0

The output has the same format as qa.json, so it can replace it via --qa in method/run.py,
evaluation/scoring.py and evaluation/llm_judge.py (or QA_FILE in method/evaluate.sh).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--num-dialogues', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--qa', type=Path, default=HERE / 'qa.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    qas = json.loads(args.qa.read_text())
    dialogue_ids = sorted({q['dialogue_id'] for q in qas})
    if not 1 <= args.num_dialogues <= len(dialogue_ids):
        p.error(f'--num-dialogues must be between 1 and {len(dialogue_ids)}')
    chosen = set(random.Random(args.seed).sample(dialogue_ids, args.num_dialogues))
    subset = [q for q in qas if q['dialogue_id'] in chosen]
    args.output.write_text(json.dumps(subset, ensure_ascii=False, indent=1) + '\n')
    print(f'{len(chosen)} dialogues, {len(subset)} questions -> {args.output}')
    print('question types:', dict(Counter(q['method'] for q in subset)))


if __name__ == '__main__':
    main()
