#!/usr/bin/env python3
"""LExGraph: build a memory graph for each dialogue, then answer the questions over it.

    export LLM_API_KEY=...
    python run.py --output runs/my_run --base-url http://localhost:8000/v1 --model <model> \
        --embedding-model /path/to/gte-Qwen2-1.5B-instruct

The method reads only dialogues/<id>/dialogue.json and, per question, question_id / dialogue_id /
question. Outputs in --output:
  graphs/<dialogue_id>/graph.json      units (message_ids, title, summary, key_facts, parent_id) and edges
  graphs/<dialogue_id>/calls/          every LLM call of the write phase (prompt, response, usage, latency)
  answers/questions/<question_id>/     answer call and result (prediction, selected units)
  answers/results.json                 all predictions (input to ../benchmark/evaluation)
  answers/usage.json                   LLM tokens and latency by phase
(answers_full_dialogue/ instead of answers/ with --with-full-dialogue.)
Reruns resume from what is already on disk.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

from answer import PROMPTS, answer_question
from common import digest, read, write
from dialogue import load_messages
from embedding import Embedder
from graph_build import build_graph

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'benchmark' / 'data'
PHASES = (('segment_', 'segmentation'), ('key_', 'key_generation'), ('edges_', 'edge_building'),
          ('answer_call', 'answer'))


def usage_summary(run_dir):
    """Sum token usage and request latency over all LLM calls (failed retries included)."""
    totals = defaultdict(lambda: {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'seconds': 0.0})
    for path in run_dir.rglob('response_*.json'):
        phase = next((name for prefix, name in PHASES if path.parent.name.startswith(prefix)), 'other')
        data = read(path)
        usage = data.get('usage') or {}
        for key in (phase, 'write' if phase != 'answer' else 'read', 'overall'):
            t = totals[key]
            t['calls'] += 1
            t['prompt_tokens'] += usage.get('prompt_tokens', 0) or 0
            t['completion_tokens'] += usage.get('completion_tokens', 0) or 0
            t['seconds'] += data.get('seconds', 0.0)
    return dict(totals)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, default=DATA, help='benchmark data directory (dialogues/, qa.json)')
    p.add_argument('--qa', type=Path, help='questions to answer (default: <data>/qa.json); e.g. a subset from '
                   '../benchmark/data/sample_subset.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dialogue-ids', nargs='+', help='default: all dialogues in --data')
    p.add_argument('--model', required=True)
    p.add_argument('--base-url', required=True, help='OpenAI-compatible API prefix, e.g. http://localhost:8000/v1')
    p.add_argument('--api-key-env', default='LLM_API_KEY', help='environment variable holding the API key')
    p.add_argument('--json-mode', action='store_true', help='request response_format=json_object')
    # write phase
    p.add_argument('--n', type=int, default=20, help='turns read per segmentation call')
    p.add_argument('--similarity-threshold', type=float, default=0.75,
                   help='min Key cosine for an off-frontier unit to become an edge candidate')
    # read phase
    p.add_argument('--token-budget', type=int, default=4096, help='budget on the original text of anchor units')
    p.add_argument('--neighbor-token-budget', type=int, default=4096, help='budget on title+summary of neighbors')
    p.add_argument('--neighbor-hops', type=int, default=1)
    p.add_argument('--rrf-k', type=int, default=10)
    p.add_argument('--with-full-dialogue', action='store_true',
                   help='give the reading LLM the full dialogue in addition to the retrieved memory')
    # embedding
    p.add_argument('--embedding-model', type=Path, required=True, help='local gte-Qwen2-1.5B-instruct directory')
    p.add_argument('--embedding-device', default='auto')
    p.add_argument('--embedding-dtype', choices=('auto', 'float32', 'float16', 'bfloat16'), default='auto')
    p.add_argument('--embedding-batch-size', type=int, default=8)
    p.add_argument('--embedding-max-length', type=int, default=8192)
    # execution
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=16000, help='output token cap per call')
    p.add_argument('--build-workers', type=int, default=4)
    p.add_argument('--answer-workers', type=int, default=8)
    p.add_argument('--build-only', action='store_true')
    args = p.parse_args()
    api_key = os.environ.get(args.api_key_env) or 'EMPTY'
    if min(args.n, args.attempts, args.build_workers, args.answer_workers) < 1 or args.neighbor_hops < 0:
        p.error('Invalid limits')

    dialogues_dir = args.data / 'dialogues'
    all_qas = read(args.qa or args.data / 'qa.json')
    dialogue_ids = args.dialogue_ids or sorted({q['dialogue_id'] for q in all_qas})
    qas = [{k: q[k] for k in ('question_id', 'dialogue_id', 'question')}
           for q in all_qas if q['dialogue_id'] in set(dialogue_ids)]
    llm = {'model': args.model, 'base_url': args.base_url, 'api_key': api_key, 'timeout': args.timeout,
           'attempts': args.attempts, 'json_mode': args.json_mode, 'max_tokens': args.max_tokens}

    graph_config = {'n': args.n, 'similarity_threshold': args.similarity_threshold, 'model': args.model,
                    'embedding_model': args.embedding_model.name,
                    'prompts': digest({n: (HERE / 'prompts' / f'{n}.txt').read_text()
                                       for n in ('segmentation', 'key_generation', 'edge_building')})}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'graph_config.json').exists() and read(args.output / 'graph_config.json') != graph_config:
        p.error('Graph settings changed; use a new output directory')
    write(args.output / 'graph_config.json', graph_config)

    embedder = Embedder(args.output / 'embeddings_cache.npz', args.embedding_model,
                        device=args.embedding_device, dtype=args.embedding_dtype,
                        batch_size=args.embedding_batch_size, max_length=args.embedding_max_length)

    def build_one(did):
        graph_dir = args.output / 'graphs' / did
        if (graph_dir / 'graph.json').exists():
            return did, read(graph_dir / 'graph.json')
        try:
            return did, build_graph(did, dialogues_dir, embedder, graph_dir, n=args.n,
                                    similarity_threshold=args.similarity_threshold, llm=llm)
        except Exception as exc:
            write(graph_dir / 'error.json', {'error': f'{type(exc).__name__}: {exc}'})
            print('graph build failed:', did, exc, flush=True)
            return did, None

    try:
        graphs = {}
        with ThreadPoolExecutor(max_workers=args.build_workers) as pool:
            for future in as_completed([pool.submit(build_one, did) for did in dialogue_ids]):
                did, graph = future.result()
                graphs[did] = graph
                print('graph', did, 'failed' if graph is None else f"{len(graph['units'])} units", flush=True)
        if args.build_only:
            return

        answer_dir = args.output / ('answers_full_dialogue' if args.with_full_dialogue else 'answers')
        answer_config = {'token_budget': args.token_budget, 'neighbor_token_budget': args.neighbor_token_budget,
                         'neighbor_hops': args.neighbor_hops, 'rrf_k': args.rrf_k, 'model': args.model,
                         'with_full_dialogue': args.with_full_dialogue,
                         'prompt': digest(PROMPTS[args.with_full_dialogue])}
        answer_dir.mkdir(parents=True, exist_ok=True)
        if (answer_dir / 'config.json').exists() and read(answer_dir / 'config.json') != answer_config:
            p.error(f'Answer settings changed; remove or rename {answer_dir}')
        write(answer_dir / 'config.json', answer_config)
        messages = {did: load_messages(did, dialogues_dir) for did, g in graphs.items() if g is not None}

        def answer_one(qa):
            folder = answer_dir / 'questions' / qa['question_id']
            if (folder / 'result.json').exists():
                return read(folder / 'result.json')
            did = qa['dialogue_id']
            try:
                if graphs[did] is None:
                    raise RuntimeError('graph build failed')
                return answer_question(qa, graphs[did], messages[did], embedder, folder,
                                       token_budget=args.token_budget,
                                       neighbor_token_budget=args.neighbor_token_budget,
                                       neighbor_hops=args.neighbor_hops, rrf_k=args.rrf_k,
                                       with_full_dialogue=args.with_full_dialogue, llm=llm)
            except Exception as exc:
                return {'question_id': qa['question_id'], 'dialogue_id': did, 'status': 'error',
                        'error': f'{type(exc).__name__}: {exc}'}

        results = []
        with ThreadPoolExecutor(max_workers=args.answer_workers) as pool:
            for future in as_completed([pool.submit(answer_one, qa) for qa in qas]):
                results.append(future.result())
                print(len(results), '/', len(qas), results[-1]['question_id'],
                      results[-1].get('status', 'ok'), flush=True)
        write(answer_dir / 'results.json', sorted(results, key=lambda r: r['question_id']))
        usage = {'write_phase': usage_summary(args.output / 'graphs'), 'read_phase': usage_summary(answer_dir)}
        write(answer_dir / 'usage.json', usage)
        print('errors:', sum(r.get('status') == 'error' for r in results), flush=True)
    finally:
        embedder.close()


if __name__ == '__main__':
    main()
