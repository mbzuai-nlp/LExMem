#!/usr/bin/env python3
"""Step 7: question generation over all completed dialogues of step 6.

Per dialogue: sample foci (sample.py), then for each candidate run generation, the question-quality check and
the dialogue-answerability check (generate_qa.py), and message-level evidence localization (localize_evidence.py).
Only questions passing every stage reach qa.json.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from generate_qa import read, write, dialogue_blocks, process
from sample import prepare, sampling_counts
from localize_evidence import index_messages, localize_question, exportable_questions

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import llm  # noqa: E402

TEMPLATES = ('dense', 'sparse', 'same_type', 'question_check', 'dialogue_check', 'evidence_localize', 'evidence_check')
CODE = ('run_qa.py', 'generate_qa.py', 'sample.py', 'localize_evidence.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Output directory of step 6')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--count', type=int, help='Override both sampled-method limits (defaults: dense 5, sparse 7)')
    p.add_argument('--sparse-count', type=int, help='Override the sparse limit independently of --count')
    p.add_argument('--methods', nargs='+', choices=('dense', 'sparse', 'same_type'), default=['dense', 'sparse', 'same_type'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--call-attempts', type=int, default=2, help='Attempts per model call on technical/format errors')
    p.add_argument('--evidence-retries', type=int, choices=range(4), default=3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.workers, args.call_attempts) < 1:
        p.error('Invalid limits')
    try:
        sampling_counts(args.count, args.sparse_count)
    except ValueError as exc:
        p.error(str(exc))
    if len(set(args.methods)) != len(args.methods):
        p.error('Duplicate methods')
    names = [s['job'] for s in read(args.input / 'summary.json')['jobs'] if s['status'] == 'completed']
    jobs = []
    for name in sorted(names):
        gp, dp = args.input / name / 'graph.json', args.input / name / 'annotated_dialogue.json'
        graph, annotated = read(gp), read(dp)
        jobs.append({'name': name, 'graph': graph, 'annotated': annotated, 'blocks': dialogue_blocks(graph, annotated),
                     'messages': index_messages(annotated, name), 'hashes': {'graph': digest(gp), 'dialogue': digest(dp)}})
    templates = {n: (HERE / 'prompts' / f'{n}.txt').read_text() for n in TEMPLATES}
    files = [HERE / n for n in CODE] + sorted((HERE / 'prompts').glob('*.txt'))
    config = {'models': llm.models(), 'count': args.count, 'sparse_count': args.sparse_count, 'methods': args.methods,
              'seed': args.seed,
              'call_attempts': args.call_attempts, 'evidence_retries': args.evidence_retries,
              'sources': {j['name']: j['hashes'] for j in jobs},
              'code': {str(f.relative_to(HERE)): digest(f) for f in files}}
    if args.output.exists():
        if not args.resume:
            p.error('Output exists; use --resume with identical inputs and configuration')
        if read(args.output / 'config.json') != config:
            p.error('Inputs/code/settings changed; use a new directory')
    else:
        if args.resume:
            p.error('Cannot resume a missing output')
        args.output.mkdir(parents=True)
        write(args.output / 'config.json', config)
        for f in files:
            dest = args.output / 'program' / f.relative_to(HERE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    tasks = []
    for job in jobs:
        folder = args.output / 'jobs' / job['name']
        job['folder'] = folder
        write(folder / 'graph.json', job['graph'])
        write(folder / 'annotated_dialogue.json', job['annotated'])
        write(folder / 'dialogue_index.json', {'dialogue_id': job['name'], 'messages': job['messages']})
        cp = folder / 'candidates.json'
        if cp.exists():
            candidates = read(cp)
        else:
            sampled = prepare(job['graph'], count=args.count, seed=args.seed, sparse_count=args.sparse_count, methods=args.methods)
            write(folder / 'sampling.json', sampled)
            candidates = [{'candidate_id': f'{method}_{i:02d}', 'method': method, 'focus': focus}
                          for method in ('dense', 'sparse', 'same_type') for i, focus in enumerate(sampled[method], 1)]
            write(cp, candidates)
        job['results'] = {}
        job['candidate_count'] = len(candidates)
        tasks.extend((job, i, c) for i, c in enumerate(candidates, 1))
    print(f'PREPARED {len(jobs)} dialogues, {len(tasks)} candidates, {args.workers} workers', flush=True)

    def work(job, i, candidate):
        saved = job['folder'] / 'final' / f'{candidate["candidate_id"]}.json'
        if saved.exists():
            result = read(saved)
            if result['status'] != 'error':
                return job, result
        genargs = SimpleNamespace(**{**vars(args), 'output': job['folder'] / 'generation'})
        generated = process(candidate, job['graph'], job['blocks'], templates, genargs)
        result = {'graph_id': job['name'], 'candidate_id': candidate['candidate_id'], 'method': candidate['method'],
                  'generation_status': generated['status'], 'status': generated['status']}
        if 'qa' in generated:
            result['qa'] = {**generated['qa'], 'question_id': job['name'] + '_' + candidate['candidate_id'], 'method': candidate['method']}
        if generated['status'] == 'accepted':
            evargs = SimpleNamespace(**{**vars(args), 'output': job['folder'] / 'evidence'})
            evidence_saved = evargs.output / f'Q{i:03d}' / 'result.json'
            if evidence_saved.exists() and read(evidence_saved)['status'] != 'error':
                ev = read(evidence_saved)
            else:
                ev = localize_question(i, result['qa'], job['messages'], job['name'], templates, evargs)
            result.update(status=ev['status'], evidence_status=ev['status'], annotation_attempts=ev['annotation_attempts'])
            if ev['status'] == 'accepted':
                result['qa'] = exportable_questions([ev], job['messages'])[0]
            elif ev['status'] == 'error':
                result['error'] = ev['error']
        elif generated['status'] == 'error':
            result['error'] = generated['error']
        write(saved, result)
        print(job['name'], candidate['candidate_id'], result['status'], flush=True)
        return job, result

    def publish(job):
        rows = sorted(job['results'].values(), key=lambda r: r['candidate_id'])
        write(job['folder'] / 'results.json', rows)
        write(job['folder'] / 'qa.json', [r['qa'] for r in rows if r['status'] == 'accepted'])
        allrows = [r for j in jobs for r in j['results'].values()]
        write(args.output / 'qa.json', [r['qa'] for r in sorted(allrows, key=lambda r: (r['graph_id'], r['candidate_id']))
                                       if r['status'] == 'accepted'])
        write(args.output / 'summary.json', {
            'dialogues': len(jobs), 'total_candidates': len(tasks), 'finished': len(allrows),
            'counts': dict(Counter(r['status'] for r in allrows)),
            'generation_counts': dict(Counter(r['generation_status'] for r in allrows)),
            'by_method': {m: dict(Counter(r['status'] for r in allrows if r['method'] == m)) for m in ('dense', 'sparse', 'same_type')}})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(work, j, i, c): (j, c) for j, i, c in tasks}
        for future in as_completed(pending):
            try:
                job, result = future.result()
            except Exception as exc:
                job, c = pending[future]
                result = {'graph_id': job['name'], 'candidate_id': c['candidate_id'], 'method': c['method'],
                          'generation_status': 'error', 'status': 'error', 'error': str(exc)}
                write(job['folder'] / 'final' / f'{c["candidate_id"]}.json', result)
            job['results'][result['candidate_id']] = result
            publish(job)
    print(json.dumps(read(args.output / 'summary.json')), flush=True)
    if any(r['status'] == 'error' for j in jobs for r in j['results'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    raise SystemExit(llm.run_guarded(main))
