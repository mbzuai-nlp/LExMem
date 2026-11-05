#!/usr/bin/env python3
"""Step 2 (batch): evolve every skeleton graph accepted in step 1, then screen each final graph again.

For each graph: evolve_graph.py (initial review, rule-triggered continuations with per-round checks), render the final
graph, and screen its JSON + PNG with step1_initial_graphs/check_graphs.py. Graphs passing the rule check with a
"reasonable" verdict and no issues are exported to <output>/accepted/.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from evolve_graph import validate, validate_review

HERE = Path(__file__).resolve().parent
STEP1 = HERE.parent / 'step1_initial_graphs'
LOCK = threading.Lock()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def publish(root, manifest):
    manifest['counts'] = dict(Counter(r['status'] for r in manifest['graphs']))
    save(root / 'batch.json', manifest)


def export_accepted(root, manifest):
    out = root / 'accepted'
    out.mkdir(exist_ok=True)
    entries = []
    for record in manifest['graphs']:
        if record['status'] != 'accepted':
            continue
        name = record['id']
        evo, screen = root / record['evolution_dir'], root / record['screening_dir']
        screened = screen / 'accepted' / f'{name}.json'
        assert screened.read_bytes() == (evo / 'final.json').read_bytes()
        for src, dst in [('final.json', name + '.json'), ('final.input.json', name + '.input.json'),
                         ('final.png', name + '.png'), ('final.svg', name + '.svg')]:
            shutil.copy2(evo / src, out / dst)
        entries.append({'id': name, 'sha256': hashlib.sha256(screened.read_bytes()).hexdigest(),
                        'proposals': record['proposals'], 'nodes': record['final_nodes'],
                        'rounds': record['rounds'], 'screening': record['screening_dir']})
    save(out / 'manifest.json', {'criterion': 'local checks passed AND reasonable AND no issues',
                                 'kept': len(entries), 'candidate_count': manifest['count'], 'graphs': entries})


def invoke(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_review(folder, graph):
    for path in sorted(folder.glob('review_*/parsed.json')):
        review = load(path)
        try:
            validate_review(review, graph)
        except (ValueError, TypeError, KeyError):
            continue
        return review
    return None


def job(root, manifest, record):
    name = record['id']
    folder = root / 'jobs' / name
    folder.mkdir(parents=True, exist_ok=True)
    source = root / record['initial_source']
    if digest(source) != record['initial_sha256']:
        raise ValueError('Initial snapshot changed')
    def update(**changes):
        with LOCK:
            record.update(changes)
            publish(root, manifest)
        print(name, changes, flush=True)
    common = ['--model', manifest['model'], '--timeout', str(manifest['timeout']),
              '--retries', str(manifest['retries'])]
    evo = folder / 'evolution'
    checkpoint = evo / 'checkpoint.json'
    complete = checkpoint.exists() and load(checkpoint)['status'] == 'complete'
    if not complete:
        if checkpoint.exists():
            command = [sys.executable, str(HERE/'evolve_graph.py'), '--output', str(evo), '--resume']
        elif evo.exists() and any(evo.iterdir()):
            review = completed_review(evo/'initial_review', load(source))
            if review and review['verdict'] != 'reasonable':
                update(status='rejected', failed_stage='initial_review', verdict=review['verdict'])
                return
            update(status='failed', failed_stage='initial_review', error='Interrupted before checkpoint; retained for inspection, no verdict reroll')
            return
        else:
            command = [sys.executable, str(HERE/'evolve_graph.py'), '--initial', str(source),
                       '--output', str(evo), '--trigger-probability', str(manifest['probability']),
                       '--seed', str(record['seed']), *common]
        update(status='evolving', evolution_dir=str(evo.relative_to(root)))
        attempt = len(list(folder.glob('evolve_*.log'))) + 1
        code = invoke(command, folder/f'evolve_{attempt:03d}.log')
        if code and checkpoint.exists():
            update(status='resuming')
            code = invoke([sys.executable, str(HERE/'evolve_graph.py'), '--output', str(evo), '--resume'],
                          folder/f'evolve_{attempt+1:03d}.log')
        if code:
            review = completed_review(evo/'initial_review', load(source))
            if not checkpoint.exists() and review and review['verdict'] != 'reasonable':
                update(status='rejected', failed_stage='initial_review', verdict=review['verdict'])
            else:
                update(status='failed', failed_stage='evolution' if checkpoint.exists() else 'initial_review', exit_code=code)
            return
    summary = load(evo/'summary.json')
    staging = folder/'screening_source'
    staging.mkdir(exist_ok=True)
    for src, dst in [('final.json', name+'.json'), ('final.input.json', name+'.input.json'), ('final.png', name+'.png')]:
        shutil.copy2(evo/src, staging/dst)
    update(status='screening', evolution_dir=str(evo.relative_to(root)), rounds=summary['rounds'], final_nodes=summary['nodes'])
    review = None
    # reuse a completed final verdict before making new calls
    for screen in sorted(folder.glob('screening_[0-9]*')):
        report = screen/f'{name}.check.json'
        if report.exists() and load(report)['status'] == 'completed':
            review = load(report)
            validate_review(review['review'], load(evo/'final.json'))
            if review['input_json_sha256'] != digest(evo/'final.json'):
                raise ValueError('Final graph differs from reviewed graph')
            break
    if review is None:
        offset = max([int(p.name.rsplit('_',1)[1]) for p in folder.glob('screening_[0-9]*')], default=0)
        for attempt in range(offset+1, offset+3):
            screen = folder/f'screening_{attempt:03d}'
            update(screening_dir=str(screen.relative_to(root)))
            invoke([sys.executable, str(STEP1/'check_graphs.py'), '--input', str(staging), '--images', str(staging),
                    '--output', str(screen), *common], folder/f'screen_{attempt:03d}.log')
            report = screen/f'{name}.check.json'
            if report.exists() and load(report)['status'] == 'completed':
                review = load(report)
                break
    if review is None:
        update(status='failed', failed_stage='screening')
        return
    keep = review['local_check']['passed'] and review['review'] == {'verdict':'reasonable','issues':[]}
    update(status='accepted' if keep else 'rejected', screening_dir=str(screen.relative_to(root)), verdict=review['review']['verdict'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='accepted/ directory of step-1 screening')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--probability', type=float, default=0.25, help='Trigger probability of every evolution rule')
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.retries < 0 or not 0 <= args.probability <= 1:
        parser.error('Invalid settings')
    source, root = args.input.resolve(), args.output.resolve()
    accepted = load(source/'manifest.json')
    names = accepted['graphs']
    if not names or len(names) != len(set(names)) or any(Path(n).name != n or not n.startswith('graph_') or not n.endswith('.json') for n in names):
        parser.error('Expected an accepted initial graph manifest')
    fingerprints = {name:digest(source/name) for name in names}
    config = {'input':str(source), 'inputs':fingerprints, 'model':args.model, 'probability':args.probability,
              'timeout':args.timeout, 'retries':args.retries,
              'code':{str(p.relative_to(HERE.parent)):digest(p) for p in
                      [HERE/'evolve_graph.py', HERE/'config.json', HERE/'evolution_rule_prompts.txt',
                       STEP1/'graph_check_prompt.txt', STEP1/'check_graphs.py', STEP1/'graph_io.py']}}
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error('Output exists; use --resume')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if load(root/'config.json') != config:
                parser.error('Resume configuration or inputs changed')
            manifest = load(root/'batch.json')
            manifest.pop('finished_at',None)
            manifest['workers'] = args.workers
        else:
            save(root/'config.json',config)
            (root/'initials').mkdir()
            records = []
            for name in sorted(names,key=lambda n:int(Path(n).stem.split('_')[1])):
                graph=load(source/name)
                stats=validate(graph)
                snapshot=root/'initials'/name
                shutil.copy2(source/name,snapshot)
                records.append({'id':Path(name).stem,'proposals':stats['proposals'],
                                'seed':3000+int(Path(name).stem.split('_')[1]),'status':'queued',
                                'initial_source':str(snapshot.relative_to(root)),
                                'initial_sha256':fingerprints[name],'initial_nodes':len(graph['nodes'])})
            manifest={'count':len(records),'probability':args.probability,'model':args.model,
                      'workers':args.workers,'timeout':args.timeout,'retries':args.retries,
                      'started_at':time.time(),'graphs':records,'input':str(source)}
        manifest['status']='running'
        publish(root,manifest)
        pending=[r for r in manifest['graphs'] if r['status'] not in ('accepted','rejected')]
        print(f'Starting {len(pending)}/{manifest["count"]} graphs with {args.workers} workers',flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(job,root,manifest,r):r for r in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    with LOCK:
                        futures[future].update(status='failed',error=str(exc))
                        publish(root,manifest)
                    print(futures[future]['id'],'FAILED',str(exc),flush=True)
        export_accepted(root,manifest)
        manifest['finished_at']=time.time()
        manifest['status']='incomplete' if any(r['status']=='failed' for r in manifest['graphs']) else 'complete'
        publish(root,manifest)
        # programmatic audit: prefix preservation, requested outcomes and export provenance
        from audit_batch import audit
        audit(root)
        print(root/'accepted/index.html',flush=True)
        if manifest['status']!='complete':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
