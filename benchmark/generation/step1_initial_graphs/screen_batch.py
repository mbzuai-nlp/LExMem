#!/usr/bin/env python3
"""Step 1b (batch): render and screen every generated skeleton graph in parallel (resumable per graph).

Accepted graphs (rule check passed, model verdict "reasonable" with no issues) are exported to <output>/accepted/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from check_graphs import keep_graph, save_accepted, save_summary, validate_review
from generate_graphs import write_json
from graph_io import validate

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(path, out, model, timeout, retries):
    job = out / 'jobs' / path.stem
    source = job / 'source'
    images = job / 'visualizations'
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, source / path.name)
    seed = path.with_suffix('.input.json')
    if seed.exists():
        shutil.copy2(seed, source / seed.name)
    # reuse a completed verdict from an interrupted run
    for previous_report in sorted(job.glob('review_*/' + path.stem + '.check.json')):
        saved = json.loads(previous_report.read_text())
        if saved.get('status') == 'completed':
            return collect_record(path, out, previous_report.parent)
    with (job / 'visualize.log').open('w') as log:
        result = subprocess.run([sys.executable, str(HERE / 'visualize_graphs.py'),
                                 '--input', str(source), '--output', str(images)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode:
        raise RuntimeError(f'Visualization failed: {job / "visualize.log"}')
    # a new directory per technical retry; a completed verdict is never re-requested
    previous = [int(p.name.split('_')[-1]) for p in job.glob('review_*') if p.is_dir()]
    checks = job / f'review_{max(previous, default=0) + 1:03d}'
    with (job / f'{checks.name}.log').open('w') as log:
        result = subprocess.run([sys.executable, str(HERE / 'check_graphs.py'),
                                 '--input', str(source), '--images', str(images),
                                 '--output', str(checks), '--model', model,
                                 '--timeout', str(timeout), '--retries', str(retries)],
                                stdout=log, stderr=subprocess.STDOUT,
                                timeout=(timeout + 30) * (retries + 1) + 120)
    report = checks / f'{path.stem}.check.json'
    if not report.exists():
        raise RuntimeError(f'Checker exited {result.returncode}: {job / (checks.name + ".log")}')
    return collect_record(path, out, checks)


def collect_record(path, out, checks):
    report = checks / f'{path.stem}.check.json'
    record = json.loads(report.read_text())
    if record['input_json_sha256'] != digest(path):
        raise ValueError(f'Changed input: {path}')
    for suffix, key in (('.json', 'input_json_sha256'), ('.png', 'input_png_sha256')):
        if digest(checks / 'inputs' / f'{path.stem}{suffix}') != record[key]:
            raise ValueError(f'Changed input snapshot: {path.stem}{suffix}')
    if record['status'] == 'completed':
        validate_review(record['review'], json.loads(path.read_text()))
    for suffix in ('.json', '.png'):
        shutil.copy2(checks / 'inputs' / f'{path.stem}{suffix}', out / 'inputs' / f'{path.stem}{suffix}')
    for attempt in record['attempts']:
        attempt['response'] = str((checks / attempt['response']).relative_to(out))
    record['source_report'] = str(report.relative_to(out))
    write_json(out / f'{path.stem}.check.json', record)
    return record


def run_batch(args):
    source, out = args.input.resolve(), args.output.resolve()
    paths = sorted((p for p in source.glob('graph_*.json') if re.fullmatch(r'graph_\d+\.json', p.name)),
                   key=lambda p: int(p.stem.split('_')[1]))
    if not paths:
        raise ValueError('No graph files found')
    config = {'model': args.model, 'timeout': args.timeout, 'retries': args.retries,
              'source': str(source), 'graphs': {p.name: digest(p) for p in paths},
              'seeds': {p.name: digest(p.with_suffix('.input.json')) if p.with_suffix('.input.json').exists() else None for p in paths},
              'checker_files': {name: digest(HERE / name) for name in
                                ('check_graphs.py', 'graph_check_prompt.txt', 'graph_io.py', 'visualize_graphs.py')}}
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Nonempty output; use --resume')
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.screening.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = out / 'config.json'
        if args.resume:
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Resume input, model or checker configuration changed')
        else:
            write_json(config_path, config)
        (out / 'inputs').mkdir(exist_ok=True)
        (out / 'jobs').mkdir(exist_ok=True)
        shutil.copy2(HERE / 'graph_check_prompt.txt', out / 'prompt.txt')
        records = {}
        pending = []
        for path in paths:
            record_path = out / f'{path.stem}.check.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record['status'] == 'completed':
                    graph = json.loads(path.read_text())
                    validate_review(record['review'], graph)
                    if record['input_json_sha256'] != config['graphs'][path.name]:
                        raise ValueError(f'Changed input: {path}')
                    for suffix, key in (('.json', 'input_json_sha256'), ('.png', 'input_png_sha256')):
                        if digest(out / 'inputs' / f'{path.stem}{suffix}') != record[key]:
                            raise ValueError(f'Changed review snapshot: {path.stem}{suffix}')
                    records[path.name] = record
                    continue
            pending.append(path)
        def publish(status):
            ordered = [records[p.name] for p in paths if p.name in records]
            save_summary(out, ordered, len(paths))
            if status != 'running':
                save_accepted(out, ordered)
            write_json(out / 'progress.json', {
                'status': status, 'requested_count': len(paths), 'processed_count': len(ordered),
                'completed_count': sum(r['status'] == 'completed' for r in ordered),
                'accepted_count': sum(keep_graph(r) for r in ordered),
                'failed_count': sum(r['status'] != 'completed' for r in ordered),
                'workers': args.workers,
            })
        publish('running')
        print(f'{len(records)} reused, {len(pending)} pending, {args.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, p, out, args.model, args.timeout, args.retries): p for p in pending}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    seed_path = path.with_suffix('.input.json')
                    try:
                        stats = validate(json.loads(path.read_text()), seed_input=json.loads(seed_path.read_text()) if seed_path.exists() else None)
                        local = {'passed': True, 'stats': stats}
                    except (ValueError, KeyError, TypeError) as validation_error:
                        local = {'passed': False, 'error': str(validation_error)}
                    record = {'graph': path.name, 'status': 'failed', 'failure': str(exc), 'local_check': local}
                    write_json(out / f'{path.stem}.check.json', record)
                records[path.name] = record
                publish('running')
                print(f'[{len(records)}/{len(paths)}] {path.name}: {record.get("review", {}).get("verdict", "failed")}', flush=True)
        failed = any(r['status'] != 'completed' for r in records.values())
        publish('incomplete' if failed else 'complete')
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.retries < 0:
        parser.error('Invalid workers, timeout or retries')
    raise SystemExit(run_batch(args))


if __name__ == '__main__':
    main()
