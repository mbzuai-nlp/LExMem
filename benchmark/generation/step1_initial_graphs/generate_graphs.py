#!/usr/bin/env python3
"""Step 1a: generate skeleton graphs (node types, exploration tree, typed relations) with the Codex CLI."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from graph_io import TYPES, RELATIONS, validate

HERE = Path(__file__).resolve().parent


def output_schema():
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    node = obj({'id': {'type': 'string'}, 'type': {'type': 'string', 'enum': list(TYPES)},
                'order': {'type': 'integer'}, 'exploration_parent_id': {'type': ['string', 'null']}})
    edge = obj({'source': {'type': 'string'}, 'relation': {'type': 'string', 'enum': list(RELATIONS)}, 'target': {'type': 'string'}})
    return obj({'nodes': {'type': 'array', 'items': node}, 'edges': {'type': 'array', 'items': edge}})


def build_input(proposal_count):
    if proposal_count not in (3, 4, 5):
        raise ValueError('Input must contain 3–5 proposals')
    return {'goal': {'id': 'g1', 'type': 'goal'},
            'proposals': [{'id': f'p{i}', 'type': 'proposal'} for i in range(1, proposal_count + 1)]}


def build_prompt(template, seed_input):
    if template.count('{{INPUT_JSON}}') != 1:
        raise ValueError('Prompt template must contain exactly one {{INPUT_JSON}} placeholder')
    return template.replace('{{INPUT_JSON}}', json.dumps(seed_input, ensure_ascii=False, indent=2))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def generate_one(index, args, prompt, out, raw, work, schema):
    stem = f'graph_{index:02d}'
    seed_input = build_input(args.proposal_counts[(index - 1) % len(args.proposal_counts)])
    write_json(out / f'{stem}.input.json', seed_input)
    graph_prompt = build_prompt(prompt, seed_input)
    previous = [int(p.name.split('_attempt_')[1].split('.')[0]) for p in raw.glob(f'{stem}_attempt_*.prompt.txt')]
    offset = max(previous, default=0)
    error = ''
    for attempt in range(offset + 1, offset + args.retries + 2):
        tag = f'{stem}_attempt_{attempt}'
        response = raw / f'{tag}.json'
        request = graph_prompt + f'\n\nGenerate independent sample {index} of {args.count}. Return JSON only. Do not use tools, read files, or run commands.\n'
        if error:
            request += f'Your previous attempt failed validation: {error}. Generate a complete valid graph.\n'
        (raw / f'{tag}.prompt.txt').write_text(request, encoding='utf-8')
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check',
                   '--sandbox', 'read-only', '-c', 'approval_policy="never"',
                   '--cd', work, '--color', 'never', '--output-schema', str(schema),
                   '--output-last-message', str(response)]
        if args.model:
            command += ['--model', args.model]
        command += ['-']
        print(f'{stem}: calling the model (attempt {attempt}) ...', flush=True)
        started = time.monotonic()
        with (raw / f'{tag}.stdout.log').open('w') as stdout, (raw / f'{tag}.stderr.log').open('w') as stderr:
            try:
                result = subprocess.run(command, input=request, text=True, stdout=stdout, stderr=stderr, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                error = f'Codex timed out after {args.timeout}s'
                print(f'{stem}: {error}', flush=True)
                continue
        if result.returncode:
            error = f'Codex exited with {result.returncode}; see {tag}.stderr.log'
            print(f'{stem}: {error}', flush=True)
            continue
        try:
            graph = json.loads(response.read_text(encoding='utf-8'))
            stats = validate(graph, seed_input=seed_input)
        except (ValueError, OSError) as exc:
            error = str(exc)
            print(f'{stem}: validation failed: {error}', flush=True)
            continue
        log = (raw / f'{tag}.stderr.log').read_text(encoding='utf-8')
        reported = re.search(r'^model: (.+)$', log, re.MULTILINE)
        entry = {'file': f'{stem}.json', 'input_file': f'{stem}.input.json',
                 'input_proposals': len(seed_input['proposals']), 'raw_response': str(response.relative_to(out)),
                 'model_reported': reported.group(1) if reported else None, 'attempt': attempt,
                 'seconds': round(time.monotonic() - started, 2), **stats}
        # per-graph record, so an interrupted run can be resumed before the manifest is updated
        write_json(raw / f'{stem}.record.json', entry)
        write_json(out / f'{stem}.json', graph)
        return entry
    raise RuntimeError(f'{stem}: failed after {args.retries + 1} new attempts: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--proposal-counts', type=int, nargs='+', choices=(3, 4, 5), default=[3, 4, 5],
                        help='Input proposal counts, cycled across graphs (default: 3 4 5)')
    parser.add_argument('--prompt', type=Path, default=HERE / 'graph_generation_prompt.txt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=300, help='Seconds per model call')
    parser.add_argument('--retries', type=int, default=1, help='Retries after invalid JSON or structural validation failure')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent model calls')
    parser.add_argument('--resume', action='store_true', help='Validate and reuse completed graphs')
    parser.add_argument('--extend', action='store_true', help='Allow --resume to increase the total count')
    args = parser.parse_args()
    if args.extend and not args.resume:
        parser.error('--extend requires --resume')
    if args.count < 1 or args.retries < 0 or args.timeout < 1 or args.workers < 1:
        parser.error('count, timeout and workers must be positive; retries must be nonnegative')
    if not shutil.which('codex'):
        parser.error('codex is not installed. Install and log into Codex CLI first.')
    prompt = args.prompt.read_text(encoding='utf-8')
    build_prompt(prompt, build_input(args.proposal_counts[0]))
    out = args.output.resolve()
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error(f'Refusing to overwrite a nonempty directory: {out}; choose a new --output')
    out.mkdir(parents=True, exist_ok=True)
    run_lock = (out / '.generation.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another generator is already using this output directory')
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)
    manifest = {'proposal_count_schedule': args.proposal_counts, 'backend': 'codex exec', 'model_requested': args.model,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'requested_count': args.count, 'graphs': []}
    completed = set()
    if args.resume:
        saved = json.loads((out / 'manifest.json').read_text())
        old_count = saved['requested_count']
        if args.count != old_count and not (args.extend and args.count > old_count):
            parser.error('Changing requested_count requires --resume --extend and a larger --count')
        for key in ('version', 'proposal_count_schedule', 'backend', 'model_requested', 'prompt_sha256'):
            if saved.get(key) != manifest[key]:
                parser.error(f'Resume configuration mismatch: {key}')
        manifest = saved
        for entry in manifest['graphs']:
            index = int(entry['file'].split('_')[1].split('.')[0])
            seed = build_input(args.proposal_counts[(index - 1) % len(args.proposal_counts)])
            if not 1 <= index <= args.count or entry['file'] in completed:
                parser.error('Invalid or duplicate graph in manifest')
            if json.loads((out / entry['input_file']).read_text()) != seed:
                parser.error(f'Seed mismatch: {entry["file"]}')
            validate(json.loads((out / entry['file']).read_text()), seed_input=seed)
            completed.add(entry['file'])
        for index in range(1, args.count + 1):
            stem = f'graph_{index:02d}'
            graph_path = out / f'{stem}.json'
            record_path = raw / f'{stem}.record.json'
            if graph_path.name not in completed and graph_path.exists():
                if not record_path.exists():
                    parser.error(f'Untracked graph without provenance: {graph_path}')
                seed = build_input(args.proposal_counts[(index - 1) % len(args.proposal_counts)])
                validate(json.loads(graph_path.read_text()), seed_input=seed)
                entry = json.loads(record_path.read_text())
                manifest['graphs'].append(entry)
                completed.add(graph_path.name)
    else:
        (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    if args.resume and args.count > old_count:
        snapshot = out / f'manifest_before_extension_{old_count}_to_{args.count}.json'
        if snapshot.exists():
            parser.error(f'Extension snapshot already exists: {snapshot}')
        write_json(snapshot, manifest)
        manifest.setdefault('extensions', []).append({'previous_count': old_count, 'requested_count': args.count})
        manifest['requested_count'] = args.count
    pending = [i for i in range(1, args.count + 1) if f'graph_{i:02d}.json' not in completed]
    manifest.update(workers=args.workers, status='running', failures=[])
    write_json(out / 'manifest.json', manifest)
    print(f'Resuming {len(completed)}/{args.count}; {len(pending)} pending, {args.workers} workers', flush=True)
    with tempfile.TemporaryDirectory(prefix='graph-generation-') as work:
        schema = Path(work) / 'schema.json'
        write_json(schema, output_schema())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_one, index, args, prompt, out, raw, work, schema): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    entry = future.result()
                except Exception as exc:
                    manifest['failures'].append({'index': index, 'error': str(exc)})
                    print(f'graph_{index:02d}: FAILED: {exc}', flush=True)
                else:
                    manifest['graphs'].append(entry)
                    manifest['graphs'].sort(key=lambda e: int(e['file'].split('_')[1].split('.')[0]))
                    print(f"[{len(manifest['graphs'])}/{args.count}] {entry['file']}: {entry['nodes']} nodes, {entry['edges']} edges", flush=True)
                write_json(out / 'manifest.json', manifest)
    manifest['status'] = 'complete' if len(manifest['graphs']) == args.count else 'incomplete'
    write_json(out / 'manifest.json', manifest)
    run_lock.close()
    print(f'Generated {len(manifest["graphs"])} validated graphs in {out}', flush=True)
    if manifest['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
