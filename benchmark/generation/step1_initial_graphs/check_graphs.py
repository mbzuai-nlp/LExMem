#!/usr/bin/env python3
"""Step 1b: rule-check each skeleton graph, then have the model review its JSON together with a rendered PNG."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from generate_graphs import write_json
from graph_io import validate

HERE = Path(__file__).resolve().parent


def review_schema():
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    edge = obj({key: {'type': 'string'} for key in ('source', 'relation', 'target')})
    issue = obj({'severity': {'type': 'string', 'enum': ['error', 'concern']},
                 'node_ids': {'type': 'array', 'items': {'type': 'string'}},
                 'edge': {'anyOf': [edge, {'type': 'null'}]},
                 'reason': {'type': 'string'}, 'suggestion': {'type': 'string'}})
    return obj({'verdict': {'type': 'string', 'enum': ['reasonable', 'needs_review', 'unreasonable']},
                'issues': {'type': 'array', 'items': issue}})


def build_request(template, graph):
    if template.count('{{GRAPH_JSON}}') != 1:
        raise ValueError('Check prompt must contain exactly one {{GRAPH_JSON}} placeholder')
    return template.replace('{{GRAPH_JSON}}', json.dumps(graph, ensure_ascii=False, indent=2)) + '''

The attached PNG visualizes this same graph. Inspect both the JSON and the image.
Use JSON as the authoritative source for IDs, node types, relations, and order.
Circles show type abbreviations and #order, not node IDs; map them to IDs using JSON.
Dashed gray lines show the exploration tree; labeled colored arrows show semantic relations.
Visual overlap or difficult-to-read labels are not structural graph errors.
Return only the requested review JSON. Do not edit the graph or use tools.
'''


def validate_review(review, graph):
    """Reject malformed reviews, inconsistent verdicts, and invented references.

    This does not certify whether the model's reasoning is correct.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(isinstance(review, dict) and set(review) == {'verdict', 'issues'}, 'Expected verdict and issues')
    require(review['verdict'] in ('reasonable', 'needs_review', 'unreasonable'), 'Invalid verdict')
    require(isinstance(review['issues'], list), 'issues must be an array')
    known_ids = {n['id'] for n in graph['nodes']}
    known_edges = {(e['source'], e['relation'], e['target']) for e in graph['edges']}
    known_ids.update(x for e in graph['edges'] for x in (e['source'], e['target']))
    severities = []
    for issue in review['issues']:
        require(isinstance(issue, dict) and set(issue) == {'severity', 'node_ids', 'edge', 'reason', 'suggestion'}, 'Invalid issue fields')
        require(issue['severity'] in ('error', 'concern'), 'Invalid severity')
        require(isinstance(issue['node_ids'], list) and all(isinstance(n, str) and n in known_ids for n in issue['node_ids']), 'Issue cites nonexistent node IDs')
        require(all(isinstance(issue[k], str) and issue[k].strip() for k in ('reason', 'suggestion')), 'Missing explanation or suggestion')
        edge = issue['edge']
        if edge is not None:
            require(isinstance(edge, dict) and set(edge) == {'source', 'relation', 'target'}, 'Invalid issue edge')
            require(all(isinstance(edge[k], str) for k in edge), 'Invalid edge value')
            require((edge['source'], edge['relation'], edge['target']) in known_edges, 'Issue cites an edge absent from the input')
        severities.append(issue['severity'])
    expected = 'unreasonable' if 'error' in severities else 'needs_review' if severities else 'reasonable'
    require(review['verdict'] == expected, 'Verdict is inconsistent with issue severity')
    return Counter(severities)


def save_summary(out, records, requested_count):
    completed = [r for r in records if r['status'] == 'completed']
    summary = {'requested_count': requested_count, 'completed_count': len(completed),
               'failed_count': sum(r['status'] == 'failed' for r in records),
               'verdict_counts': dict(Counter(r['review']['verdict'] for r in completed)), 'graphs': records}
    write_json(out / 'summary.json', summary)
    lines = ['# Skeleton graph screening', '',
             f'Completed {len(completed)}/{requested_count} graphs.', '',
             '| Graph | Rule check | Model verdict | error | concern |', '|---|---|---|---:|---:|']
    for record in records:
        counts = Counter(i['severity'] for i in record.get('review', {}).get('issues', []))
        verdict = record.get('review', {}).get('verdict', 'call failed')
        local = 'passed' if record['local_check']['passed'] else 'failed'
        lines.append(f'| {record["graph"]} | {local} | {verdict} | {counts["error"]} | {counts["concern"]} |')
    for record in records:
        stem = Path(record['graph']).stem
        lines += ['', f'## {stem}', '', f'[JSON](inputs/{stem}.json) · [PNG](inputs/{stem}.png) · [report]({stem}.check.json)', '']
        if not record['local_check']['passed']:
            lines += [f'Rule check: {record["local_check"]["error"]}', '']
        if record['status'] == 'failed':
            lines += [f'Call failed: {record["failure"]}', '']
            continue
        issues = record['review']['issues']
        if not issues:
            lines += ['No issues reported.', '']
        for i, issue in enumerate(issues, 1):
            edge = issue['edge']
            reference = f'{edge["source"]} — {edge["relation"]} → {edge["target"]}' if edge else ', '.join(issue['node_ids']) or 'whole graph'
            lines += [f'### {i}. {issue["severity"]}: {reference}', '', issue['reason'], '', f'Suggestion: {issue["suggestion"]}', '']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def keep_graph(record):
    return (record.get('status') == 'completed'
            and record.get('local_check', {}).get('passed') is True
            and record.get('review', {}).get('verdict') == 'reasonable'
            and record.get('review', {}).get('issues') == [])


def save_accepted(out, records):
    """Export only accepted JSON/PNG pairs, preserving input and review files."""
    accepted = out / 'accepted'
    selected = [r for r in records if keep_graph(r)]
    files = {}
    for record in selected:
        name = record['graph']
        if not re.fullmatch(r'graph_\d+\.json', name):
            raise ValueError(f'Invalid graph filename: {name}')
        for suffix, hash_key in (('.json', 'input_json_sha256'), ('.png', 'input_png_sha256')):
            source = out / 'inputs' / Path(name).with_suffix(suffix)
            if hashlib.sha256(source.read_bytes()).hexdigest() != record[hash_key]:
                raise ValueError(f'Input changed since review: {source}')
            files[source.name] = source
    if accepted.exists():
        extra = {p.name for p in accepted.iterdir()} - set(files) - {'manifest.json'}
        manifest_path = accepted / 'manifest.json'
        if 'visualizations' in extra and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get('graphs') == [r['graph'] for r in selected]:
                extra.remove('visualizations')
        if extra:
            raise ValueError(f'Accepted directory contains stale files: {sorted(extra)}')
    accepted.mkdir(exist_ok=True)
    for name, source in files.items():
        shutil.copy2(source, accepted / name)
    write_json(accepted / 'manifest.json', {
        'criterion': 'completed + local_check.passed + reasonable + no issues',
        'kept_count': len(selected), 'excluded_count': len(records) - len(selected),
        'graphs': [r['graph'] for r in selected], 'source_summary': '../summary.json',
    })
    print(f'Kept {len(selected)}/{len(records)} graphs: {accepted}', flush=True)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, default=HERE / 'graph_check_prompt.txt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--filter-only', action='store_true', help='Export accepted graphs from existing reports without model calls')
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--retries', type=int, default=1, help='Retries for invalid review format/references, not for negative verdicts')
    args = parser.parse_args()
    if args.filter_only:
        out = args.output.resolve()
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        save_accepted(out, summary['graphs'])
        return
    if args.timeout < 1 or args.retries < 0:
        parser.error('timeout must be positive and retries nonnegative')
    if not shutil.which('codex'):
        parser.error('Codex CLI must be installed and logged in')
    template = args.prompt.read_text(encoding='utf-8')
    paths = sorted(p for p in args.input.glob('graph_*.json') if re.fullmatch(r'graph_\d+\.json', p.name))
    if not paths:
        parser.error(f'No graphs found in {args.input}')
    for path in paths:
        if not (args.images / f'{path.stem}.png').is_file():
            parser.error(f'Missing PNG for {path.name}')
        build_request(template, json.loads(path.read_text(encoding='utf-8')))
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'Refusing to overwrite nonempty directory: {out}')
    (out / 'raw').mkdir(parents=True)
    (out / 'inputs').mkdir()
    (out / 'prompt.txt').write_text(template, encoding='utf-8')
    records = []
    save_summary(out, records, len(paths))
    with tempfile.TemporaryDirectory(prefix='graph-review-') as work:
        schema = Path(work) / 'review.schema.json'
        write_json(schema, review_schema())
        for path in paths:
            stem = path.stem
            graph_file, image_file = out / 'inputs' / path.name, out / 'inputs' / f'{stem}.png'
            shutil.copy2(path, graph_file)
            shutil.copy2(args.images / f'{stem}.png', image_file)
            graph = json.loads(graph_file.read_text(encoding='utf-8'))
            seed_path = path.with_suffix('.input.json')
            seed = json.loads(seed_path.read_text()) if seed_path.exists() else None
            try:
                local = {'passed': True, 'stats': validate(graph, seed_input=seed)}
            except (ValueError, KeyError, TypeError) as exc:
                local = {'passed': False, 'error': str(exc)}
            record = {'graph': path.name, 'status': 'failed', 'local_check': local,
                      'input_json_sha256': hashlib.sha256(graph_file.read_bytes()).hexdigest(),
                      'input_png_sha256': hashlib.sha256(image_file.read_bytes()).hexdigest(),
                      'model_requested': args.model, 'attempts': []}
            error = ''
            for attempt in range(1, args.retries + 2):
                tag = f'{stem}_attempt_{attempt}'
                raw = out / 'raw' / f'{tag}.json'
                request = build_request(template, graph)
                if error:
                    request += f'\nThe previous review format was invalid: {error}. Return a valid review; do not change your judgment just to obtain a positive verdict.\n'
                (out / 'raw' / f'{tag}.prompt.txt').write_text(request, encoding='utf-8')
                command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
                           '-c', 'approval_policy="never"', '--cd', work, '--color', 'never',
                           '--output-schema', str(schema), '--output-last-message', str(raw), '--image', str(image_file)]
                if args.model:
                    command += ['--model', args.model]
                command += ['-']
                attempt_record = {'attempt': attempt, 'command': command, 'response': str(raw.relative_to(out))}
                record['attempts'].append(attempt_record)
                started = time.monotonic()
                print(f'{stem}: reviewing JSON + PNG (attempt {attempt}) ...', flush=True)
                stderr_path = out / 'raw' / f'{tag}.stderr.log'
                try:
                    with (out / 'raw' / f'{tag}.stdout.log').open('w') as stdout, stderr_path.open('w') as stderr:
                        result = subprocess.run(command, input=request, text=True, stdout=stdout, stderr=stderr, timeout=args.timeout)
                    if result.returncode:
                        raise RuntimeError(f'Codex exited with {result.returncode}; see {stderr_path.name}')
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    record['failure'] = str(exc)
                    attempt_record['failure'] = str(exc)
                    break
                finally:
                    attempt_record['seconds'] = round(time.monotonic() - started, 2)
                log = stderr_path.read_text(encoding='utf-8')
                model = re.search(r'^model: (.+)$', log, re.MULTILINE)
                record['model_reported'] = model.group(1) if model else None
                try:
                    review = json.loads(raw.read_text(encoding='utf-8'))
                    counts = validate_review(review, graph)
                except (ValueError, OSError) as exc:
                    error = str(exc)
                    attempt_record['format_error'] = error
                    record['failure'] = error
                    print(f'{stem}: invalid review format: {error}', flush=True)
                    continue
                record.update(status='completed', review=review)
                record.pop('failure', None)
                print(f'{stem}: {review["verdict"]}; {counts["error"]} errors, {counts["concern"]} concerns', flush=True)
                break
            write_json(out / f'{stem}.check.json', record)
            records.append(record)
            save_summary(out, records, len(paths))
    save_accepted(out, records)
    print(f'Report: {out / "summary.md"}', flush=True)
    if any(r['status'] != 'completed' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
