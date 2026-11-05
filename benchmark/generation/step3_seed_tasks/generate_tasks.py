#!/usr/bin/env python3
"""Step 3: generate seed tasks (task description + 1-2 measurable requirements), one independent call per assignment.

Each assignment fixes the task type and a topic hint ("auto" lets the model choose the field).
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
TYPES = ['research_idea_exploration', 'software_development']
SCHEMA = {'type': 'object', 'additionalProperties': False,
          'required': ['task_type', 'task_description', 'requirements'],
          'properties': {'task_type': {'type': 'string', 'enum': TYPES},
                         'task_description': {'type': 'string'},
                         'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                                          'items': {'type': 'string'}}}}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate(value, expected_type=None):
    """Structural checks: fields, type, word limits (description <= 20, requirement <= 15), a numeric target."""
    if not isinstance(value, dict) or set(value) != set(SCHEMA['required']):
        raise ValueError('Expected exactly task_type, task_description and requirements')
    if value['task_type'] not in TYPES:
        raise ValueError('Invalid task_type')
    if not isinstance(value['task_description'], str) or not value['task_description'].strip():
        raise ValueError('Empty task description')
    if expected_type and value['task_type'] != expected_type:
        raise ValueError('Task type does not match assignment')
    description_limit, requirement_limit, count_limit = 20, 15, 2
    if len(value['task_description'].split()) > description_limit:
        raise ValueError(f'Description exceeds {description_limit} words')
    requirements = value['requirements']
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= count_limit:
        raise ValueError(f'Expected 1–{count_limit} requirements')
    if any(not isinstance(item, str) or not item.strip() for item in requirements):
        raise ValueError('Requirements must be nonempty strings')
    if any(len(item.split()) > requirement_limit for item in requirements):
        raise ValueError(f'Requirement exceeds {requirement_limit} words')
    if not any(c.isdigit() for item in requirements for c in item):
        raise ValueError('Missing numeric acceptance target')
    if len(set(item.strip().casefold() for item in requirements)) != len(requirements):
        raise ValueError('Duplicate requirements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', type=Path, default=HERE / 'task_prompt.txt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--assignments', type=Path, default=HERE / 'topic_assignments.json')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--attempts', type=int, default=3)
    args = parser.parse_args()
    if min(args.count, args.workers, args.timeout, args.attempts) < 1:
        parser.error('Counts, timeout and attempts must be positive')
    if not shutil.which('codex'):
        parser.error('An authenticated Codex CLI is required')
    prompt = args.prompt.read_text(encoding='utf-8')
    assignments = json.loads(args.assignments.read_text(encoding='utf-8'))
    if not isinstance(assignments, list) or len(assignments) < args.count:
        parser.error('Provide at least count assignments')
    assignments = assignments[:args.count]
    for item in assignments:
        if (not isinstance(item, dict) or set(item) != {'task_type', 'topic'}
                or item['task_type'] not in TYPES or not isinstance(item['topic'], str) or not item['topic'].strip()):
            parser.error('Each assignment needs an allowed task_type and nonempty topic')
    if '{task_type}' not in prompt or '{topic}' not in prompt:
        parser.error('Prompt must contain task_type and topic placeholders')
    config = {'assignments': assignments, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'model_requested': args.model,
              'backend': 'codex-exec', 'schema': SCHEMA, 'word_limits': {'description': 20, 'requirement': 15}}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output.resolve() / fingerprint
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'schema.json', SCHEMA)
    (output / 'prompt.txt').write_text(prompt, encoding='utf-8')

    def generate(index):
        folder = output / f'{index:03d}'
        folder.mkdir(exist_ok=True)
        assignment = assignments[index - 1]
        actual_prompt = prompt.replace('{task_type}', assignment['task_type']).replace('{topic}', assignment['topic'])
        (folder / 'prompt.txt').write_text(actual_prompt, encoding='utf-8')
        write_json(folder / 'assignment.json', assignment)
        final = folder / 'sample.json'
        if final.exists():
            try:
                sample = json.loads(final.read_text(encoding='utf-8'))
                validate(sample, assignment['task_type'])
                return index, sample
            except (ValueError, TypeError):
                pass
        errors = []
        for attempt in range(1, args.attempts + 1):
            tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            raw = folder / f'response_{tag}.txt'
            log_path = folder / f'codex_{tag}.log'
            command = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral',
                       '--sandbox', 'read-only', '--disable', 'shell_tool', '--disable', 'apps',
                       '--disable', 'hooks', '--disable', 'skill_search', '--color', 'never',
                       '--output-schema', str(output / 'schema.json'),
                       '--output-last-message', str(raw)]
            if args.model:
                command += ['--model', args.model]
            print(f'Generating {index:03d}, attempt {attempt}', flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix='seed_task_') as work:
                    with log_path.open('w', encoding='utf-8') as log:
                        subprocess.run(command + ['-'], input=actual_prompt, text=True, cwd=work,
                                       stdout=log, stderr=log, timeout=args.timeout, check=True)
                sample = json.loads(raw.read_text(encoding='utf-8'))
                validate(sample, assignment['task_type'])
                write_json(folder / 'metadata.json', {
                    'sample_index': index, 'assignment': assignment, 'generated_at': tag,
                    'rendered_prompt_sha256': hashlib.sha256(actual_prompt.encode()).hexdigest(),
                    'model_requested': config['model_requested'], 'model_log': log_path.name,
                    'prompt_sha256': config['prompt_sha256'], 'structural_validation': 'passed',
                    'semantic_validation': 'not_automatically_checked'})
                write_json(final, sample)
                print(f'Saved {index:03d}: {sample["task_type"]}', flush=True)
                return index, sample
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                errors.append({'attempt': attempt, 'error': str(exc), 'log': log_path.name})
                write_json(folder / f'errors_{tag}.json', errors)
                print(f'Failed {index:03d}: {exc}', flush=True)
        return index, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(generate, range(1, args.count + 1)))
    samples = [sample for _, sample in rows if sample is not None]
    write_json(output / 'tasks.json', samples)
    (output / 'tasks.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples), encoding='utf-8')
    lines = ['# Generated seed tasks', '']
    for index, sample in rows:
        if sample is None:
            continue
        lines += [f'## {index:03d} — {sample["task_type"]}', '', sample['task_description'], '']
        lines += [f'- {item}' for item in sample['requirements']]
        lines.append('')
    (output / 'tasks.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output / 'summary.json', {'requested': args.count, 'completed': len(samples),
               'failed_indices': [i for i, s in rows if s is None],
               'type_counts': dict(Counter(s['task_type'] for s in samples)),
               'exact_duplicate_count': len(samples) - len({json.dumps(s, sort_keys=True) for s in samples})})
    print(f'Completed {len(samples)}/{args.count}: {output}', flush=True)
    return 0 if len(samples) == args.count else 1


if __name__ == '__main__':
    raise SystemExit(main())
