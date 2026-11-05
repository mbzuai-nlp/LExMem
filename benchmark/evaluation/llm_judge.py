#!/usr/bin/env python3
"""LLM-judged answer accuracy (ACC).

For each question the judge model sees the question, the gold answer split into required parts
(canonical answer + aliases per part) and the predicted answer items, and returns one grade:
yes / partial / no = 1 / 0.5 / 0 (see judge_prompt.txt). ACC is the mean over questions.
Missing predictions, failed predictions and empty answers score 0 without a judge call.

Predictions: a JSON list (or JSONL file) of rows {"question_id": ..., "prediction": {"answer_items": [...]}};
rows with a "status" other than "ok" count as failed. A directory is read as <dir>/results.json.

Output (resumable): questions/<question_id>/{request.json, response_N.json, result.json},
results.json, summary.json (ACC overall and per question type, judge token usage), config.json.

    JUDGE_API_KEY=... python llm_judge.py --predictions method_run/results.json --output judge_out \
        --base-url https://api.openai.com/v1 --model-name gpt-4o-mini --json-mode
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
CREDIT = {'yes': 1.0, 'partial': 0.5, 'no': 0.0}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def endpoint_url(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('base_url must be an HTTP(S) API prefix')
    url = base_url.rstrip('/')
    return url if url.endswith('/chat/completions') else url + '/chat/completions'


def http_call(endpoint, key, payload, timeout):
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
                                     method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, list):
        return [redact(x, key) for x in value]
    if isinstance(value, dict):
        return {k: redact(v, key) for k, v in value.items()}
    return value


def load_predictions(path):
    if path.is_dir():
        path = path / 'results.json'
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def answer_items(row):
    """The predicted items, or None when the prediction itself failed."""
    if row.get('status') not in (None, 'ok') or not isinstance(row.get('prediction'), dict):
        return None
    items = row['prediction'].get('answer_items')
    return [x for x in items if isinstance(x, str) and x.strip()] if isinstance(items, list) else None


def build_request(qa, items, template, args):
    parts = [{'id': k, 'answer': v['canonical'], 'aliases': v.get('aliases', [])}
             for k, v in qa['answer_keywords'].items()]
    data = {'question': qa['question'], 'correct_answer_parts': parts, 'model_response': items}
    payload = {'model': args.model_name, 'seed': 42, 'temperature': 0, 'stream': False, 'max_tokens': args.max_tokens,
               'messages': [{'role': 'user', 'content': template + '\n' + json.dumps(data, ensure_ascii=False, indent=1)}]}
    if args.json_mode:
        payload['response_format'] = {'type': 'json_object'}
    return payload


def parse_verdict(content):
    text = re.sub(r'^<think>[\s\S]*?</think>\s*', '', content.strip())
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)
    value = json.loads(text)
    label = str(value['correct']).strip().lower()
    if label not in CREDIT:
        raise ValueError(f'Bad verdict {label!r}')
    return CREDIT[label], value


def judge_one(qa, row, template, args, key):
    qid = qa['question_id']
    folder = args.output / 'questions' / qid
    base = {'question_id': qid, 'dialogue_id': qa['dialogue_id'], 'method': qa.get('method')}
    items = None if row is None else answer_items(row)
    if not items:
        reason = 'missing_prediction' if row is None else 'prediction_error' if items is None else 'empty_answer'
        return {**base, 'status': 'ok', 'judged': False, 'reason': reason, 'acc': 0.0, 'prediction': items or []}
    payload = build_request(qa, items, template, args)
    fp = digest(payload)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        cached = read(folder / 'result.json')
        if cached.get('status') == 'ok' and cached.get('fingerprint') == fp:
            return cached
        for old in folder.glob('response_*.json'):
            old.unlink()
    write(folder / 'request.json', payload)
    error = None
    for attempt in range(1, args.attempts + 1):
        t0 = time.time()
        permanent = False
        try:
            raw = http_call(args.endpoint, key, payload, args.timeout)
            seconds = time.time() - t0
            write(folder / f'response_{attempt}.json', {**redact(raw, key), 'seconds': seconds})
            choice = raw['choices'][0]
            if choice.get('finish_reason') == 'length':  # retry with twice the output budget
                payload = {**payload, 'max_tokens': payload['max_tokens'] * 2}
                raise ValueError('Truncated output (finish_reason=length)')
            credit, value = parse_verdict(choice['message']['content'])
            result = {**base, 'status': 'ok', 'judged': True, 'acc': credit, 'judge_output': value,
                      'prediction': items, 'fingerprint': fp, 'attempts': attempt,
                      'usage': raw.get('usage'), 'seconds': seconds}
            write(folder / 'result.json', result)
            return result
        except urllib.error.HTTPError as exc:
            error = f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:2000]}'
            permanent = exc.code not in (408, 409, 425, 429) and exc.code < 500
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        if permanent:
            break
        if attempt < args.attempts:
            time.sleep(min(2 ** attempt, 8))
    result = {**base, 'status': 'error', 'error': redact(error, key), 'fingerprint': fp}
    write(folder / 'result.json', result)
    return result


def summarize(results, total):
    ok = [r for r in results if r['status'] == 'ok']
    groups = defaultdict(list)
    for r in ok:
        groups[r.get('method')].append(r['acc'])
    usage = Counter()
    for r in ok:
        for k in ('prompt_tokens', 'completion_tokens'):
            usage[k] += (r.get('usage') or {}).get(k, 0)
    return {'total_questions': total, 'finished': len(results),
            'counts': dict(Counter(r['status'] for r in results)),
            'not_judged': dict(Counter(r['reason'] for r in ok if not r['judged'])),
            'acc': sum(r['acc'] for r in ok) / len(ok) if ok else None,
            'acc_errors_as_zero': sum(r['acc'] for r in ok) / len(results) if results else None,
            'grades': dict(Counter(str(r['acc']) for r in ok if r['judged'])),
            'acc_by_method': {k: {'n': len(v), 'acc': sum(v) / len(v)}
                              for k, v in sorted(groups.items(), key=lambda x: str(x[0]))},
            'judge_tokens': dict(usage)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--qa', type=Path, default=HERE.parent / 'data' / 'qa.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default=os.environ.get('JUDGE_BASE_URL'))
    p.add_argument('--model-name', default=os.environ.get('JUDGE_MODEL_NAME', 'gpt-4o-mini'))
    p.add_argument('--api-key-env', default='JUDGE_API_KEY', help='environment variable holding the API key')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--json-mode', action='store_true', help='request response_format=json_object')
    p.add_argument('--prompt', type=Path, default=HERE / 'judge_prompt.txt')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if not args.base_url:
        p.error('Supply --base-url (or JUDGE_BASE_URL)')
    key = os.environ.get(args.api_key_env)
    if not key:
        p.error(f'Set the API key in ${args.api_key_env}')
    args.endpoint = endpoint_url(args.base_url)
    qas = read(args.qa)[:args.limit] if args.limit else read(args.qa)
    rows = {r['question_id']: r for r in load_predictions(args.predictions)}
    missing = sum(q['question_id'] not in rows for q in qas)
    if missing:
        print(f'WARNING: {missing} questions have no prediction; they score 0', flush=True)
    template = args.prompt.read_text().rstrip('\n')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'config.json', {'predictions': str(args.predictions), 'qa': str(args.qa),
                                        'model_name': args.model_name, 'seed': 42, 'temperature': 0,
                                        'max_tokens': args.max_tokens, 'json_mode': args.json_mode,
                                        'prompt': template})
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge_one, qa, rows.get(qa['question_id']), template, args, key) for qa in qas]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0 or len(results) == len(qas):
                write(args.output / 'results.json', sorted(results, key=lambda r: r['question_id']))
                write(args.output / 'summary.json', summarize(results, len(qas)))
    print(json.dumps(read(args.output / 'summary.json'), ensure_ascii=False, indent=1), flush=True)
    if any(r['status'] == 'error' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
