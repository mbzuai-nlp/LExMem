"""Shared I/O, token counting and LLM calls (OpenAI-compatible Chat Completions)."""
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

_ENCODER = None


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


def count_tokens(text):
    """Token count used for context budgets (tiktoken cl100k_base)."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding('cl100k_base')
    return len(_ENCODER.encode(text))


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def build_prompt(template, payload):
    """Fill {{INPUT_JSON}} when present; otherwise append the JSON payload."""
    if '{{INPUT_JSON}}' in template:
        return template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False))
    return template + '\n\n' + json.dumps(payload, ensure_ascii=False)


def strip_fence(text):
    """Tolerate a ```json fence or a leading <think>...</think> block around the JSON reply."""
    text = text.strip()
    think = re.match(r'\A<think>[\s\S]*?</think>\s*', text)
    if think:
        text = text[think.end():].strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            text = match.group(1)
    return text


def redact(value, key):
    if not key:
        return value
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, key) for k, v in value.items()}
    return value


def endpoint_url(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('base_url must be an HTTP(S) API prefix')
    url = base_url.rstrip('/')
    return url if url.endswith('/chat/completions') else url + '/chat/completions'


def http_chat_call(base_url, api_key, model, prompt, timeout, json_mode=False, max_tokens=16000):
    """One user-turn call (temperature 0, seed 42). Returns (content, usage, finish_reason)."""
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
               'seed': 42, 'temperature': 0, 'stream': False, 'max_tokens': max_tokens}
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    request = urllib.request.Request(
        endpoint_url(base_url), data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read().decode())
    choice = raw['choices'][0]
    finish_reason = choice.get('finish_reason')
    if finish_reason == 'length':
        raise ValueError('Truncated output (finish_reason=length)')
    content = choice['message']['content']
    if not isinstance(content, str):
        raise ValueError('Model message content must be a string')
    return content, (raw.get('usage') or {}), finish_reason


def repair_prompt(base_prompt, feedback):
    """Retry prompt: the original prompt plus the previous validation error."""
    return (base_prompt + '\n\nYour previous reply was invalid: ' + feedback +
            '\n\nReturn corrected JSON only, following all the instructions above.')


def invoke_llm(folder, template, payload, *, model, timeout, attempts, validator,
               base_url, api_key, json_mode=False, max_tokens=16000):
    """Build the prompt, call the model, validate the JSON reply; retry up to `attempts` times.

    Every call is stored under `folder` (prompt, responses with token usage and latency, errors);
    a validated result.json is reused on reruns.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base_prompt = build_prompt(template, payload)
    (folder / 'prompt.txt').write_text(base_prompt)
    if (folder / 'result.json').exists():
        value = read(folder / 'result.json')
        validator(value)
        return value
    previous = [int(p.stem.split('_')[-1]) for p in folder.glob('response_*.json')]
    start = 1 + max(previous, default=0)
    errors = []
    feedback = None
    if previous and (folder / f'error_{max(previous)}.json').exists():
        feedback = read(folder / f'error_{max(previous)}.json')['error']
    for attempt in range(start, start + attempts):
        prompt = base_prompt if feedback is None else repair_prompt(base_prompt, feedback)
        try:
            started = time.time()
            content, usage, finish_reason = http_chat_call(
                base_url, api_key, model, prompt, timeout, json_mode, max_tokens=max_tokens)
            write(folder / f'response_{attempt}.json',
                  {'content': redact(content, api_key), 'usage': usage, 'finish_reason': finish_reason,
                   'seconds': round(time.time() - started, 3)})
            value = json.loads(strip_fence(content))
            validator(value)
            write(folder / 'result.json', value)
            return value
        except urllib.error.HTTPError as exc:
            error = redact(f'HTTP {exc.code}: {exc.read().decode(errors="replace")[:2000]}', api_key)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            error = redact(f'{type(exc).__name__}: {exc}', api_key)
        errors.append(error)
        write(folder / f'error_{attempt}.json', {'error': error})
        feedback = error
    raise RuntimeError('; '.join(errors))
