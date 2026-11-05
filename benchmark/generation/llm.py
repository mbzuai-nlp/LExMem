"""Chat-completions interface shared by steps 4-7.

Settings come from a JSON config (path in $LLM_CONFIG, default configs/stages4to6.json) with two call groups,
"generation" and "filter". API keys are read from the environment variable named in the config. If $TOKEN_LOG is set,
every call appends one JSON line (step, model, token usage, latency) to that file.
"""
import fcntl, json, os, re, threading, time
from pathlib import Path

from openai import (APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, OpenAI,
                    PermissionDeniedError, RateLimitError)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads(Path(os.environ.get('LLM_CONFIG', HERE / 'configs' / 'stages4to6.json')).read_text())

# (prompt prefix, step, group): every prompt is routed to the "generation" or "filter" model by its first words
STEPS = [
    ('Fill the current node with concrete content', '4.1 node generation', 'generation'),
    ('NEXT GRAPH NODE', '4.1 node generation', 'generation'),
    ('NEXT GRAPH CHECK', '4.2 node check', 'filter'),
    ("Check the current node's content against", '4.2 node check', 'filter'),
    ('Check whether a complete instantiated graph', '4.3 graph check', 'filter'),
    ('Divide the supplied instantiated graph', '5 session partition', 'filter'),
    ('Expand the current node into an English', '6.1 dialogue generation', 'generation'),
    ('NEXT NODE', '6.1 dialogue generation', 'generation'),
    ('Check a newly generated English dialogue fragment', '6.2 dialogue node check', 'filter'),
    ('NEXT CHECK', '6.2 dialogue node check', 'filter'),
    ('Check a completed dialogue against', '6.3 dialogue full check', 'filter'),
    ('Generate exactly one question', '7.1 QA generation', 'generation'),
    ('Check whether this QA is well formed', '7.2 question quality check', 'filter'),
    ('Check whether the supplied dialogue supports', '7.3 dialogue answerability check', 'filter'),
    ('Locate necessary dialogue evidence', '7.4 evidence localization', 'filter'),
    ('Verify the proposed dialogue evidence', '7.4 evidence check', 'filter'),
]


class LLMError(OSError):
    """A call could not be completed (subclass of OSError so the step code treats it as a failed attempt)."""


class PauseRun(BaseException):
    """Stop the run and keep its checkpoint (service unavailable after retries, quota exhausted, or bad key).

    Not an Exception subclass, so step-level `except Exception` handlers do not turn it into a failed node attempt.
    """


PAUSED = threading.Event()
_pause_reason = ['']


def _pause(step, group, cfg, kind, detail):
    _pause_reason[0] = _pause_reason[0] or f'{kind}: {detail}'
    if not PAUSED.is_set():
        _log({'ts': time.time(), 'step': step, 'group': group, 'model': cfg['model'], 'ok': False, 'error': kind})
    PAUSED.set()
    raise PauseRun(_pause_reason[0])


def run_guarded(main):
    """Run a step's main(); a pause exits with code 75 so the run can be restarted from its checkpoint."""
    try:
        return main()
    except PauseRun as exc:
        print(f'PAUSED (checkpoint kept; rerun the same command to continue): {exc}', flush=True)
        return 75


def models():
    """Model names of both call groups (recorded in run configurations)."""
    return {group: CONFIG[group]['model'] for group in ('generation', 'filter')}


_clients, _lock = {}, threading.Lock()


def classify(prompt):
    for prefix, step, group in STEPS:
        if prompt.startswith(prefix):
            return step, group
    raise LLMError('Unrecognised prompt; add it to llm.STEPS: ' + prompt[:60])


def _client(group):
    with _lock:
        if group not in _clients:
            cfg = CONFIG[group]
            _clients[group] = OpenAI(base_url=cfg['base_url'], api_key=os.environ[cfg['api_key_env']],
                                     timeout=cfg.get('timeout', 900), max_retries=0)
        return _clients[group]


def _log(record):
    path = os.environ.get('TOKEN_LOG')
    if not path:
        return
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(record) + '\n'); fcntl.flock(f, fcntl.LOCK_UN)


def call(prompt, schema=None, history=None, cache_key=None):
    """Send one prompt and return the message text.

    schema: optional JSON schema for strict structured output.
    history: optional earlier {"role", "content"} messages sent before `prompt` (used by the append mode of steps 4
    and 6, where one graph is processed as a growing multi-turn conversation so the shared prefix can be cached).
    """
    step, group = classify(prompt)
    if PAUSED.is_set():
        raise PauseRun(_pause_reason[0])
    cfg = CONFIG[group]
    kwargs = {'model': cfg['model'], 'messages': list(history or []) + [{'role': 'user', 'content': prompt}],
              'max_completion_tokens': cfg.get('max_completion_tokens', 32000)}
    if cfg.get('temperature') is not None:
        kwargs['temperature'] = cfg['temperature']
    if cfg.get('reasoning_effort'):
        kwargs['reasoning_effort'] = cfg['reasoning_effort']
    if cache_key and cfg.get('prompt_cache_key', True):
        kwargs['prompt_cache_key'] = cache_key
    if cfg.get('service_tier'):
        kwargs['service_tier'] = cfg['service_tier']
    if schema is not None:
        kwargs['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'out', 'schema': schema, 'strict': True}}
    started, transient, notes = time.time(), 0, []
    flex = kwargs.get('service_tier') == 'flex'
    # When the flex tier stays unavailable after `transient_retries` backoffs: "pause" stops the run (default),
    # "standard" resends that request on the standard tier after `flex_fallback_after` failures.
    on_unavailable = cfg.get('flex_on_unavailable', 'pause')
    fallback_after = cfg.get('flex_fallback_after', 3)
    retries = cfg.get('transient_retries', 5)
    while True:
        if PAUSED.is_set():
            raise PauseRun(_pause_reason[0])
        try:
            response = _client(group).chat.completions.create(**kwargs)
            break
        except (AuthenticationError, PermissionDeniedError) as exc:
            _pause(step, group, cfg, 'AuthError', str(exc)[:200])
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            if isinstance(exc, RateLimitError) and ('insufficient_quota' in str(exc) or 'spend limit' in str(exc).lower()):
                _pause(step, group, cfg, 'QuotaExhausted', str(exc)[:200])
            if flex and on_unavailable == 'standard' and transient >= fallback_after and kwargs.get('service_tier') == 'flex':
                kwargs['service_tier'] = 'auto'; notes.append('flex_fallback_to_standard'); transient = 0; continue
            if transient >= retries:
                _pause(step, group, cfg, 'FlexUnavailable' if flex else 'ApiUnavailable',
                       f'{type(exc).__name__} after {retries} backoffs: {str(exc)[:150]}')
            transient += 1
            PAUSED.wait((10 * 2 ** transient) if flex else (5 * 3 ** transient))
        except BadRequestError as exc:
            # drop an optional parameter the endpoint rejects, and record it in the log
            text = str(exc).lower()
            if 'temperature' in text and 'temperature' in kwargs:
                del kwargs['temperature']; notes.append('temperature_dropped'); continue
            if 'prompt_cache_key' in text and 'prompt_cache_key' in kwargs:
                del kwargs['prompt_cache_key']; notes.append('prompt_cache_key_dropped'); continue
            if 'service_tier' in text and 'service_tier' in kwargs:
                del kwargs['service_tier']; notes.append('service_tier_dropped'); flex = False; continue
            if 'response_format' in text and 'response_format' in kwargs:
                del kwargs['response_format']; notes.append('schema_dropped'); continue
            _log({'ts': time.time(), 'step': step, 'group': group, 'model': cfg['model'], 'ok': False, 'error': 'BadRequest'})
            raise LLMError(f'{step}: bad request: {exc}')
    usage = response.usage
    prompt_details = getattr(usage, 'prompt_tokens_details', None)
    completion_details = getattr(usage, 'completion_tokens_details', None)
    _log({'ts': time.time(), 'step': step, 'group': group, 'model': cfg['model'], 'ok': True,
          'input_tokens': usage.prompt_tokens, 'cached_input_tokens': getattr(prompt_details, 'cached_tokens', 0) or 0,
          'output_tokens': usage.completion_tokens, 'reasoning_tokens': getattr(completion_details, 'reasoning_tokens', 0) or 0,
          'seconds': round(time.time() - started, 2), 'temperature': kwargs.get('temperature'),
          'reasoning_effort': kwargs.get('reasoning_effort'), 'service_tier_requested': cfg.get('service_tier'),
          'service_tier_used': getattr(response, 'service_tier', None), 'notes': notes})
    text = response.choices[0].message.content or ''
    if schema is None:  # tolerate a Markdown fence around plain JSON
        m = re.match(r'^\s*```(?:json)?\s*(.*?)\s*```\s*$', text, re.S)
        text = m.group(1) if m else text
    return text
