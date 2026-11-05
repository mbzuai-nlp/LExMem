#!/usr/bin/env python3
"""Step 2: evolve one skeleton graph with probabilistic, rule-triggered continuations.

The controller scans the graph in node order. When a newly visible edge matches an evolution rule (support, refute,
hypothesis/artifact revision, revival) and the rule fires with its configured probability, the graph is cut at that
node and the model regenerates the future from the immutable prefix so that the requested development appears
(e.g. later evidence refuting a supported hypothesis). Each continuation must pass structural checks and a model review.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP1 = HERE.parent / 'step1_initial_graphs'
sys.path.insert(0, str(STEP1))
from graph_io import validate
from generate_graphs import output_schema, build_input, build_prompt
from check_graphs import review_schema, validate_review

TEMPLATES = {'S': ('S1',), 'F': ('F1',), 'RH': ('RH1', 'RH2'),
             'RA': ('RA1',), 'V': ('V1', 'V2')}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def ek(edge):
    return edge['source'], edge['relation'], edge['target']


def prefix(graph, cutoff):
    nodes = copy.deepcopy(graph['nodes'][:cutoff])
    ids = {n['id'] for n in nodes}
    return {'nodes': nodes, 'edges': [copy.deepcopy(e) for e in graph['edges']
                                    if e['source'] in ids and e['target'] in ids]}


def load_prompts(text):
    shared = text.split('SHARED PROMPT\n=============\n', 1)[1].split('\nRULE S:', 1)[0].strip()
    templates = dict(re.findall(
        r'^(S1|F1|RH1|RH2|RA1|V1|V2) [^\n]+\n(?:Eligibility:[^\n]+\n)?Prompt:\n(.*?)\nCheck:',
        text, re.M | re.S))
    expected = {t for ts in TEMPLATES.values() for t in ts}
    if set(templates) != expected:
        raise ValueError('Prompt TXT must contain exactly the seven supported templates')
    return shared, {k: v.strip() for k, v in templates.items()}


def fill(text, values):
    missing = set(re.findall(r'\{\{(\w+)\}\}', text)) - values.keys()
    if missing:
        raise ValueError(f'Unbound prompt placeholders: {sorted(missing)}')
    return re.sub(r'\{\{(\w+)\}\}', lambda m: str(values[m[1]]), text)


def validate_config(config):
    if set(config) != set(TEMPLATES):
        raise ValueError('Configuration must contain S, F, RH, RA, V')
    for rule, spec in config.items():
        if set(spec) != {'probability', 'templates'}:
            raise ValueError(f'Unexpected configuration fields: {rule}')
        p = spec['probability']
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Invalid trigger probability: {rule}')
        weights = spec['templates']
        if not isinstance(weights, dict) or set(weights) != set(TEMPLATES[rule]):
            raise ValueError(f'Invalid template names: {rule}')
        if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError(f'Invalid template weights: {rule}')
    return config


def matches_at(graph, cutoff):
    """Only edges first visible at cutoff; all eligibility checks use this prefix."""
    history = prefix(graph, cutoff)
    ns = {n['id']: n for n in history['nodes']}
    edges = history['edges']
    def has(relation, target):
        return any(e['relation'] == relation and e['target'] == target for e in edges)
    matches = []
    for e in sorted(edges, key=ek):
        s, r, t = ek(e)
        if max(ns[s]['order'], ns[t]['order']) != cutoff:
            continue
        st, tt = ns[s]['type'], ns[t]['type']
        rule, bindings, eligible = None, {}, []
        if r == 'support' and st == 'evidence' and tt in ('hypothesis', 'proposal') and not has('refute', t):
            rule, bindings = 'S', {'SUPPORT_EVIDENCE_ID': s, 'TARGET_ID': t}
            eligible = ['S1']
        elif r == 'refute' and st == 'evidence' and tt in ('hypothesis', 'proposal') and not has('support', t):
            rule, bindings = 'F', {'REFUTING_EVIDENCE_ID': s, 'TARGET_ID': t}
            eligible = ['F1']
        elif r == 'revise' and st == tt == 'hypothesis':
            rule = 'RH'
            bindings = {'REVISED_HYPOTHESIS_ID': s, 'EARLIER_HYPOTHESIS_ID': t}
            eligible = ['RH1'] + ([] if has('refute', s) else ['RH2'])
        elif r == 'revise' and st == tt == 'artifact':
            rule = 'RA'
            bindings = {'REVISED_ARTIFACT_ID': s, 'EARLIER_ARTIFACT_ID': t}
            eligible = ['RA1']
        elif r == 'revive' and st in ('evidence', 'decision') and tt in ('hypothesis', 'proposal'):
            later_result = any(x['target'] == t and x['relation'] in ('support', 'refute')
                               and ns[x['source']]['order'] > ns[s]['order'] for x in edges)
            if not later_result:
                rule, bindings, eligible = 'V', {'REVIVAL_SOURCE_ID': s, 'TARGET_ID': t}, ['V1', 'V2']
        if rule:
            matches.append({'rule': rule, 'edge': e, 'cutoff': cutoff,
                            'bindings': bindings, 'eligible_templates': eligible})
    return matches


def choose(matches, config, rng):
    """Randomize simultaneous matches to avoid giving JSON edge order priority."""
    candidates = copy.deepcopy(matches)
    rng.shuffle(candidates)
    trace = []
    for match in candidates:
        spec = config[match['rule']]
        choices = [t for t in match['eligible_templates'] if spec['templates'][t] > 0]
        draw = rng.random()
        entry = {'match': match, 'probability': spec['probability'], 'draw': draw,
                 'triggered': bool(choices) and draw < spec['probability']}
        trace.append(entry)
        if entry['triggered']:
            match['template'] = rng.choices(choices, [spec['templates'][t] for t in choices])[0]
            entry['selected_template'] = match['template']
            return match, trace
    return None, trace


def outcome(graph, request):
    """Minimum outcome only: never require an activity/evidence/decision chain."""
    ns = {n['id']: n for n in graph['nodes']}
    after = request['cutoff']
    def witnesses(relation, target, source_type):
        return [e for e in graph['edges'] if e['relation'] == relation and e['target'] == target
                and ns[e['source']]['order'] > after and ns[e['source']]['type'] == source_type]
    t, b = request['template'], request['bindings']
    if t in ('S1', 'F1', 'V1', 'V2'):
        relation = 'refute' if t in ('S1', 'V2') else 'support'
        requirements = [(relation, b['TARGET_ID'], 'evidence')]
    elif t == 'RH1':
        requirements = [('revise', b['REVISED_HYPOTHESIS_ID'], 'hypothesis')]
    elif t == 'RH2':
        requirements = [('support', b['EARLIER_HYPOTHESIS_ID'], 'evidence'),
                        ('refute', b['REVISED_HYPOTHESIS_ID'], 'evidence')]
    elif t == 'RA1':
        requirements = [('revise', b['REVISED_ARTIFACT_ID'], 'artifact')]
    else:
        raise ValueError(f'Unknown template: {t}')
    found = [witnesses(*requirement) for requirement in requirements]
    return {'passed': all(found), 'requirements': requirements, 'witnesses': found}


def validate_continuation(graph, history, seeds, obligations):
    stats = validate(graph, seed_input=seeds)
    k = len(history['nodes'])
    if len(graph['nodes']) <= k:
        raise ValueError('Continuation must contain new nodes after the cutoff')
    actual = prefix(graph, k)
    if actual['nodes'] != history['nodes'] or {ek(e) for e in actual['edges']} != {ek(e) for e in history['edges']}:
        raise ValueError('Immutable prefix changed (nodes, parents, order, or old-old edges)')
    ns = {n['id']: n for n in graph['nodes']}
    for e in graph['edges']:
        s, r, t = ek(e)
        so, to = ns[s]['order'], ns[t]['order']
        if max(so, to) <= k:
            continue
        if r == 'produce' and so >= to:
            raise ValueError(f'Activity must precede its new output: {e}')
        if r == 'evaluate' and so <= to:
            raise ValueError(f'Evaluation must follow its target: {e}')
    checks = [outcome(graph, obligation) for obligation in obligations]
    if not all(c['passed'] for c in checks):
        missing = [o['template'] for o, c in zip(obligations, checks) if not c['passed']]
        raise ValueError(f'Missing requested evolution outcome(s): {missing}')
    return {'stats': stats, 'outcomes': checks}


def request_prompt(shared, templates, history, seeds, selected, previous):
    values = {'SEED_INPUT_JSON': json.dumps(seeds, ensure_ascii=False),
              'PREFIX_JSON': json.dumps(history, ensure_ascii=False),
              'CUTOFF_ORDER': len(history['nodes'])}
    text = fill(shared, values) + '\n\nSELECTED TASK\n' + fill(templates[selected['template']], selected['bindings'])
    # A later cutoff can precede an earlier round's requested result. Regenerate
    # that still-future result rather than silently losing the earlier task.
    pending = [o for o in previous if not outcome(history, o)['passed']]
    if pending:
        text += '\n\nEARLIER REQUESTED DEVELOPMENTS NOT YET PRESENT IN THIS PREFIX\n'
        text += 'Also retain these outcomes in the regenerated future; their former suffix was discarded.\n'
        text += '\n'.join(fill(templates[o['template']], o['bindings']) for o in pending)
    return text


class Codex:
    def __init__(self, model=None, timeout=300):
        self.model, self.timeout = model, timeout

    def __call__(self, prompt, schema, folder):
        folder = Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        prompt += '\nReturn JSON only. Do not use tools, read files, or run commands.\n'
        (folder / 'prompt.txt').write_text(prompt)
        save(folder / 'schema.json', schema)
        response = folder / 'response.json'
        with tempfile.TemporaryDirectory(prefix='graph-evolution-') as work:
            command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check',
                       '--sandbox', 'read-only', '-c', 'approval_policy="never"',
                       '--cd', work, '--color', 'never', '--output-schema', str(folder / 'schema.json'),
                       '--output-last-message', str(response)]
            if self.model:
                command += ['--model', self.model]
            command += ['-']
            save(folder / 'command.json', command)
            with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
                result = subprocess.run(command, input=prompt, text=True, stdout=out, stderr=err,
                                        timeout=self.timeout)
            if result.returncode:
                raise RuntimeError(f'Codex exited {result.returncode}; see {folder / "stderr.log"}')
        return read(response)


def review_graph(client, graph, prompt, folder, retries, task=''):
    request = prompt.replace('{{GRAPH_JSON}}', json.dumps(graph, ensure_ascii=False))
    request += '\nReview the supplied JSON; no image is attached.\n'
    if task:
        request += '\nCONTINUATION TASK CONTEXT\n' + task + '\nAssess whether the development is plausibly integrated. Do not demand a fixed node chain.\n'
    error = ''
    for attempt in range(1, retries + 2):
        dest = folder / f'review_{attempt:02d}'
        try:
            result = client(request + ('\nPrevious review format error: ' + error if error else ''), review_schema(), dest)
            save(dest / 'parsed.json', result)
            validate_review(result, graph)
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            save(dest / 'validation.json', {'passed': False, 'error': error})
            continue
        save(dest / 'validation.json', {'passed': True})
        # Never retry a well-formed negative review just to obtain approval.
        if result['verdict'] != 'reasonable':
            raise ValueError('Model review rejected candidate: ' + json.dumps(result, ensure_ascii=False))
        return result
    raise ValueError(f'Invalid review after retries: {error}')


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, list) else value


def run(state, out, client, shared, templates, check_prompt, retries):
    rng = random.Random()
    rng.setstate(tuple_tree(state['rng_state']))
    def checkpoint():
        state['rng_state'] = rng.getstate()
        save(out / 'checkpoint.json', state)
    state['status'] = 'running'
    state.pop('error', None)
    checkpoint()
    try:
        while state['cursor'] <= len(state['graph']['nodes']):
            k = state['cursor']
            if not state.get('pending'):
                selected, trace = choose(matches_at(state['graph'], k), state['config'], rng)
                with (out / 'scan.jsonl').open('a') as log:
                    log.write(json.dumps({'round': state['round'], 'order': k, 'matches': trace}) + '\n')
                if not selected:
                    state['cursor'] += 1
                    checkpoint()
                    continue
                state['pending'] = selected
                checkpoint()
            selected = state['pending']
            k = selected['cutoff']
            history = prefix(state['graph'], k)
            round_dir = out / 'rounds' / f'{state["round"] + 1:04d}'
            round_dir.mkdir(parents=True, exist_ok=True)
            save(round_dir / 'prefix.json', history)
            save(round_dir / 'selection.json', selected)
            obligations = state['obligations'] + [selected]
            prompt = request_prompt(shared, templates, history, state['seeds'], selected, state['obligations'])
            (round_dir / 'prompt.txt').write_text(prompt)
            error = ''
            for attempt in range(1, retries + 2):
                # keep earlier attempts when resuming
                attempt_id = 1
                while (round_dir / f'attempt_{attempt_id:03d}').exists():
                    attempt_id += 1
                dest = round_dir / f'attempt_{attempt_id:03d}'
                dest.mkdir()
                print(f'Round {state["round"]+1}: cutoff={k}, template={selected["template"]}, attempt={attempt_id}', flush=True)
                try:
                    graph = client(prompt + ('\nPrevious candidate failed: ' + error if error else ''), output_schema(), dest / 'generation')
                    save(dest / 'candidate.json', graph)
                    checks = validate_continuation(graph, history, state['seeds'], obligations)
                    save(dest / 'structural_check.json', checks)
                    task = fill(templates[selected['template']], selected['bindings'])
                    review_graph(client, graph, check_prompt, dest, retries, task)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                    save(dest / 'rejection.json', {'error': error})
                    continue
                save(round_dir / 'accepted.json', graph)
                state.update(graph=graph, cursor=k+1, round=state['round']+1,
                             pending=None, obligations=obligations)
                checkpoint()
                print(f'Accepted {len(graph["nodes"])} nodes; resume at #{k+1}', flush=True)
                break
            else:
                raise RuntimeError(f'Candidate retries exhausted: {error}')
        # the last accepted graph becomes final once the scan reaches the end of the graph
        save(out / 'final.json', state['graph'])
        save(out / 'final.input.json', state['seeds'])
        from visualize_graphs import render
        render(state['graph'], 'final', out, tight=True)
        state['status'] = 'complete'
        checkpoint()
        save(out / 'summary.json', {'status': 'complete', 'rounds': state['round'],
                                   'nodes': len(state['graph']['nodes']),
                                   'outcomes': [outcome(state['graph'], o) for o in state['obligations']]})
        return state['graph']
    except BaseException as exc:
        state['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        state['error'] = str(exc)
        checkpoint()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, help='Existing full graph; otherwise generate from seeds')
    parser.add_argument('--proposal-count', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=HERE / 'evolution_rule_prompts.txt')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--trigger-probability', type=float, help='Override probability for all rules')
    parser.add_argument('--seed', type=int, default=42, help='Controller RNG seed, not model sampling seed')
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per model call, not graph stopping rule')
    parser.add_argument('--retries', type=int, default=1, help='Retries for failed generation/review format')
    parser.add_argument('--dry-run', action='store_true', help='List matches and bound prompts for --initial; no model calls')
    parser.add_argument('--resume', action='store_true', help='Resume saved run; use saved prompts/config/model/settings')
    args = parser.parse_args()
    if args.timeout < 1 or args.retries < 0:
        parser.error('timeout must be positive; retries nonnegative')
    out = args.output.resolve()
    if args.resume:
        if args.dry_run or args.initial:
            parser.error('--resume cannot be combined with --dry-run or --initial')
        state = read(out / 'checkpoint.json')
        if state['status'] == 'complete':
            print(f'Already complete: {out / "final.json"}')
            return
        text = (out / 'prompts.txt').read_text()
        if hashlib.sha256(text.encode()).hexdigest() != state['prompt_sha256']:
            parser.error('Saved prompt snapshot changed')
        shared, templates = load_prompts(text)
        settings = state['settings']
        run(state, out, Codex(settings['model'], settings['timeout']), shared, templates,
            (out / 'check_prompt.txt').read_text(), settings['retries'])
        return
    if out.exists() and any(out.iterdir()):
        parser.error(f'Refusing to overwrite nonempty output: {out}')
    if args.dry_run and not args.initial:
        parser.error('--dry-run requires --initial')
    text = args.prompts.read_text()
    shared, templates = load_prompts(text)
    config = read(args.config)
    if args.trigger_probability is not None:
        for spec in config.values():
            spec['probability'] = args.trigger_probability
    validate_config(config)
    check_prompt = (STEP1 / 'graph_check_prompt.txt').read_text()
    client = Codex(args.model, args.timeout)
    if not args.dry_run and not shutil.which('codex'):
        parser.error('Codex CLI must be installed and logged in')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prompts.txt').write_text(text)
    (out / 'check_prompt.txt').write_text(check_prompt)
    save(out / 'config.json', config)
    if args.initial:
        graph = read(args.initial)
        seeds = {'goal': {'id': 'g1', 'type': 'goal'},
                 'proposals': [{'id': n['id'], 'type': 'proposal'} for n in graph['nodes'] if n['type'] == 'proposal']}
        validate(graph, seed_input=seeds)
    else:
        seeds = build_input(args.proposal_count)
        initial_prompt = build_prompt((STEP1 / 'graph_generation_prompt.txt').read_text(), seeds)
        error = ''
        for attempt in range(1, args.retries + 2):
            dest = out / 'initial_generation' / f'attempt_{attempt:02d}'
            try:
                graph = client(initial_prompt + ('\nPrevious error: ' + error if error else ''), output_schema(), dest)
                validate(graph, seed_input=seeds)
                review_graph(client, graph, check_prompt, dest, args.retries)
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                save(dest / 'rejection.json', {'error': error})
        else:
            raise RuntimeError(f'Initial generation failed: {error}')
    save(out / 'initial.json', graph)
    save(out / 'initial.input.json', seeds)
    if args.dry_run:
        candidates = [m for k in range(1, len(graph['nodes'])+1) for m in matches_at(graph, k)]
        save(out / 'matches.json', candidates)
        for i, match in enumerate(candidates, 1):
            for template in match['eligible_templates']:
                task = dict(match, template=template)
                request = request_prompt(shared, templates, prefix(graph, match['cutoff']), seeds, task, [])
                path = out / 'requests' / f'{i:03d}_{template}.txt'
                path.parent.mkdir(exist_ok=True)
                path.write_text(request)
        print(f'Dry run: {len(candidates)} candidate matches. No probabilities sampled or model calls made. See {out}')
        return
    # an existing input graph is reviewed once more; a negative verdict stops its evolution
    if args.initial:
        review_graph(client, graph, check_prompt, out / 'initial_review', args.retries)
    state = {'status': 'ready', 'graph': graph, 'seeds': seeds, 'config': config,
             'cursor': 1, 'round': 0, 'pending': None, 'obligations': [],
             'rng_state': random.Random(args.seed).getstate(), 'controller_seed': args.seed,
             'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
             'settings': {'model': args.model, 'timeout': args.timeout, 'retries': args.retries}}
    run(state, out, client, shared, templates, check_prompt, args.retries)
    print(f'Final graph: {out / "final.json"}')


if __name__ == '__main__':
    main()
