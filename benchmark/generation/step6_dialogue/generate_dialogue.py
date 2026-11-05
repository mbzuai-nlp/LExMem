#!/usr/bin/env python3
"""Step 6: expand each graph into a user-assistant dialogue, node by node, within the fixed session partition.

6.1 the generation model writes the dialogue fragment of the current node given all previous dialogue, the node and
its relations to earlier nodes; 6.2 the filter model checks the fragment; a rejected fragment is regenerated with the
issues as feedback. 6.3 one check of the complete dialogue. A graph is discarded if a node exhausts its attempts or
the final check fails. Session membership and the source node of each message are assigned by the program.
Outputs per graph: dialogue.json (messages only) and annotated_dialogue.json (messages grouped by source node).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import llm  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def obj(properties):
    return {'type': 'object', 'additionalProperties': False,
            'required': list(properties), 'properties': properties}


STRING = {'type': 'string'}
GEN_SCHEMA = obj({'messages': {'type': 'array', 'items': obj({
    'role': {'type': 'string', 'enum': ['user', 'assistant']}, 'content': STRING})}})
CHECK_SCHEMA = obj({'passed': {'type': 'boolean'},
                    'issues': {'type': 'array', 'items': STRING}})


def validate_review(value):
    if not isinstance(value, dict) or set(value) != {'passed', 'issues'}:
        raise ValueError('Invalid review fields')
    if type(value['passed']) is not bool or not isinstance(value['issues'], list):
        raise ValueError('Invalid review types')
    if any(not isinstance(x, str) or not x.strip() for x in value['issues']):
        raise ValueError('Invalid review issue')
    if value['passed'] != (not value['issues']):
        raise ValueError('Review verdict and issues disagree')


def validate_fragment(value, first_role):
    if not isinstance(value, dict) or set(value) != {'messages'}:
        raise ValueError('Expected messages only')
    if not isinstance(value['messages'], list) or not value['messages']:
        raise ValueError('Empty or invalid messages')
    role = first_role
    total = len(value['messages'])
    for position, message in enumerate(value['messages'], 1):
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError(f'Message {position} of {total} has invalid fields; use exactly role and content')
        if message['role'] != role:
            raise ValueError(
                f"Message {position} of {total} has role '{message['role']}' but role '{role}' is required there. "
                f"Roles must alternate strictly, starting with '{first_role}'; two consecutive messages with the same "
                f"role are not allowed. Resend the whole fragment and remove or merge the offending message")
        if not isinstance(message['content'], str) or not message['content'].strip():
            raise ValueError('Empty message content')
        role = 'assistant' if role == 'user' else 'user'


def node_content(node):
    return {k: node[k] for k in ('id', 'order', 'type', 'content', 'requirements') if k in node}


def make_payload(graph, node, history, session_id, feedback):
    by_id = {n['id']: n for n in graph['nodes']}
    relations = []
    for edge in graph['edges']:
        other = edge['target'] if edge['source'] == node['id'] else edge['source'] if edge['target'] == node['id'] else None
        if other is not None and by_id[other]['order'] < node['order']:
            relations.append({**edge, 'earlier_node': node_content(by_id[other])})
    last = history[-1]['blocks'][-1]['messages'][-1]['role'] if history else None
    return {'previous_dialogue': history, 'current_node': node_content(node),
            'earlier_relations': relations,
            'continuation': {'session_id': session_id, 'next_role': 'assistant' if last == 'user' else 'user'},
            'retry_feedback': feedback}


NEXT_GEN = ('NEXT NODE\nThe instructions above still apply. The previous dialogue is the accepted fragments earlier in this '
            'conversation, in order. Expand this node:\n')
NEXT_CHK = ('NEXT CHECK\nThe instructions above still apply. The earlier accepted candidates in this conversation are the '
            'previous dialogue. Check this candidate:\n')
PASSED = '{"passed": true, "issues": []}'


def gen_text(template, payload, first):
    if first:
        return template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False))
    return NEXT_GEN + json.dumps({k: v for k, v in payload.items() if k != 'previous_dialogue'}, ensure_ascii=False)


def chk_text(template, payload, candidate, first):
    if first:
        return template.replace('{{INPUT_JSON}}', json.dumps({'generation_input': payload, 'candidate': candidate}, ensure_ascii=False))
    body = {'generation_input': {k: v for k, v in payload.items() if k != 'previous_dialogue'}, 'candidate': candidate}
    return NEXT_CHK + json.dumps(body, ensure_ascii=False)


def invoke(folder, template, payload, schema, args, validator, user_text=None, history=None, cache_key=None):
    folder.mkdir(parents=True, exist_ok=True)
    prompt = user_text if user_text is not None else template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False))
    (folder / 'prompt.txt').write_text(prompt)
    write(folder / 'input.json', payload)
    write(folder / 'schema.json', schema)
    raw = (folder / 'response.json').resolve()
    raw.write_text(llm.call(prompt, schema, history=history, cache_key=cache_key) if (history or cache_key) else llm.call(prompt, schema))
    value = json.loads(raw.read_text())
    validator(value)
    write(folder / 'result.json', value)
    return value


def review(folder, template, payload, args, user_text=None, history=None, cache_key=None):
    # invalid or failed reviewer calls are retried; a valid negative verdict is final
    for attempt in range(1, 4):
        af = folder / f'call_{attempt:02d}'
        try:
            if (af / 'result.json').exists():
                result = json.loads((af / 'result.json').read_text())
                validate_review(result)
                return result
            if (af / 'error.json').exists():
                continue
            return invoke(af, template, payload, CHECK_SCHEMA, args, validate_review, user_text=user_text, history=history, cache_key=cache_key)
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            write(af / 'error.json', {'error': str(exc)})
    raise RuntimeError('Reviewer unavailable or invalid after three calls')


def load_source(folder):
    graph = json.loads((folder / 'graph.json').read_text())
    partition = json.loads((folder / 'sessions.json').read_text())
    nodes = sorted(graph['nodes'], key=lambda n: n['order'])
    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids) or len({n['order'] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate node IDs or orders')
    if [i for s in partition['sessions'] for i in s['node_ids']] != ids:
        raise ValueError('Partition does not exactly cover nodes in order')
    if any(not s['node_ids'] for s in partition['sessions']):
        raise ValueError('Empty session')
    if len({s['session_id'] for s in partition['sessions']}) != len(partition['sessions']):
        raise ValueError('Duplicate session IDs')
    if any(e['source'] not in ids or e['target'] not in ids for e in graph['edges']):
        raise ValueError('Unknown edge endpoint')
    return graph, partition, nodes


def export(folder, history):
    visible = {'sessions': [{'session_id': s['session_id'],
                            'messages': [m for b in s['blocks'] for m in b['messages']]} for s in history]}
    write(folder / 'dialogue.json', visible)
    write(folder / 'annotated_dialogue.json', {'sessions': history})
    lines = []
    for s in visible['sessions']:
        lines += [f'## {s["session_id"]}', '']
        for m in s['messages']:
            lines += [f'**{m["role"]}**', '', m['content'], '']
    (folder / 'dialogue.md').write_text('\n'.join(lines))


def run_graph(name, args, templates):
    box = {'af': None, 'nf': None}
    try:
        return _run_graph(name, args, templates, box)
    except llm.PauseRun:
        # paused run: drop the unfinished attempt so the restart does not count it against the node's retry budget
        if box['af'] is not None and not (box['nf'] / 'accepted.json').exists():
            shutil.rmtree(box['af'], ignore_errors=True)
        raise


def _run_graph(name, args, templates, box):
    graph, partition, nodes = load_source(args.input / name)
    folder = args.output / name
    folder.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps([graph, partition, templates, args.model, args.max_node_retries] + ([True] if getattr(args, 'append_mode', False) else []), sort_keys=True).encode()).hexdigest()
    fp = folder / 'fingerprint.txt'
    if fp.exists() and fp.read_text() != signature:
        raise ValueError('Graph, partition, prompts or configuration changed; choose a new output directory')
    fp.write_text(signature)
    status_path = folder / 'status.json'
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old['status'] in {'completed', 'discarded_node_retry', 'discarded_full_check'}:
            return old
    write(folder / 'graph.json', graph)
    write(folder / 'partition.json', partition)
    history = []
    gen_hist, chk_hist = [], []   # append mode: the conversations sent to the generator / checker so far
    append = getattr(args, 'append_mode', False)
    owners = {nid: s['session_id'] for s in partition['sessions'] for nid in s['node_ids']}

    def checkpoint(status, **extra):
        value = {'job': name, 'status': status, 'completed_nodes': sum(len(s['blocks']) for s in history),
                 'total_nodes': len(nodes), **extra}
        write(folder / 'prefix.json', {'sessions': history})
        write(status_path, value)
        return value

    for node in nodes:
        nf = folder / 'nodes' / f'{node["order"]:03d}_{node["id"]}'
        payload = make_payload(graph, node, history, owners[node['id']], [])
        payload_base = dict(payload)   # feedback-free copy; the conversation history never contains reviewer feedback
        checkpoint('running', current_node_id=node['id'])
        if (nf / 'accepted.json').exists():
            candidate = json.loads((nf / 'accepted.json').read_text())
            validate_fragment(candidate, payload['continuation']['next_role'])
        else:
            feedback = []
            candidate = None
            for attempt in range(1, args.max_node_retries + 2):
                af = nf / f'attempt_{attempt:02d}'
                box['af'], box['nf'] = af, nf
                outcome = af / 'outcome.json'
                if outcome.exists():
                    saved = json.loads(outcome.read_text())
                    if saved['passed']:
                        candidate = json.loads((af / 'generation/result.json').read_text())
                        validate_fragment(candidate, payload['continuation']['next_role'])
                        break
                    feedback.extend(saved['issues'])
                    continue
                payload['retry_feedback'] = feedback
                try:
                    if (af / 'generation/result.json').exists():
                        candidate = json.loads((af / 'generation/result.json').read_text())
                        validate_fragment(candidate, payload['continuation']['next_role'])
                    elif (af / 'generation').exists():
                        raise ValueError('Interrupted generation consumes this attempt')
                    else:
                        candidate = invoke(af / 'generation', templates[0], payload, GEN_SCHEMA, args,
                                           lambda v: validate_fragment(v, payload['continuation']['next_role']),
                                           user_text=gen_text(templates[0], payload, not gen_hist) if append else None,
                                           history=gen_hist if append else None, cache_key=f'{name}:gen' if append else None)
                except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                    verdict = {'passed': False, 'issues': [f'Generation/format failure: {exc}']}
                else:
                    verdict = review(af / 'check', templates[1], {'generation_input': payload, 'candidate': candidate}, args,
                                     user_text=chk_text(templates[1], payload, candidate, not chk_hist) if append else None,
                                     history=chk_hist if append else None, cache_key=f'{name}:chk' if append else None)
                write(outcome, verdict)
                if verdict['passed']:
                    break
                feedback.extend(verdict['issues'])
                candidate = None
            else:
                return checkpoint('discarded_node_retry', node_id=node['id'], issues=feedback)
            write(nf / 'accepted.json', candidate)
        if append:   # history keeps the feedback-free user turns and the accepted outputs only
            gen_hist += [{'role': 'user', 'content': gen_text(templates[0], payload_base, not gen_hist)},
                         {'role': 'assistant', 'content': json.dumps(candidate, ensure_ascii=False)}]
            chk_hist += [{'role': 'user', 'content': chk_text(templates[1], payload_base, candidate, not chk_hist)},
                         {'role': 'assistant', 'content': PASSED}]
        # provenance and session membership are assigned here, never by the model
        if not history or history[-1]['session_id'] != owners[node['id']]:
            history.append({'session_id': owners[node['id']], 'blocks': []})
        history[-1]['blocks'].append({'source_node_id': node['id'], 'node_order': node['order'], 'messages': candidate['messages']})
        checkpoint('running')
        print(f'{name}: node {node["order"]} accepted', flush=True)
    checkpoint('checking_full_dialogue')
    verdict = review(folder / 'full_check', templates[2],
                     {'graph': graph, 'partition': {'sessions': [{'session_id': s['session_id'], 'node_ids': s['node_ids']} for s in partition['sessions']]},
                      'annotated_dialogue': {'sessions': history}}, args)
    if not verdict['passed']:
        return checkpoint('discarded_full_check', issues=verdict['issues'])
    export(folder, history)
    return checkpoint('completed', messages=sum(len(b['messages']) for s in history for b in s['blocks']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Output directory of step 5')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', nargs='+', help='Graph directory names; default: completed entries in input summary')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent graphs; nodes within each graph remain sequential')
    parser.add_argument('--append-mode', action=argparse.BooleanOptionalAction, default=True,
                        help='Send 6.1/6.2 of one graph as growing multi-turn conversations (prompt-cache friendly); --no-append-mode sends independent calls')
    parser.add_argument('--max-node-retries', type=int, choices=range(10), default=9)
    args = parser.parse_args()
    args.model = llm.models()
    if args.workers < 1:
        parser.error('--workers must be positive')
    templates = [(HERE / f).read_text() for f in ('node_dialogue_prompt.txt', 'node_dialogue_check_prompt.txt', 'full_dialogue_check_prompt.txt')]
    jobs = args.jobs or [s['job'] for s in json.loads((args.input / 'summary.json').read_text())['jobs'] if s['status'] == 'completed']
    if len(jobs) != len(set(jobs)) or any(Path(n).name != n or n in {'.', '..'} for n in jobs):
        parser.error('Jobs must be unique direct directory names')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'config.json', {'input': str(args.input.resolve()), 'jobs': jobs, 'model': args.model,
                                      'max_node_retries': args.max_node_retries, 'prompts': templates, 'workers': args.workers})
    def execute(name):
        try:
            return run_graph(name, args, templates)
        except Exception as exc:
            status = {'job': name, 'status': 'error', 'error': str(exc)}
            write(args.output / name / 'status.json', status)
            return status

    statuses = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, name) for name in jobs]
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            write(args.output / 'summary.json', {'jobs': sorted(statuses, key=lambda s: s['job'])})
            print(json.dumps(status, ensure_ascii=False), flush=True)
    if any(s['status'] != 'completed' for s in statuses):
        raise SystemExit(1)


if __name__ == '__main__':
    raise SystemExit(llm.run_guarded(main))
