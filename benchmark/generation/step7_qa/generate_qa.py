"""Step 7.1-7.3 for one candidate: generate a question, check its form, then check it is answerable from the dialogue.

7.1 the generation model writes one question for a sampled focus (dense subgraph, sparse goal path, or all nodes of
one type) given the full instantiated graph; 7.2 the filter model checks question/answer form; 7.3 the filter model
checks that the dialogue blocks of the cited evidence nodes support the answer. A negative verdict is final.
"""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def validate_generation(value, node_ids):
    if not isinstance(value, dict) or set(value) != {'questions'}:
        raise ValueError('Expected questions object')
    questions = value['questions']
    if not isinstance(questions, list) or len(questions) > 1:
        raise ValueError('Expected at most one question')
    for q in questions:
        if set(q) != {'question', 'answer_mode', 'answer', 'answer_keywords', 'evidence_node_ids'}:
            raise ValueError('Invalid QA fields')
        text = q['question']
        if not isinstance(text, str) or text.count('?') != 1 or not text.rstrip().endswith('?') or '？' in text:
            raise ValueError('Expected one terminal question mark')
        if q['answer_mode'] not in ('named_slots', 'unordered_set'):
            raise ValueError('Unknown answer mode')
        answer, keywords, evidence = q['answer'], q['answer_keywords'], q['evidence_node_ids']
        if not all(isinstance(x, dict) for x in (answer, keywords, evidence)) or not answer:
            raise ValueError('Expected nonempty answer mappings')
        if set(answer) != set(keywords) or set(answer) != set(evidence):
            raise ValueError('Answer slot keys differ')
        for key, v in answer.items():
            if not isinstance(v, str) or not v.strip():
                raise ValueError('Empty/non-string answer')
            k = keywords[key]
            if not isinstance(k, dict) or set(k) != {'canonical', 'aliases'} or k['canonical'] != v:
                raise ValueError('Canonical answer mismatch')
            if not isinstance(k['aliases'], list) or not all(isinstance(a, str) and a.strip() for a in k['aliases']):
                raise ValueError('Invalid aliases')
            ids = evidence[key]
            if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x in node_ids for x in ids):
                raise ValueError('Missing/unknown evidence node')


def validate_review(v):
    if not isinstance(v, dict) or set(v) != {'passed', 'issues'}:
        raise ValueError('Invalid review fields')
    if type(v['passed']) is not bool or not isinstance(v['issues'], list):
        raise ValueError('Invalid review types')
    if not all(isinstance(i, str) and i.strip() for i in v['issues']) or v['passed'] != (not v['issues']):
        raise ValueError('Inconsistent review')


def dialogue_blocks(graph, annotated):
    blocks = []
    for session in annotated['sessions']:
        for block in session['blocks']:
            blocks.append({'session_id': session['session_id'], **block})
    expected = [n['id'] for n in sorted(graph['nodes'], key=lambda n: n['order'])]
    if [b['source_node_id'] for b in blocks] != expected:
        raise ValueError('Dialogue node order/coverage differs from graph')
    for b in blocks:
        if not b['messages'] or not all(m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str) and m['content'].strip() for m in b['messages']):
            raise ValueError('Invalid dialogue block')
    return blocks


def evidence_dialogue(qa, blocks):
    ids = {x for values in qa['evidence_node_ids'].values() for x in values}
    selected = [b for b in blocks if b['source_node_id'] in ids]
    if {b['source_node_id'] for b in selected} != ids:
        raise ValueError('Evidence has no dialogue mapping')
    return selected


def invoke(folder, template, payload, args, validator):
    folder.mkdir(parents=True, exist_ok=True)
    prompt = template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False))
    (folder / 'prompt.txt').write_text(prompt)
    write(folder / 'input.json', payload)
    if (folder / 'result.json').exists():
        value = read(folder / 'result.json'); validator(value); return value
    start = 1 + max([int(p.stem.split('_')[-1]) for p in folder.glob('response_*.json')] or [0])
    errors = []
    for attempt in range(start, start + args.call_attempts):
        raw = (folder / f'response_{attempt}.json').resolve()
        try:
            raw.write_text(llm.call(prompt))
            value = read(raw); validator(value)
            write(folder / 'result.json', value)
            return value
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            errors.append(str(exc)); write(folder / f'error_{attempt}.json', {'error': str(exc)})
    raise RuntimeError('; '.join(errors))


def process(candidate, graph, blocks, templates, args):
    folder = args.output / candidate['candidate_id']
    saved = folder / 'result.json'
    if saved.exists():
        result = read(saved)
        if result['status'] != 'error': return result
    base = {'candidate_id': candidate['candidate_id'], 'method': candidate['method']}
    try:
        gen = invoke(folder / 'generation', templates[candidate['method']],
                     {'full_graph': graph, **candidate['focus']}, args,
                     lambda v: validate_generation(v, {n['id'] for n in graph['nodes']}))
        if not gen['questions']:
            result = {**base, 'status': 'skipped'}
        else:
            qa = gen['questions'][0]
            quality = invoke(folder / 'question_check', templates['question_check'], {'qa': qa}, args, validate_review)
            if not quality['passed']:
                result = {**base, 'status': 'rejected_quality', 'qa': qa, 'quality_review': quality}
            else:
                selected = evidence_dialogue(qa, blocks)
                review = invoke(folder / 'check', templates['dialogue_check'], {'qa': qa, 'dialogue': selected}, args, validate_review)
                result = {**base, 'status': 'accepted' if review['passed'] else 'rejected_dialogue',
                          'qa': qa, 'quality_review': quality, 'review': review}
    except Exception as exc:
        result = {**base, 'status': 'error', 'error': str(exc)}
    write(saved, result)
    print(candidate['candidate_id'], result['status'], flush=True)
    return result
