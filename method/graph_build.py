"""Write phase: read a dialogue turn by turn and build its memory graph.

1. Segmentation: every n turns, the LLM marks the turns that start a new unit
   (a shift of discussion focus); each start closes the unit before it.
2. Key generation: a closed unit gets a title, summary and key facts; summary + key facts are embedded.
3. Edge building: candidates are all units on the right frontier plus earlier units whose Key
   similarity is >= similarity_threshold. The LLM picks related candidates, a direction and a
   free-text relation for each, and one parent on the right frontier.
4. Right frontier update: truncate the frontier after the parent and append the new unit.
"""
from pathlib import Path

from common import cosine, invoke_llm, write
from dialogue import build_turns, content_only, load_messages

HERE = Path(__file__).resolve().parent
SEGMENTATION_PROMPT = (HERE / 'prompts' / 'segmentation.txt').read_text()
KEY_PROMPT = (HERE / 'prompts' / 'key_generation.txt').read_text()
EDGE_PROMPT = (HERE / 'prompts' / 'edge_building.txt').read_text()


def turn_payload(turn):
    """Segmentation sees the user message of each turn."""
    return {'turn_number': turn['turn_number'], 'user': content_only(turn['messages'][0])['content']}


def unit_messages(turns):
    return [content_only(m) for t in turns for m in t['messages']]


def validate_segmentation(value, batch_turn_numbers):
    if not isinstance(value, dict) or set(value) != {'new_unit_starts'}:
        raise ValueError('Expected new_unit_starts only')
    starts = value['new_unit_starts']
    if not isinstance(starts, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in starts):
        raise ValueError('new_unit_starts must be a list of integers')
    if len(set(starts)) != len(starts):
        raise ValueError('Duplicate split point in new_unit_starts')
    if not all(t in batch_turn_numbers for t in starts):
        raise ValueError('Segmentation position outside the batch window; '
                         f'new_unit_starts must be a subset of {sorted(batch_turn_numbers)}')


def validate_key(value):
    if not isinstance(value, dict) or set(value) != {'title', 'summary', 'key_facts'}:
        raise ValueError('Expected title, summary, key_facts only')
    if not isinstance(value['title'], str) or not value['title'].strip():
        raise ValueError('Empty title')
    if not isinstance(value['summary'], str) or not value['summary'].strip():
        raise ValueError('Empty summary')
    facts = value['key_facts']
    if not isinstance(facts, list) or not facts or not all(isinstance(x, str) and x.strip() for x in facts):
        raise ValueError('key_facts must be a nonempty list of nonempty strings')


def validate_edges(value, candidate_ids, frontier_ids):
    if not isinstance(value, dict) or set(value) != {'edges', 'parent_id'}:
        raise ValueError('Expected edges and parent_id only')
    edges = value['edges']
    if not isinstance(edges, list) or not edges:
        raise ValueError('edges must be a nonempty list (must include the parent)')
    seen = set()
    for e in edges:
        if not isinstance(e, dict) or set(e) != {'unit_id', 'direction', 'relation'}:
            raise ValueError('Expected unit_id, direction, relation only')
        if e['unit_id'] not in candidate_ids:
            raise ValueError(f"Edge target not in candidate set: {e['unit_id']!r}; "
                             f'valid candidate unit_ids are {sorted(candidate_ids)}')
        if e['unit_id'] in seen:
            raise ValueError(f"Duplicate edge target: {e['unit_id']}")
        seen.add(e['unit_id'])
        if e['direction'] not in ('new_to_candidate', 'candidate_to_new'):
            raise ValueError('Invalid edge direction')
        if not isinstance(e['relation'], str) or not e['relation'].strip():
            raise ValueError('Empty relation description')
    parent_id = value['parent_id']
    if parent_id not in frontier_ids:
        raise ValueError(f'parent_id {parent_id!r} is not on the right frontier; '
                         f'it must be exactly one of {list(frontier_ids)} (the '
                         'on_right_frontier=true candidates), regardless of which '
                         'candidate is most topically similar to the new unit')
    if parent_id not in seen:
        raise ValueError(f'No edge in "edges" targets the chosen parent_id {parent_id!r}; '
                         'add an edge connecting the new unit to its parent')


def key_text(unit):
    return unit['summary'] + '\n' + '\n'.join(unit['key_facts'])


def close_unit(order, turns, *, right_frontier, units_by_id, embedder, similarity_threshold, llm, calls_dir,
               candidate_log):
    """Steps 2-4 for one closed unit; returns (new_edges, new_right_frontier)."""
    unit_id = f'm_{order}'
    key_value = invoke_llm(calls_dir / f'key_{order:04d}', KEY_PROMPT, {'messages': unit_messages(turns)},
                           validator=validate_key, **llm)
    key_vector = embedder.embed_documents([key_text(key_value)])[0]
    unit = {'unit_id': unit_id, 'order': order,
            'message_ids': [m['message_id'] for t in turns for m in t['messages']],
            'title': key_value['title'], 'summary': key_value['summary'], 'key_facts': key_value['key_facts'],
            'parent_id': None, 'key_vector': key_vector}
    units_by_id[unit_id] = unit
    if order == 1:
        candidate_log.append({'unit_id': unit_id, 'right_frontier_candidates': [],
                              'similarity_candidates': [], 'connected': [], 'parent_id': None})
        return [], [unit]

    frontier_ids = [u['unit_id'] for u in right_frontier]
    older = [u for u in units_by_id.values() if u['unit_id'] not in frontier_ids and u['unit_id'] != unit_id]
    older.sort(key=lambda u: -cosine(key_vector, u['key_vector']))
    similarity_candidates = [u for u in older if cosine(key_vector, u['key_vector']) >= similarity_threshold]
    candidates = (
        [{'unit_id': u['unit_id'], 'title': u['title'], 'summary': u['summary'], 'on_right_frontier': True}
         for u in right_frontier]
        + [{'unit_id': u['unit_id'], 'title': u['title'], 'summary': u['summary'], 'on_right_frontier': False}
           for u in similarity_candidates])
    candidate_ids = {c['unit_id'] for c in candidates}
    edge_payload = {'new_unit': {'title': key_value['title'], 'summary': key_value['summary'],
                                 'key_facts': key_value['key_facts']},
                    'candidates': candidates}
    edges_value = invoke_llm(calls_dir / f'edges_{order:04d}', EDGE_PROMPT, edge_payload,
                             validator=lambda v: validate_edges(v, candidate_ids, set(frontier_ids)), **llm)

    new_edges = []
    for e in edges_value['edges']:
        source, target = (unit_id, e['unit_id']) if e['direction'] == 'new_to_candidate' else (e['unit_id'], unit_id)
        new_edges.append({'source': source, 'target': target, 'relation': e['relation']})
    parent_id = edges_value['parent_id']
    unit['parent_id'] = parent_id
    candidate_log.append({'unit_id': unit_id, 'right_frontier_candidates': frontier_ids,
                          'similarity_candidates': [{'unit_id': u['unit_id'],
                                                     'score': cosine(key_vector, u['key_vector'])}
                                                    for u in similarity_candidates],
                          'connected': [e['unit_id'] for e in edges_value['edges']], 'parent_id': parent_id})
    return new_edges, right_frontier[:frontier_ids.index(parent_id) + 1] + [unit]


def build_graph(dialogue_id, dialogues_dir, embedder, output_dir, *, n, similarity_threshold, llm):
    """llm: keyword settings for invoke_llm (model, base_url, api_key, timeout, attempts, ...)."""
    turns = build_turns(load_messages(dialogue_id, dialogues_dir))
    calls_dir = output_dir / 'calls'
    units_by_id, all_edges, candidate_log = {}, [], []
    right_frontier, open_turns, order = [], [], 0

    def close(turns_to_close):
        nonlocal order, right_frontier
        order += 1
        new_edges, right_frontier = close_unit(
            order, turns_to_close, right_frontier=right_frontier, units_by_id=units_by_id, embedder=embedder,
            similarity_threshold=similarity_threshold, llm=llm, calls_dir=calls_dir, candidate_log=candidate_log)
        all_edges.extend(new_edges)

    for batch_index, start in enumerate(range(0, len(turns), n), 1):
        batch = turns[start:start + n]
        batch_numbers = {t['turn_number'] for t in batch}
        payload = {'open_unit_turns': [turn_payload(t) for t in open_turns] or None,
                   'new_turns': [turn_payload(t) for t in batch]}
        seg_value = invoke_llm(calls_dir / f'segment_{batch_index:04d}', SEGMENTATION_PROMPT, payload,
                               validator=lambda v: validate_segmentation(v, batch_numbers), **llm)
        starts = set(seg_value['new_unit_starts'])
        segments = [[]]
        for t in batch:
            if t['turn_number'] in starts:
                segments.append([])
            segments[-1].append(t)
        open_turns = open_turns + segments[0]
        for seg in segments[1:]:
            if open_turns:
                close(open_turns)
            open_turns = list(seg)
    if open_turns:
        close(open_turns)

    units = sorted(units_by_id.values(), key=lambda u: u['order'])
    graph = {'dialogue_id': dialogue_id,
             'units': [{k: v for k, v in u.items() if k != 'key_vector'} for u in units],
             'edges': all_edges}
    write(output_dir / 'graph.json', graph)
    write(output_dir / 'candidate_log.json', candidate_log)
    return graph
