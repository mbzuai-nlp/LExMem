"""Read phase: assemble the retrieved context and ask the reading LLM."""
from pathlib import Path

from common import count_tokens, invoke_llm, write
from dialogue import visible
from retrieve import score_and_select

HERE = Path(__file__).resolve().parent
PROMPTS = {False: (HERE / 'prompts' / 'answer.txt').read_text(),
           True: (HERE / 'prompts' / 'answer_with_full_dialogue.txt').read_text()}


def validate_answer(value):
    if not isinstance(value, dict) or set(value) != {'answer_items'}:
        raise ValueError('Expected answer_items only')
    items = value['answer_items']
    if not isinstance(items, list) or not all(isinstance(x, str) and x.strip() for x in items):
        raise ValueError('answer_items must be a list of nonempty strings')


def assemble_context(anchor_units, edges, messages_by_id):
    """Anchor units in dialogue order, each with its Key and original messages; an edge between two
    anchors is listed under the later one."""
    ordered = sorted(anchor_units, key=lambda u: u['order'])
    order_of = {u['unit_id']: u['order'] for u in ordered}
    edges_by_later_unit = {u['unit_id']: [] for u in ordered}
    for e in edges:
        if e['source'] in order_of and e['target'] in order_of:
            later, earlier = ((e['target'], e['source']) if order_of[e['target']] > order_of[e['source']]
                              else (e['source'], e['target']))
            edges_by_later_unit[later].append({'other_unit_id': earlier, 'source_unit_id': e['source'],
                                               'target_unit_id': e['target'], 'relation': e['relation']})
    return [{'unit_id': u['unit_id'], 'title': u['title'], 'summary': u['summary'], 'key_facts': u['key_facts'],
             'messages': '\n'.join(f"{messages_by_id[mid]['role']}: {messages_by_id[mid]['content']}"
                                   for mid in u['message_ids']),
             'edges_to_earlier_selected_units': edges_by_later_unit[u['unit_id']]}
            for u in ordered]


def related_context_payload(neighbor_units, edges, anchor_ids):
    """Neighbor units: title, summary and the relation connecting them to an anchor."""
    relation_of = {}
    for e in edges:
        relation_of[(e['source'], e['target'])] = e['relation']
        relation_of[(e['target'], e['source'])] = e['relation']
    payload = []
    for u in sorted(neighbor_units, key=lambda u: u['order']):
        connected_to, relation = None, ''
        for aid in anchor_ids:
            if (u['unit_id'], aid) in relation_of:
                connected_to, relation = aid, relation_of[(u['unit_id'], aid)]
                break
        payload.append({'unit_id': u['unit_id'], 'title': u['title'], 'summary': u['summary'],
                        'connected_to': connected_to, 'relation': relation})
    return payload


def answer_question(qa, graph, messages, embedder, output_folder, *, token_budget, neighbor_token_budget,
                    neighbor_hops, rrf_k, with_full_dialogue, llm):
    """Only question_id, dialogue_id and question are read from `qa`."""
    messages_by_id = {m['message_id']: m for m in messages}
    units, edges = graph['units'], graph['edges']
    anchors, neighbors = score_and_select(
        units, edges, embedder, qa['question'], token_budget=token_budget,
        neighbor_token_budget=neighbor_token_budget, messages_by_id=messages_by_id,
        neighbor_hops=neighbor_hops, rrf_k=rrf_k)
    anchor_ids = {u['unit_id'] for u in anchors}
    memory = {'conversation_units': assemble_context(anchors, edges, messages_by_id),
              'related_context': related_context_payload(neighbors, edges, anchor_ids)}
    if with_full_dialogue:
        payload = {'conversation': [visible(m) for m in messages], 'retrieved_memory': memory,
                   'question': qa['question']}
    else:
        payload = {**memory, 'question': qa['question']}
    prediction = invoke_llm(output_folder / 'answer_call', PROMPTS[with_full_dialogue], payload,
                            validator=validate_answer, **llm)
    result = {'question_id': qa['question_id'], 'dialogue_id': qa['dialogue_id'], 'question': qa['question'],
              'prediction': prediction,
              'selected_unit_ids': sorted(anchor_ids),
              'neighbor_unit_ids': sorted(u['unit_id'] for u in neighbors),
              'anchor_text_tokens': sum(count_tokens(messages_by_id[mid]['content'])
                                        for u in anchors for mid in u['message_ids'])}
    write(output_folder / 'result.json', result)
    return result
