"""Load a dialogue and split it into turns.

Only dialogues/<dialogue_id>/dialogue.json (plain messages) is read; the annotated dialogue,
the session partition and the generation graph are never used by the method.
"""
from common import read


def load_messages(dialogue_id, dialogues_dir):
    """Messages with message_id = {dialogue_id}:{session_id}:M{0001,...} (numbered per session)."""
    data = read(dialogues_dir / dialogue_id / 'dialogue.json')
    messages = []
    for session in data['sessions']:
        sid = session['session_id']
        for i, m in enumerate(session['messages'], 1):
            if m['role'] not in ('user', 'assistant') or not isinstance(m.get('content'), str) or not m['content'].strip():
                raise ValueError(f'Invalid message in {dialogue_id}:{sid}')
            messages.append({'message_id': f'{dialogue_id}:{sid}:M{i:04d}', 'role': m['role'], 'content': m['content']})
    if not messages:
        raise ValueError(f'Empty dialogue: {dialogue_id}')
    return messages


def build_turns(messages):
    """A turn is a user message plus the assistant reply that follows it.

    Session boundaries are ignored: the dialogue is one continuous sequence of turns.
    A trailing user message without a reply forms the last turn by itself.
    """
    turns = []
    i, n = 0, len(messages)
    while i < n:
        if messages[i]['role'] != 'user':
            raise ValueError(f'Expected a user message to start turn {len(turns) + 1} at index {i}')
        if i + 1 < n and messages[i + 1]['role'] == 'assistant':
            turns.append({'turn_number': len(turns) + 1, 'messages': [messages[i], messages[i + 1]]})
            i += 2
        elif i == n - 1:
            turns.append({'turn_number': len(turns) + 1, 'messages': [messages[i]]})
            i += 1
        else:
            raise ValueError(f'Unpaired user message before dialogue end at index {i}')
    return turns


def visible(message):
    return {'message_id': message['message_id'], 'role': message['role'], 'content': message['content']}


def content_only(message):
    """role/content without message_id, so session numbering cannot influence graph construction."""
    return {'role': message['role'], 'content': message['content']}
