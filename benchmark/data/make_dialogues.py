#!/usr/bin/env python3
"""Write dialogues/<id>/dialogue.json (plain messages, the only input for evaluated systems)
from dialogues/<id>/annotated_dialogue.json by dropping the source-node blocks."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

for folder in sorted((HERE / 'dialogues').iterdir()):
    annotated = json.loads((folder / 'annotated_dialogue.json').read_text())
    plain = {'sessions': [{'session_id': s['session_id'],
                           'messages': [{'role': m['role'], 'content': m['content']}
                                        for block in s['blocks'] for m in block['messages']]}
                          for s in annotated['sessions']]}
    (folder / 'dialogue.json').write_text(json.dumps(plain, ensure_ascii=False, indent=2) + '\n')
print('done')
