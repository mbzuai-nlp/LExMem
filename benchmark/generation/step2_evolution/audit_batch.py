#!/usr/bin/env python3
"""Audit an evolution batch: input provenance, immutable prefixes, requested outcomes, screening and export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evolve_graph import read, save, prefix, validate_continuation, outcome, validate


def audit(root):
    batch=read(root/'batch.json');results=[]
    lines=['# Evolution batch', '',
           f'Model: {batch["model"]}; trigger probability: {batch["probability"]}; graphs: {batch["count"]}.', '',
           '| Graph | proposals | status | initial nodes | final nodes | rounds |',
           '|---|---:|---|---:|---:|---:|']
    for record in batch['graphs']:
        result={'id':record['id'],'status':record['status'],'verified':False}
        if record.get('initial_source'):
            source=root/record['initial_source']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==record['initial_sha256']
        if record.get('evolution_dir'):
            evo=root/record['evolution_dir']
            if (evo/'initial.json').exists():
                previous=read(evo/'initial.json'); seeds=read(evo/'initial.input.json')
                validate(previous,seed_input=seeds);obligations=[];cutoffs=[]
                for rd in sorted((evo/'rounds').glob('*')):
                    if not (rd/'selection.json').exists():continue
                    selection=read(rd/'selection.json');history=read(rd/'prefix.json')
                    assert history==prefix(previous,selection['cutoff']),str(rd)
                    cutoffs.append(selection['cutoff'])
                    if (rd/'accepted.json').exists():
                        current=read(rd/'accepted.json');obligations.append(selection)
                        validate_continuation(current,history,seeds,obligations);previous=current
                assert cutoffs==sorted(set(cutoffs))
                if (evo/'final.json').exists():
                    assert read(evo/'final.json')==previous
                    state=read(evo/'checkpoint.json')
                    assert state['status']=='complete' and state['cursor']==len(previous['nodes'])+1
                    result.update(verified=True,rounds=len(obligations),cutoffs=cutoffs,
                                  templates=dict(Counter(o['template'] for o in obligations)),
                                  nodes=len(previous['nodes']),outcomes=[outcome(previous,o) for o in obligations])
        if record['status'] in ('accepted','rejected') and record.get('failed_stage') != 'initial_review':
            screen=root/record['screening_dir'];check=read(screen/f'{record["id"]}.check.json')
            assert check['status']=='completed'
            keep=check['local_check']['passed'] and check['review']=={'verdict':'reasonable','issues':[]}
            assert keep==(record['status']=='accepted')
            result['review']=check['review']
            if keep:
                assert result['verified']
                assert (root/'accepted'/f'{record["id"]}.json').read_bytes()==(root/record['evolution_dir']/'final.json').read_bytes()
        results.append(result)
        lines.append(f'| {record["id"]} | {record["proposals"]} | {record["status"]} | {record.get("initial_nodes","—")} | {record.get("final_nodes","—")} | {record.get("rounds","—")} |')
    accepted={r['id'] for r in batch['graphs'] if r['status']=='accepted'}
    actual={p.stem for p in (root/'accepted').glob('graph_*.json') if not p.name.endswith('.input.json')}
    assert accepted==actual, 'Accepted directory contains missing or unapproved graphs'
    counts=dict(Counter(r['status'] for r in batch['graphs']))
    save(root/'verification.json',{'counts':counts,'graphs':results})
    lines += ['', '## Counts', '', str(counts), '',
              f'Evolution rounds: {sum(r.get("rounds",0) for r in results)}.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    print(audit(p.parse_args().output))
