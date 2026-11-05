#!/usr/bin/env python3
"""Step 5: partition each instantiated graph into consecutive conversational sessions (node order is kept).

The program checks that the sessions cover every node exactly once, in order; invalid partitions are retried
with the validation error as feedback (at most 3 attempts).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import llm  # noqa: E402

STYLE = 'body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#243044}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}th{background:#eef2f7}section{border-top:5px solid #5382c5;margin:28px 0;padding-top:8px}.boundary{background:#fff4da;padding:12px}a{color:#165abc}'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def validate(result, graph):
    if not isinstance(result, dict) or set(result) != {'sessions'} or not isinstance(result['sessions'], list) or not result['sessions']:
        raise ValueError('Expected a nonempty sessions array')
    flat = []
    for i, session in enumerate(result['sessions'], 1):
        if set(session) != {'session_id', 'node_ids', 'topic', 'boundary_reason'}:
            raise ValueError('Unexpected session fields')
        if session['session_id'] != f'S{i:02d}':
            raise ValueError('Session IDs must be consecutive from S01')
        if not isinstance(session['node_ids'], list) or not session['node_ids'] or not all(isinstance(x, str) for x in session['node_ids']):
            raise ValueError('Each session needs a nonempty list of node IDs')
        if any(not isinstance(session[k], str) or not session[k].strip() for k in ('topic', 'boundary_reason')):
            raise ValueError('Missing topic or boundary reason')
        flat.extend(session['node_ids'])
    if flat != [n['id'] for n in sorted(graph['nodes'], key=lambda n:n['order'])]:
        raise ValueError('Sessions must cover every node exactly once in order, without gaps or rearrangement')

def schema():
    string = {'type':'string'}
    fields = {'session_id':string,'node_ids':{'type':'array','items':string},'topic':string,'boundary_reason':string}
    return {'type':'object','additionalProperties':False,'required':['sessions'],'properties':{'sessions':{'type':'array','items':{'type':'object','additionalProperties':False,'required':list(fields),'properties':fields}}}}

def render(folder, graph, result):
    esc = html.escape
    nodes = {n['id']:n for n in graph['nodes']}
    owners = {nid:s['session_id'] for s in result['sessions'] for nid in s['node_ids']}
    parts = [f'<h1>{esc(folder.name)}</h1>', f'<p>{esc(graph.get("task",{}).get("task_description",""))}</p>', '<p><a href="../index.html">All graphs</a> · <a href="sessions.json">Sessions JSON</a> · <a href="graph.json">Source graph</a></p>']
    md = [f'# {folder.name}', '']
    for s in result['sessions']:
        parts.append(f'<section><h2>{s["session_id"]}: {esc(s["topic"])}</h2><table><tr><th>Order / ID / Type</th><th>Content</th><th>Relations to earlier nodes</th></tr>')
        md.extend([f'## {s["session_id"]}: {s["topic"]}', ''])
        for nid in s['node_ids']:
            n=nodes[nid]; links=[]
            for e in graph['edges']:
                other=e['target'] if e['source']==nid else e['source'] if e['target']==nid else None
                if other is not None and nodes[other]['order']<n['order']:
                    label=f'{e["source"]} —{e["relation"]}→ {e["target"]}'
                    links.append(esc(label)+f' (<a href="#{esc(other)}">{owners[other]}</a>)')
            parts.append(f'<tr id="{esc(nid)}"><td>{n["order"]} / {esc(nid)} / {esc(n["type"])}</td><td>{esc(n["content"])}</td><td>'+ '<br>'.join(links)+'</td></tr>')
            md.append(f'- {n["order"]}. **{nid} ({n["type"]})**: {n["content"]}')
        parts.append(f'</table><p class="boundary">Boundary: {esc(s["boundary_reason"])}</p></section>')
        md.extend(['',f'Boundary: {s["boundary_reason"]}', ''])
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+esc(folder.name)+'</title><style>'+STYLE+'</style>'+''.join(parts))
    (folder/'sessions.md').write_text('\n'.join(md))
    cross=[e for e in graph['edges'] if owners[e['source']]!=owners[e['target']]]
    return {'nodes':len(nodes),'sessions':len(result['sessions']),'session_sizes':[len(s['node_ids']) for s in result['sessions']], 'cross_session_edges':len(cross),'cross_session_evolution_edges':sum(e['relation'] in {'refute','revise','revive'} for e in cross),'total_evolution_edges':sum(e['relation'] in {'refute','revise','revive'} for e in graph['edges'])}

def run_one(entry, args, template):
    name=entry['job']; folder=args.output/name; folder.mkdir(parents=True, exist_ok=True)
    graph=json.loads((args.input/entry['json']).read_text())
    graph['nodes']=sorted(graph['nodes'],key=lambda n:n['order'])
    ids=[n['id'] for n in graph['nodes']]; orders=[n['order'] for n in graph['nodes']]
    if len(set(ids))!=len(ids) or len(set(orders))!=len(orders):
        raise ValueError('Duplicate input IDs/orders')
    signature=hashlib.sha256((json.dumps(graph,sort_keys=True)+template+args.model).encode()).hexdigest()
    fingerprint=folder/'fingerprint.txt'
    if fingerprint.exists() and fingerprint.read_text()!=signature:
        raise ValueError('Input, prompt or model changed; choose a new output directory')
    fingerprint.write_text(signature)
    write(folder/'graph.json',graph)
    write(folder/'schema.json',schema())
    feedback=''
    result=None
    if (folder/'sessions.json').exists():
        result=json.loads((folder/'sessions.json').read_text()); validate(result,graph)
    else:
        for attempt in range(1,4):
            af=folder/f'attempt_{attempt:02d}'; af.mkdir(exist_ok=True)
            prompt=template.replace('{{INPUT_JSON}}',json.dumps(graph,ensure_ascii=False))+feedback
            (af/'prompt.txt').write_text(prompt)
            raw=(af/'response.json').resolve()
            started=time.monotonic()
            try:
                raw.write_text(llm.call(prompt, schema()))
                result=json.loads(raw.read_text()); validate(result,graph)
                write(af/'validation.json',{'passed':True,'seconds':round(time.monotonic()-started,2)})
                write(folder/'sessions.json',result)
                break
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                write(af/'error.json',{'error':str(exc)})
                feedback='\nPrevious attempt failed validation: '+str(exc)+'\nReturn a corrected complete partition.\n'
                if attempt==3: raise
    metrics=render(folder,graph,result)
    status={'job':name,'status':'completed','task_type':entry.get('task_type'),**metrics}
    write(folder/'status.json',status)
    print(f'{name}: {metrics["nodes"]} nodes -> {metrics["sessions"]} sessions; sizes={metrics["session_sizes"]}',flush=True)
    return status

def index(output, statuses):
    statuses=sorted(statuses,key=lambda s:s['job'])
    write(output/'summary.json',{'total':len(statuses),'completed':sum(s['status']=='completed' for s in statuses),'jobs':statuses})
    rows=[]
    for s in statuses:
        name=html.escape(s['job'])
        rows.append(f'<tr><td><a href="{name}/index.html">{name}</a></td><td>{s["status"]}</td><td>{s.get("nodes","")}</td><td>{s.get("sessions","")}</td><td>{s.get("session_sizes",s.get("error",""))}</td><td>{s.get("cross_session_evolution_edges","")} / {s.get("total_evolution_edges","")}</td></tr>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Session partitions</title><style>'+STYLE+'</style><h1>Session partitions</h1><p>Nodes retain their original chronological order. Cross-session relation counts are descriptive, not quality scores.</p><table><tr><th>Graph</th><th>Status</th><th>Nodes</th><th>Sessions</th><th>Nodes per session</th><th>Cross-session / all evolution edges</th></tr>'+''.join(rows)+'</table>')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='accepted/ directory of step 4')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prompt',type=Path,default=HERE/'session_partition_prompt.txt')
    ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    args.model=json.dumps(llm.models(),sort_keys=True)
    template=args.prompt.read_text(); entries=json.loads((args.input/'manifest.json').read_text())
    (args.output/'prompt.txt').write_text(template)
    write(args.output/'config.json',{'input':str(args.input.resolve()),'model':args.model,'workers':args.workers,'graphs':len(entries)})
    statuses=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_one,e,args,template):e for e in entries}
        for f in as_completed(futures):
            try: status=f.result()
            except Exception as exc:
                status={'job':futures[f]['job'],'status':'error','error':str(exc)}
                write(args.output/status['job']/'status.json',status)
                print(status,flush=True)
            statuses.append(status); index(args.output,statuses)
    if any(s['status']=='error' for s in statuses): raise SystemExit(1)

if __name__=='__main__': raise SystemExit(llm.run_guarded(main))
