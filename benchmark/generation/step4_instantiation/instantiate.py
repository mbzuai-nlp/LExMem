#!/usr/bin/env python3
"""Step 4: instantiate (seed task, evolved graph) pairs node by node in graph order.

4.1 the generation model writes the content of the current node (<= 20 words, checked by the program) given the task,
the accepted prefix and the node's edges to earlier nodes; 4.2 the filter model checks that content against those
edges; a rejected node is regenerated with the reviewer's issues as feedback. 4.3 one whole-graph check.
A graph is discarded if a node exhausts its attempts or the whole-graph check fails. Accepted graphs: <output>/accepted/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import llm  # noqa: E402

LOCK = threading.Lock()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def obj_schema(properties):
    return {'type':'object','additionalProperties':False,'required':list(properties),'properties':properties}

STR = {'type':'string'}
EDGE = obj_schema({'source':STR,'relation':STR,'target':STR})
GEN_SCHEMA = obj_schema({'node_id':STR,'content':STR})
CHECK_SCHEMA = obj_schema({'node_id':STR,'passed':{'type':'boolean'},'issues':{'type':'array','items':obj_schema({'edge':{'anyOf':[EDGE,{'type':'null'}]},'reason':STR})}})


FULL_CHECK_SCHEMA = obj_schema({'passed':{'type':'boolean'},'issues':{'type':'array','items':obj_schema({'node_ids':{'type':'array','items':STR},'edge':{'anyOf':[EDGE,{'type':'null'}]},'reason':STR})}})
TERMINAL = {'accepted','discarded_node_retry','discarded_full_graph'}

def valid_full_review(value, graph):
    if set(value) != {'passed','issues'} or type(value['passed']) is not bool or not isinstance(value['issues'],list):
        raise ValueError('Invalid full-graph review fields')
    if value['passed'] != (len(value['issues'])==0):
        raise ValueError('Full-graph passed/issues mismatch')
    ids={n['id'] for n in graph['nodes']}
    for issue in value['issues']:
        if set(issue) != {'node_ids','edge','reason'} or not isinstance(issue['node_ids'],list) or not issue['node_ids'] or not all(i in ids for i in issue['node_ids']):
            raise ValueError('Invalid full-graph issue node IDs')
        if not isinstance(issue['reason'],str) or not issue['reason'].strip() or (issue['edge'] is not None and issue['edge'] not in graph['edges']):
            raise ValueError('Invalid full-graph issue reason or edge')


def payload_for(task, graph, accepted, current):
    ids = {n['id'] for n in accepted}
    prefix_edges = [e for e in graph['edges'] if e['source'] in ids and e['target'] in ids]
    current_edges = [e for e in graph['edges'] if (e['source']==current['id'] and e['target'] in ids) or (e['target']==current['id'] and e['source'] in ids)]
    return {'task':task,'prefix_graph':{'nodes':accepted,'edges':prefix_edges},'current_node':current,'current_edges':current_edges}

def valid_generation(value, node):
    if set(value) != {'node_id','content'} or value['node_id'] != node['id'] or not isinstance(value['content'],str) or not value['content'].strip():
        raise ValueError('Invalid node id or content')

def valid_review(value, node, edges):
    if set(value) != {'node_id','passed','issues'} or value['node_id'] != node['id'] or type(value['passed']) is not bool or not isinstance(value['issues'],list):
        raise ValueError('Invalid review fields')
    if value['passed'] != (len(value['issues']) == 0):
        raise ValueError('Review passed/issues mismatch')
    for issue in value['issues']:
        if set(issue) != {'edge','reason'} or not isinstance(issue['reason'],str) or not issue['reason'].strip() or (issue['edge'] is not None and issue['edge'] not in edges):
            raise ValueError('Invalid review issue or edge')

NEXT_GEN = ('NEXT GRAPH NODE\nThe instructions above still apply. The prefix graph is the task and the accepted nodes earlier in this '
            'conversation, in order: each node\'s edges are in its own turn and its content in the reply. Fill this node:\n')
NEXT_CHK = ('NEXT GRAPH CHECK\nThe instructions above still apply. The accepted prefix is the earlier candidates in this '
            'conversation, in order, each with its edges. Check this candidate:\n')

def gen_text(template, payload, first):
    if first:
        return template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False, separators=(',',':')))
    body = {k: payload[k] for k in ('current_node','current_edges','retry_feedback') if k in payload}
    return NEXT_GEN + json.dumps(body, ensure_ascii=False, separators=(',',':'))

def chk_text(template, payload, first):
    if first:
        return template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False, separators=(',',':')))
    body = {k: payload[k] for k in ('current_node','current_edges')}
    return NEXT_CHK + json.dumps(body, ensure_ascii=False, separators=(',',':'))

def accepted_pair(gen_user, chk_user, candidate):
    # turns an accepted node adds to the generator / checker conversations (deterministic, so a resumed job rebuilds them)
    return ([{'role':'user','content':gen_user},{'role':'assistant','content':json.dumps({'node_id':candidate['id'],'content':candidate['content']},ensure_ascii=False)}],
            [{'role':'user','content':chk_user},{'role':'assistant','content':json.dumps({'node_id':candidate['id'],'passed':True,'issues':[]},ensure_ascii=False)}])

def call_model(folder, template, payload, schema, args, validator, max_calls=3, user_text=None, history=None, cache_key=None):
    folder.mkdir(parents=True, exist_ok=True)
    prompt = user_text if user_text is not None else template.replace('{{INPUT_JSON}}', json.dumps(payload, ensure_ascii=False, separators=(',',':')))
    (folder/'prompt.txt').write_text(prompt)
    write(folder/'input.json', payload)
    write(folder/'schema.json', schema)
    start=max([int(p.stem.split('_')[-1]) for pat in ('response_*.txt','error_*.json') for p in folder.glob(pat)],default=0)+1
    for retry in range(start,start+max_calls):
        raw = (folder/f'response_{retry}.txt').resolve()
        t0=time.monotonic()
        try:
            raw.write_text(llm.call(prompt, schema, history=history, cache_key=cache_key) if (history or cache_key) else llm.call(prompt, schema))
            value=json.loads(raw.read_text())
            validator(value)
            write(folder/'result.json',value)
            write(folder/'metadata.json',{'model':args.model,'elapsed_seconds':time.monotonic()-t0,'raw_response':raw.name,'completed_at':datetime.now(timezone.utc).isoformat()})
            return value
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            write(folder/f'error_{retry}.json',{'error':str(exc)})
            if retry==start+max_calls-1: raise

STYLE='body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:9px;text-align:left;vertical-align:top}th{background:#eef2f7}code{font-size:12px}.meta{color:#555}a{color:#165abc}'
def render_job(folder,task,graph,accepted,status):
    lines=['# '+folder.name,'',task['task_description'],'']+['- '+x for x in task['requirements']]+['',f'Status: {status}; accepted {len(accepted)}/{len(graph["nodes"])} nodes.','']
    rows=[]
    for n in accepted:
        es=[e for e in graph['edges'] if (e['source']==n['id'] and any(a['id']==e['target'] and a['order']<n['order'] for a in accepted)) or (e['target']==n['id'] and any(a['id']==e['source'] and a['order']<n['order'] for a in accepted))]
        edge_text='; '.join(f'{e["source"]} —{e["relation"]}→ {e["target"]}' for e in es)
        lines += [f'## {n["order"]}. {n["id"]} ({n["type"]})','',n['content'],'',edge_text,'']
        link='' if n['type']=='goal' else f'<a href="nodes/{n["order"]:03d}_{n["id"]}/accepted_review.json">check</a>'
        rows.append(f'<tr><td>{n["order"]}</td><td>{html.escape(n["id"])}</td><td>{n["type"]}</td><td>{html.escape(n["content"])}</td><td>{html.escape(edge_text)}</td><td>{link}</td></tr>')
    (folder/'nodes.md').write_text('\n'.join(lines))
    (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{folder.name}</title><style>{STYLE}</style><h1>{folder.name}</h1><p>{html.escape(task["task_description"])}</p><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in task['requirements'])+f'</ul><p>{status}: {len(accepted)}/{len(graph["nodes"])} nodes | <a href="prefix.json">accepted prefix</a> | <a href="nodes.md">Markdown</a></p><table><tr><th>Order</th><th>ID</th><th>Type</th><th>Content</th><th>Relations to earlier nodes</th><th>Review</th></tr>'+''.join(rows)+'</table>')

def run_job(job,args,templates):
    task,graph,folder=job
    folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        if previous['status'] in TERMINAL:
            return previous
    write(folder/'task.json',task)
    write(folder/'skeleton.json',graph)
    ordered=sorted(graph['nodes'],key=lambda n:n['order'])
    assert ordered[0]['type']=='goal' and sum(n['type']=='goal' for n in ordered)==1
    assert [n['order'] for n in ordered]==list(range(1,len(ordered)+1))
    goal={**ordered[0],'content':task['task_description'],'requirements':task['requirements']}
    accepted=[goal]
    max_attempts=1+args.max_node_retries
    append=getattr(args,'append_mode',False)
    gen_hist,chk_hist=[],[]   # append mode: generator / checker conversations so far
    name=folder.name
    def checkpoint(status, **extra):
        ids={n['id'] for n in accepted}
        write(folder/'prefix.json',{'nodes':accepted,'edges':[e for e in graph['edges'] if e['source'] in ids and e['target'] in ids]})
        data={'job':folder.name,'status':status,'accepted_nodes':len(accepted),'total_nodes':len(ordered),**extra}
        write(status_path,data)
        render_job(folder,task,graph,accepted,status)
        return data
    checkpoint('running')
    af=nf=None
    try:
        for node in ordered[1:]:
            nf=folder/'nodes'/f'{node["order"]:03d}_{node["id"]}'
            nf.mkdir(parents=True,exist_ok=True)
            payload=payload_for(task,graph,accepted,node)
            if (nf/'accepted.json').exists():
                saved=json.loads((nf/'accepted.json').read_text())
                review=json.loads((nf/'accepted_review.json').read_text())
                valid_generation({'node_id':saved['id'],'content':saved['content']},node)
                assert {k:saved[k] for k in node}==node
                valid_review(review,node,payload['current_edges'])
                assert review['passed'] and len(saved['content'].split()) <= args.max_content_words
                assert max([int(x.name.split('_')[-1]) for x in nf.glob('attempt_*')],default=0)<=max_attempts
                if append:
                    gp,cp=accepted_pair(gen_text(templates[0],{**payload,'retry_feedback':[]},not gen_hist),
                                        chk_text(templates[1],{**payload,'current_node':saved},not chk_hist),saved)
                    gen_hist+=gp; chk_hist+=cp
                accepted.append(saved)
                continue
            checkpoint('running',current_node_id=node['id'])
            attempts=sorted(nf.glob('attempt_*'))
            feedback=[]
            for attempt_dir in attempts:
                for relative in ('local_check.json','generation_failure.json','check/result.json'):
                    rp=attempt_dir/relative
                    if rp.exists():
                        feedback.extend(json.loads(rp.read_text()).get('issues',[]))
            # attempts persisted before a restart count against the same budget
            start=max([int(p.name.split('_')[-1]) for p in attempts],default=0)+1
            for attempt in range(start,max_attempts+1):
                af=nf/f'attempt_{attempt:03d}'
                af.mkdir(parents=True,exist_ok=True)
                try:
                    generation=call_model(af/'generation',templates[0],{**payload,'retry_feedback':feedback},GEN_SCHEMA,args,lambda v:valid_generation(v,node),max_calls=1,
                                          user_text=gen_text(templates[0],{**payload,'retry_feedback':feedback},not gen_hist) if append else None,
                                          history=gen_hist if append else None,cache_key=f'{name}:gen' if append else None)
                except (OSError,ValueError,TypeError,subprocess.SubprocessError) as exc:
                    issue={'edge':None,'reason':f'Generation failed: {type(exc).__name__}: {exc}'}
                    feedback.append(issue)
                    write(af/'generation_failure.json',{'passed':False,'issues':[issue]})
                    continue
                candidate={**node,'content':generation['content']}
                word_count=len(candidate['content'].split())
                if word_count > args.max_content_words:
                    issue={'edge':None,'reason':f'Content has {word_count} words; maximum is {args.max_content_words}.'}
                    feedback.append(issue)
                    write(af/'local_check.json',{'passed':False,'issues':[issue]})
                    with LOCK: print(f'{folder.name}: {node["id"]} exceeds word limit ({word_count}); attempt {attempt}/{max_attempts}',flush=True)
                    continue
                chk_payload={**payload,'current_node':candidate}
                review=call_model(af/'check',templates[1],chk_payload,CHECK_SCHEMA,args,lambda v:valid_review(v,node,payload['current_edges']),
                                  user_text=chk_text(templates[1],chk_payload,not chk_hist) if append else None,
                                  history=chk_hist if append else None,cache_key=f'{name}:chk' if append else None)
                if review['passed']:
                    write(nf/'accepted_review.json',review)
                    write(nf/'accepted.json',candidate)
                    if append:
                        gp,cp=accepted_pair(gen_text(templates[0],{**payload,'retry_feedback':[]},not gen_hist),chk_text(templates[1],chk_payload,not chk_hist),candidate)
                        gen_hist+=gp; chk_hist+=cp
                    accepted.append(candidate)
                    checkpoint('running')
                    with LOCK: print(f'{folder.name}: {node["order"]}/{len(ordered)} {node["id"]} accepted (attempt {attempt}/{max_attempts})',flush=True)
                    break
                feedback.extend(review['issues'])
                with LOCK: print(f'{folder.name}: {node["id"]} rejected ({attempt}/{max_attempts}): '+ '; '.join(x['reason'] for x in review['issues'])[:220],flush=True)
            else:
                return checkpoint('discarded_node_retry',node_id=node['id'],attempts_used=max_attempts,retries_used=args.max_node_retries,issues=feedback)
        result={**graph,'task':task,'nodes':accepted}
        assert result['edges']==graph['edges']
        write(folder/'candidate.json',result)
        checkpoint('checking_full_graph')
        rf=folder/'full_graph_check'
        if (rf/'result.json').exists():
            review=json.loads((rf/'result.json').read_text())
            valid_full_review(review,result)
        else:
            review=call_model(rf,templates[2],result,FULL_CHECK_SCHEMA,args,lambda v:valid_full_review(v,result))
        if not review['passed']:
            return checkpoint('discarded_full_graph',issues=review['issues'])
        write(folder/'graph.json',result)
        write(args.output/'accepted'/f'{folder.name}.json',result)
        return checkpoint('accepted')
    except llm.PauseRun:
        # paused run: drop the unfinished attempt so the restart neither counts it nor mistakes it for a finished one
        if af is not None and nf is not None and not (nf/'accepted.json').exists(): shutil.rmtree(af,ignore_errors=True)
        raise
    except Exception as exc:
        with LOCK: print(f'{folder.name}: ERROR {exc}',flush=True)
        return checkpoint('error',error=str(exc))

def render_index(output,jobs):
    statuses=[];rows=[];kept=[]
    for task,graph,folder in jobs:
        p=folder/'status.json'
        state=json.loads(p.read_text()) if p.exists() else {'status':'pending','accepted_nodes':0,'total_nodes':len(graph['nodes'])}
        statuses.append({'job':folder.name,**state})
        relative=folder.relative_to(output).as_posix()
        link=f'<a href="accepted/{folder.name}.json">JSON</a>' if state['status']=='accepted' else ''
        if state['status']=='discarded_full_graph':link=f'<a href="{relative}/full_graph_check/result.json">Rejection</a>'
        elif state['status']=='discarded_node_retry':link=f'<a href="{relative}/status.json">Rejection</a>'
        rows.append(f'<tr><td><a href="{relative}/index.html">{folder.name}</a></td><td>{html.escape(task["task_description"])}</td><td>{state["status"]}</td><td>{state["accepted_nodes"]}/{state["total_nodes"]}</td><td>{link}</td></tr>')
        if state['status']=='accepted':
            kept.append({'job':folder.name,'task_type':task.get('task_type'),'json':f'{folder.name}.json','preview':f'../{relative}/index.html'})
    counts={status:sum(s['status']==status for s in statuses) for status in ('accepted','discarded_node_retry','discarded_full_graph','error','running','checking_full_graph','pending')}
    write(output/'summary.json',{'total':len(jobs),**counts,'jobs':statuses})
    (output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Filtered instantiated graphs</title><style>{STYLE}</style><h1>Instantiation and filtering</h1><p><a href="accepted/index.html">Accepted graphs only</a></p><p>Per-node generation and check, then one whole-graph check. Rejections are final.</p><table><tr><th>Task / Graph</th><th>Task</th><th>Status</th><th>Nodes</th><th>Result</th></tr>'+''.join(rows)+'</table>')
    out=output/'accepted';out.mkdir(exist_ok=True)
    write(out/'manifest.json',kept)
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Accepted graphs</title><style>{STYLE}</style><h1>Accepted graphs ({len(kept)})</h1><p>Passed node checks and final whole-graph review.</p><ul>'+''.join(f'<li><a href="{r["preview"]}">{r["job"]}</a> — {r["task_type"]} — <a href="{r["json"]}">JSON</a></li>' for r in kept)+'</ul>')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=Path,required=True,help='JSON list of {"job", "task", "graph"} (see make_pairs.py)')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--max-node-retries',type=int,default=9,help='Retries per node after the first attempt; a node that exhausts them discards the graph')
    ap.add_argument('--max-content-words',type=int,default=20)
    ap.add_argument('--append-mode',action=argparse.BooleanOptionalAction,default=True,
                    help='Send 4.1/4.2 of one graph as growing multi-turn conversations (prompt-cache friendly); --no-append-mode sends independent calls')
    a=ap.parse_args()
    pairs=json.loads(a.pairs.read_text())
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(output=out,workers=a.workers,max_node_retries=a.max_node_retries,max_content_words=a.max_content_words,
                         append_mode=a.append_mode,model=llm.models())
    templates=[(HERE/p).read_text() for p in ('node_instantiation_prompt.txt','node_relation_check_prompt.txt','graph_instantiation_check_prompt.txt')]
    jobs=[(p['task'],json.loads(Path(p['graph']).read_text()),out/'jobs'/p['job']) for p in pairs]
    render_index(out,jobs)
    with ThreadPoolExecutor(a.workers) as pool:
        futures=[pool.submit(run_job,j,args,templates) for j in jobs]
        for f in as_completed(futures):
            print(json.dumps(f.result(),ensure_ascii=False)[:300],flush=True)
            render_index(out,jobs)
    s=json.loads((out/'summary.json').read_text())
    print('Finished',{k:v for k,v in s.items() if k!='jobs'})
    return 0 if s['error']==0 else 1

if __name__=='__main__':
    raise SystemExit(llm.run_guarded(main))
