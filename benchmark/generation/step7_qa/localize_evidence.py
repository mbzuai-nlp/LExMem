"""Step 7.4: localize message-level evidence for an accepted question and have it checked independently.

The filter model selects, per answer slot, the messages (from the slot's cited source nodes) that support the answer;
a second call verifies them. Rejections are fed back for at most `evidence_retries` regenerations.
"""
from generate_qa import write, invoke, validate_review
MAX_EVIDENCE_MESSAGES_PER_SLOT=20


def index_messages(annotated, dialogue_id):
    result=[];seen=set()
    for session in annotated['sessions']:
        sid=session['session_id']
        if sid in seen: raise ValueError('Duplicate session ID')
        seen.add(sid);number=0
        for block in session['blocks']:
            for m in block['messages']:
                number+=1
                result.append({'message_id':f'{dialogue_id}:{sid}:M{number:04d}',
                    'session_id':sid,'source_node_id':block['source_node_id'],
                    'role':m['role'],'content':m['content']})
    return result


def validate_evidence_review(value):
    # accept structured issues in a negative verdict by flattening them to strings
    if isinstance(value, dict) and value.get('passed') is False and isinstance(value.get('issues'), list):
        normalized=[]
        for item in value['issues']:
            if isinstance(item, dict) and set(item)=={'slot','message_ids','issue'}:
                if (not isinstance(item['slot'],str) or not isinstance(item['issue'],str)
                        or not item['issue'].strip() or not isinstance(item['message_ids'],list)
                        or not all(isinstance(x,str) for x in item['message_ids'])):
                    raise ValueError('Invalid structured review issue')
                normalized.append(f"{item['slot']} [{', '.join(item['message_ids'])}]: {item['issue']}")
            else: normalized.append(item)
        value['issues']=normalized
    validate_review(value)


def validate_localization_response(value):
    """Validate the response shape only; invalid evidence is handled as a rejection with feedback."""
    if not isinstance(value,dict) or set(value)!={'passed','evidence','issues'}:
        raise ValueError('Invalid localization fields')
    validate_review({k:value[k] for k in ('passed','issues')})
    if not isinstance(value['evidence'],dict):raise ValueError('Evidence must be an object')


def validate_evidence(value, qa, messages, max_messages=MAX_EVIDENCE_MESSAGES_PER_SLOT):
    if not isinstance(value,dict) or set(value)!={'passed','evidence','issues'}:
        raise ValueError('Invalid localization fields')
    validate_review({k:value[k] for k in ('passed','issues')})
    evidence=value['evidence']
    if not value['passed']:
        if evidence!={}:raise ValueError('Failed localization must have empty evidence')
        return
    if not isinstance(evidence,dict) or set(evidence)!=set(qa['answer']):
        raise ValueError('Evidence must cover all answer slots')
    if type(max_messages) is not int or max_messages<1:
        raise ValueError('Invalid evidence message limit')
    by_id={m['message_id']:m for m in messages}
    for slot, bundle in evidence.items():
        if not isinstance(bundle,list) or not bundle:raise ValueError('Missing evidence messages')
        if any(isinstance(mid,list) for mid in bundle):
            raise ValueError('Evidence alternatives are not supported')
        if not all(isinstance(mid,str) for mid in bundle):raise ValueError('Invalid evidence message ID')
        if len(bundle)>max_messages:
            raise ValueError(f'Too many evidence messages for {slot}: {len(bundle)} > {max_messages}')
        if len(set(bundle))!=len(bundle):raise ValueError('Repeated message ID')
        for mid in bundle:
            if mid not in by_id or by_id[mid]['source_node_id'] not in qa['evidence_node_ids'][slot]:
                raise ValueError('Evidence outside slot source nodes')



def localize_question(i, qa, messages, dialogue_id, templates, args):
    """One initial annotation plus bounded feedback-driven regeneration attempts."""
    folder=args.output/f'Q{i:03d}'
    ids={n for values in qa['evidence_node_ids'].values() for n in values}
    selected=[m for m in messages if m['source_node_id'] in ids]
    history=[]
    for attempt in range(1,args.evidence_retries+2):
        attempt_folder=folder/'attempts'/f'{attempt:02d}'
        payload={'qa':qa,'messages':selected,'constraints':{'max_evidence_messages_per_slot':MAX_EVIDENCE_MESSAGES_PER_SLOT}}
        if history:
            payload['previous_rejections']=[{
                'attempt':h['attempt'], 'stage':h['status'],
                'evidence':h.get('invalid_localization',h['localization'])['evidence'],
                'issues':h.get('review',h['localization'])['issues']
            } for h in history]
        try:
            proposal=invoke(attempt_folder/'localize',templates['evidence_localize'],payload,args,
                            validate_localization_response)
            record={'attempt':attempt,'localization':proposal}
            try:
                validate_evidence(proposal,qa,selected)
            except (ValueError,TypeError,KeyError) as exc:
                record['invalid_localization']=proposal
                proposal={'passed':False,'evidence':{},'issues':['Invalid evidence annotation: '+str(exc)]}
                record['localization']=proposal
            if not proposal['passed']:
                record['status']='rejected_localization'
            else:
                review=invoke(attempt_folder/'check',templates['evidence_check'],
                              {'qa':qa,'evidence':proposal['evidence'],'messages':selected,'constraints':{'max_evidence_messages_per_slot':MAX_EVIDENCE_MESSAGES_PER_SLOT}},args,validate_evidence_review)
                validate_evidence_review(review)
                record.update(status='accepted' if review['passed'] else 'rejected_evidence',review=review)
        except Exception as exc:
            # technical failure (already retried inside invoke): not a semantic rejection, and the QA is not exported
            record={'attempt':attempt,'status':'error','error':str(exc)}
        history.append(record)
        write(attempt_folder/'result.json',record)
        write(folder/'attempt_history.json',history)
        if record['status'] in ('accepted','error'):break
    last=history[-1]
    if last['status']=='accepted':
        result={'status':'accepted','qa':{**qa,'dialogue_id':dialogue_id,
                'evidence_unit':'message','evidence':last['localization']['evidence']},
                'review':last['review']}
    elif last['status']=='error':
        result={'status':'error','qa':qa,'error':last['error']}
    else:
        result={'status':'discarded_evidence','qa':qa,
                'issues':last.get('review',last['localization'])['issues'],
                'last_rejection_stage':last['status']}
    result.update(index=i,annotation_attempts=len(history),max_annotation_attempts=args.evidence_retries+1)
    write(folder/'result.json',result)
    print(i,result['status'],f"({len(history)} annotation attempts)",flush=True)
    return result


def exportable_questions(results, messages):
    """Re-validate accepted results; incomplete or unchecked evidence never reaches qa.json."""
    accepted=[]
    for r in results:
        if r['status']!='accepted':continue
        validate_evidence_review(r['review'])
        if not r['review']['passed']:raise ValueError('Accepted result has a negative review')
        qa=r['qa']
        validate_evidence({'passed':True,'evidence':qa.get('evidence'),'issues':[]},qa,messages)
        accepted.append(qa)
    return accepted
