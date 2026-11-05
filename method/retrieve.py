"""Read phase: score units for a question and select anchors and neighbors within token budgets.

Units are scored by cos(question, Key) and BM25(question, Key text), fused by reciprocal rank
fusion (RRF). Anchors are the top units under a budget on their original text; neighbors are units
within `hops` edges of the anchors, ranked by the same score under a budget on title + summary.
"""
from collections import Counter
import math
import re

from common import cosine, count_tokens
from graph_build import key_text

RRF_K = 10


def _tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


def _bm25_scores(query_tokens, docs_tokens, k1=1.5, b=0.75):
    """BM25 over all units of one dialogue."""
    n = len(docs_tokens)
    df = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    avgdl = sum(len(t) for t in docs_tokens) / n
    idf = {t: math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1) for t in set(query_tokens)}
    scores = []
    for toks in docs_tokens:
        dl = len(toks)
        tf = Counter(toks)
        s = 0.0
        for t in query_tokens:
            if t in tf:
                f = tf[t]
                s += idf.get(t, 0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _rrf_fuse(cos_scores, bm25_scores, k=RRF_K):
    cos_rank = {uid: i for i, uid in enumerate(sorted(cos_scores, key=lambda u: -cos_scores[u]))}
    bm25_rank = {uid: i for i, uid in enumerate(sorted(bm25_scores, key=lambda u: -bm25_scores[u]))}
    return {uid: 1 / (k + cos_rank[uid] + 1) + 1 / (k + bm25_rank[uid] + 1) for uid in cos_scores}


def hybrid_scores(units, unit_key_vectors, question, question_vector, rrf_k=RRF_K):
    cos_scores = {u['unit_id']: cosine(question_vector, unit_key_vectors[u['unit_id']]) for u in units}
    bm25 = dict(zip((u['unit_id'] for u in units),
                    _bm25_scores(_tokenize(question), [_tokenize(key_text(u)) for u in units])))
    return _rrf_fuse(cos_scores, bm25, k=rrf_k)


def _greedy_fill(ordered_units, token_budget, text_of):
    """Take units in order, skipping any unit that would exceed the budget.
    The top-ranked unit is always taken so the selection is never empty."""
    selected, total = [], 0
    for u in ordered_units:
        cost = count_tokens(text_of(u))
        if selected and total + cost > token_budget:
            continue
        selected.append(u)
        total += cost
        if total >= token_budget:
            break
    return selected


def expand_neighbors(units, edges, anchor_ids, hops=1):
    """Units within `hops` edges of the anchor set (edge direction ignored), excluding anchors."""
    by_id = {u['unit_id']: u for u in units}
    adjacency = {}
    for e in edges:
        adjacency.setdefault(e['source'], set()).add(e['target'])
        adjacency.setdefault(e['target'], set()).add(e['source'])
    reached, frontier = set(), set(anchor_ids)
    for _ in range(hops):
        frontier = {nid for uid in frontier for nid in adjacency.get(uid, ())
                    if nid not in anchor_ids and nid not in reached}
        if not frontier:
            break
        reached.update(frontier)
    return [by_id[uid] for uid in reached]


def score_and_select(units, edges, embedder, question, *, token_budget, neighbor_token_budget,
                     messages_by_id, neighbor_hops=1, rrf_k=RRF_K):
    unit_key_vectors = dict(zip((u['unit_id'] for u in units),
                                embedder.embed_documents([key_text(u) for u in units])))
    question_vector = embedder.embed_queries([question])[0]
    s = hybrid_scores(units, unit_key_vectors, question, question_vector, rrf_k)
    rank = lambda u: (-s[u['unit_id']], u['order'])  # noqa: E731
    anchors = _greedy_fill(sorted(units, key=rank), token_budget,
                           lambda u: '\n'.join(messages_by_id[mid]['content'] for mid in u['message_ids']))
    anchor_ids = {u['unit_id'] for u in anchors}
    candidates = expand_neighbors(units, edges, anchor_ids, hops=neighbor_hops)
    neighbors = _greedy_fill(sorted(candidates, key=rank), neighbor_token_budget,
                             lambda u: u['title'] + '\n' + u['summary'])
    return anchors, neighbors
