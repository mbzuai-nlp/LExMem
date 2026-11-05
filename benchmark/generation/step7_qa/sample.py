"""Step 7.0: sample question foci from an instantiated graph (no model calls).

dense     : connected induced subgraphs with E >= 2V (4-16 nodes), preferring revision / conflicting-evidence patterns
            (a refuted claim and its revision, optionally supported by new evidence; or support and refutation of
            the same claim), with node-overlap diversity.
sparse    : chordless paths whose two endpoints are connected only through the goal node, grown until no chordless
            extension remains, preferring long paths with a large order span between the endpoints.
same_type : all nodes of one node type.
"""
from collections import defaultdict
import random


class Graph:
    def __init__(self, graph):
        self.raw = graph
        self.nodes = {n['id']: n for n in graph['nodes']}
        if not self.nodes or len(self.nodes) != len(graph['nodes']):
            raise ValueError('Empty graph or duplicate node IDs')
        if any(type(n.get('order')) is not int for n in self.nodes.values()):
            raise ValueError('Each node needs integer order')
        if len({n['order'] for n in self.nodes.values()}) != len(self.nodes):
            raise ValueError('Duplicate node orders')
        roots = [n['id'] for n in self.nodes.values() if n['type'] == 'goal']
        if len(roots) != 1:
            raise ValueError('Expected exactly one goal root')
        self.root = roots[0]
        self.edges = []
        self.adj = {n: set() for n in self.nodes}
        seen = set()
        for e in graph['edges']:
            s, r, t = e['source'], e['relation'], e['target']
            if s not in self.nodes or t not in self.nodes or s == t:
                raise ValueError('Unknown edge endpoint or self-loop')
            if (s, r, t) in seen:
                continue
            seen.add((s, r, t))
            self.edges.append(e)
            self.adj[s].add(t)
            self.adj[t].add(s)

    def ordered(self, ids):
        return sorted(ids, key=lambda x: (self.nodes[x]['order'], x))

    def induced(self, ids):
        ids = set(ids)
        return {'nodes': [self.nodes[x] for x in self.ordered(ids)],
                'edges': [e for e in self.edges if e['source'] in ids and e['target'] in ids]}

    def candidate(self, ids):
        sub = self.induced(ids)
        orders = [n['order'] for n in sub['nodes']]
        return {'node_ids': self.ordered(ids), 'subgraph': sub,
                'node_count': len(ids), 'edge_count': len(sub['edges']),
                'order_span': max(orders) - min(orders)}


def diversify(pool, count, score, tier=None):
    """Prefer tiers when supplied; penalize node overlap within each tier."""
    tier = tier or (lambda c: 0)
    pool = sorted(pool, key=lambda c: (tier(c), score(c), tuple(c['node_ids'])), reverse=True)
    result = []
    while pool and len(result) < count:
        def priority(c):
            a = set(c['node_ids'])
            overlap = max((len(a & set(s['node_ids'])) / len(a | set(s['node_ids']))
                           for s in result), default=0)
            return tier(c), score(c) * (1 - 0.65 * overlap)
        best = max(range(len(pool)), key=lambda i: priority(pool[i]))
        result.append(pool.pop(best))
    return result


def dense_context_groups(g):
    """Find connected revision/contrast patterns from explicit directed relations.

    These are structural sampling hints, not inferred causal chains or a
    guarantee that the resulting question requires multiple nodes.
    """
    support, refute = defaultdict(set), defaultdict(set)
    for e in g.edges:
        if g.nodes[e['source']]['type'] != 'evidence':
            continue
        if e['relation'] == 'support':
            support[e['target']].add(e['source'])
        elif e['relation'] == 'refute':
            refute[e['target']].add(e['source'])
    groups = {name: set() for name in
              ('challenged_revision', 'supported_revision', 'conflicting_evidence')}
    for e in g.edges:
        if e['relation'] != 'revise':
            continue
        revised, original = e['source'], e['target']
        for counterexample in refute.get(original, ()):
            base = frozenset((original, revised, counterexample))
            if len(base) != 3:
                continue
            groups['challenged_revision'].add(base)
            for evidence in support.get(revised, ()):
                extended = base | {evidence}
                if len(extended) == 4:
                    groups['supported_revision'].add(extended)
    for target in support.keys() & refute.keys():
        for positive in support[target]:
            for negative in refute[target]:
                group = frozenset((target, positive, negative))
                if len(group) == 3:
                    groups['conflicting_evidence'].add(group)
    return {name: sorted(values, key=lambda ids: tuple(g.ordered(ids)))
            for name, values in groups.items()}


def dense_tier(candidate):
    """Prefer a supported revision, then a challenged revision or contrast."""
    signals = candidate['selection_signals']
    if signals['supported_revision']:
        return 2
    return int(bool(signals['challenged_revision'] or signals['conflicting_evidence']))


def dense_samples(g, count=5, min_nodes=4, max_nodes=16, trials=3000, seed=0):
    """Prefer revision/contrast context while retaining connected E >= 2V subgraphs."""
    if count < 0 or min_nodes < 1 or max_nodes < min_nodes or trials < 1:
        raise ValueError('Invalid dense sampling limits')
    if count == 0:
        return []
    rng = random.Random(seed)
    pool = {}
    nodes = g.ordered(g.nodes)
    groups = dense_context_groups(g)
    seeds = sorted({ids for values in groups.values() for ids in values
                    if len(ids) <= max_nodes}, key=lambda ids: tuple(g.ordered(ids)))
    for trial in range(trials):
        # Retain random-node exploration; alternate with connected context seeds.
        ids = set(rng.choice(seeds)) if seeds and trial % 2 == 0 else {rng.choice(nodes)}
        for size in range(len(ids), min(max_nodes, len(nodes)) + 1):
            if size >= min_nodes:
                c = g.candidate(ids)
                if c['edge_count'] >= 2 * c['node_count']:
                    pool[tuple(c['node_ids'])] = c
            frontier = set().union(*(g.adj[x] for x in ids)) - ids
            if not frontier:
                break
            choices = g.ordered(frontier)
            # Mix uniform growth with growth toward internally well-connected nodes.
            weights = [1 + (len(g.adj[x] & ids) ** 2 if rng.random() < .7 else 0)
                       for x in choices]
            ids.add(rng.choices(choices, weights=weights, k=1)[0])
    for c in pool.values():
        ids = set(c['node_ids'])
        c['selection_signals'] = {name: sum(group <= ids for group in values)
                                  for name, values in groups.items()}
    return diversify(list(pool.values()), count,
                     lambda c: 1 + (c['edge_count'] - 2 * c['node_count']) / c['node_count'],
                     tier=dense_tier)


def sparse_samples(g, count=7, trials=6000, seed=0):
    """Grow chordless paths through goal without a node-count cap."""
    if count < 0 or trials < 1:
        raise ValueError('Invalid sparse sampling limits')
    if count == 0:
        return []
    rng = random.Random(seed)
    pool = {}
    neighbors = g.ordered(g.adj[g.root])
    if len(neighbors) < 2:
        return []
    for _ in range(trials):
        a, b = rng.sample(neighbors, 2)
        if b in g.adj[a]:
            continue
        left, right = [g.root, a], [g.root, b]
        ids = {g.root, a, b}
        while True:
            path = list(reversed(left)) + right[1:]
            c = g.candidate(ids)
            c.update(endpoint_ids=[left[-1], right[-1]], root_id=g.root,
                     path_node_ids=path, path_length=len(path) - 1,
                     endpoint_order_span=abs(g.nodes[left[-1]]['order'] - g.nodes[right[-1]]['order']))
            pool[tuple(c['node_ids'])] = c
            # Every new node touches exactly one selected node: no hidden chords/cycles.
            choices = [(side, x) for side, arm in enumerate((left, right))
                       for x in g.ordered(g.adj[arm[-1]] - ids)
                       if g.adj[x] & ids == {arm[-1]}]
            if not choices:
                break
            side, x = rng.choice(choices)
            (left, right)[side].append(x)
            ids.add(x)
    span = max(n['order'] for n in g.nodes.values()) - min(n['order'] for n in g.nodes.values())
    return diversify(list(pool.values()), count,
                     lambda c: 1 + c['path_length'] / max(1, len(g.nodes) - 1)
                     + c['endpoint_order_span'] / max(1, span))


def sampling_counts(count=None, sparse_count=None):
    """Keep explicit --count overrides; otherwise default to dense=5, sparse=7."""
    for value in (count, sparse_count):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Candidate counts must be nonnegative integers')
    return (5 if count is None else count,
            sparse_count if sparse_count is not None else (7 if count is None else count))


def prepare(graph, count=None, seed=0, dense_trials=3000, sparse_trials=6000,
            dense_min_nodes=4, dense_max_nodes=16, node_types=None,
            sparse_count=None, methods=None):
    dense_count, sparse_count = sampling_counts(count, sparse_count)
    methods = ('dense', 'sparse', 'same_type') if methods is None else tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - {'dense', 'sparse', 'same_type'}:
        raise ValueError('Select distinct known sampling methods')
    g = Graph(graph)
    groups = defaultdict(list)
    for x in g.ordered(g.nodes):
        groups[g.nodes[x]['type']].append(x)
    if node_types and set(node_types) - groups.keys():
        raise ValueError('Requested node type absent from graph')
    by_type = [{'target_type': kind, 'focus_node_ids': ids,
                'focus_nodes': [g.nodes[x] for x in ids]}
               for kind, ids in sorted(groups.items()) if not node_types or kind in node_types]
    return {'full_graph': graph, 'root_id': g.root,
            'sampling': {'seed': seed, 'count_per_method': {'dense': dense_count, 'sparse': sparse_count},
                         'methods': list(methods),
                         'dense_trials': dense_trials, 'sparse_trials': sparse_trials,
                         'dense_min_nodes': dense_min_nodes, 'dense_condition': 'E >= 2 * V', 'dense_max_nodes': dense_max_nodes,
                         'dense_preference': 'supported revision, then challenged revision or conflicting evidence; density and node-overlap diversity within each tier',
                         'dense_growth': 'alternate explicit-relation context seeds with random-node seeds; density threshold is never relaxed',
                         'sparse_max_nodes': None,
                         'sparse_growth': 'extend endpoints until no chordless extension remains; no node-count cap',
                         'edge_count': 'distinct directed (source, relation, target) triples',
                         'path_semantics': 'undirected simple node paths in the induced subgraph; endpoint path passes through goal',
                         'search': 'bounded randomized search; fewer results does not prove no other candidates exist'},
            'dense': dense_samples(g, dense_count, dense_min_nodes, dense_max_nodes, dense_trials, seed) if 'dense' in methods else [],
            'sparse': sparse_samples(g, count=sparse_count, trials=sparse_trials, seed=seed) if 'sparse' in methods else [],
            'same_type': by_type if 'same_type' in methods else []}
