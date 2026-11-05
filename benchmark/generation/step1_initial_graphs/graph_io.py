"""Structural checks shared by generation and visualization (standard library only)."""
from collections import Counter

TYPES = ('goal', 'proposal', 'question', 'hypothesis', 'plan', 'activity', 'evidence', 'decision', 'artifact')
ENDPOINTS = {
    'address': ({'proposal', 'hypothesis', 'plan', 'activity'}, {'question', 'goal'}),
    'produce': ({'activity'}, {'evidence', 'artifact'}),
    'evaluate': ({'activity'}, {'hypothesis', 'proposal', 'artifact'}),
    'support': ({'evidence'}, {'hypothesis', 'proposal', 'decision'}),
    'refute': ({'evidence'}, {'hypothesis', 'proposal'}),
    'depend_on': (set(TYPES), set(TYPES)),
    'revise': (set(TYPES), set(TYPES)),
    'revive': ({'evidence', 'decision'}, {'proposal', 'hypothesis'}),
}
RELATIONS = (*ENDPOINTS, 'decompose')


def validate(graph, seed_input=None):
    """Raise ValueError for malformed graphs; return measured shape statistics.

    Checks structure, not whether an empty skeleton will admit sensible content.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(graph, dict) and set(graph) == {'nodes', 'edges'}, 'Expected only nodes and edges')
    nodes, edges = graph['nodes'], graph['edges']
    require(isinstance(nodes, list) and len(nodes) > 1, 'nodes must be a nonempty array')
    require(isinstance(edges, list), 'edges must be an array')
    by_id = {}
    for i, node in enumerate(nodes, 1):
        require(isinstance(node, dict) and set(node) == {'id', 'type', 'order', 'exploration_parent_id'}, f'Invalid fields at node {i}')
        nid = node['id']
        require(isinstance(nid, str) and bool(nid) and nid not in by_id, f'Invalid/duplicate ID at node {i}')
        require(node['type'] in TYPES, f'Invalid node type: {nid}')
        require(type(node['order']) is int and node['order'] == i, f'Invalid order: {nid}')
        parent = node['exploration_parent_id']
        require(parent is None or isinstance(parent, str), f'Invalid parent: {nid}')
        by_id[nid] = node
    counts = Counter(n['type'] for n in nodes)
    require(counts['goal'] == 1 and 3 <= counts['proposal'] <= 5, 'Expected 1 goal and 3–5 proposals')
    require(nodes[0]['id'] == 'g1' and nodes[0]['type'] == 'goal' and nodes[0]['exploration_parent_id'] is None, 'Invalid goal seed')
    require(nodes[1]['id'] == 'p1' and nodes[1]['type'] == 'proposal' and nodes[1]['exploration_parent_id'] == 'g1', 'Invalid proposal seed')
    proposals = [n for n in nodes if n['type'] == 'proposal']
    expected_ids = [f'p{i}' for i in range(1, counts['proposal'] + 1)]
    if seed_input is not None:
        require(seed_input['goal'] == {'id': 'g1', 'type': 'goal'}, 'Invalid input goal')
        expected_ids = [n['id'] for n in seed_input['proposals']]
    require([n['id'] for n in proposals] == expected_ids, 'Input proposal IDs/count/order must be preserved')
    require(all(n['exploration_parent_id'] == 'g1' for n in proposals), 'All input proposals must be direct children of goal')
    children = {nid: [] for nid in by_id}
    depths = {'g1': 0}
    for node in nodes[1:]:
        parent = node['exploration_parent_id']
        require(parent in depths, f'Parent must exist before child: {node["id"]}')
        children[parent].append(node['id'])
        depths[node['id']] = depths[parent] + 1
    traversal, stack = [], ['g1']
    while stack:
        nid = stack.pop()
        traversal.append(nid)
        stack.extend(reversed(children[nid]))
    require(traversal == [n['id'] for n in nodes], 'Node order is not depth-first preorder')
    seen = set()
    prerequisites = {nid: [] for nid in by_id}
    decomposed = {nid: [] for nid in by_id}
    for edge in edges:
        require(isinstance(edge, dict) and set(edge) == {'source', 'relation', 'target'}, 'Invalid edge fields')
        s, r, t = edge['source'], edge['relation'], edge['target']
        require(all(isinstance(x, str) for x in (s, r, t)), 'Edge fields must be strings')
        require(s in by_id and t in by_id and s != t, f'Invalid edge endpoints: {edge}')
        require((s, r, t) not in seen, f'Duplicate edge: {edge}')
        seen.add((s, r, t))
        st, tt = by_id[s]['type'], by_id[t]['type']
        if r == 'decompose':
            require((st, tt) in {('goal', 'proposal'), ('proposal', 'proposal'), ('plan', 'activity')}, f'Invalid decompose direction: {edge}')
            decomposed[s].append(t)
        else:
            require(r in ENDPOINTS, f'Unsupported relation: {r}')
            sources, targets = ENDPOINTS[r]
            require(st in sources and tt in targets, f'Invalid relation direction: {edge}')
        if r in ('revise', 'revive'):
            require(by_id[s]['order'] > by_id[t]['order'], f'{r} must point to an earlier node: {edge}')
        if r == 'depend_on':
            prerequisites[s].append(t)
    # Kahn's algorithm detects cycles without recursion limits.
    indegree = Counter(t for targets in prerequisites.values() for t in targets)
    ready = [nid for nid in by_id if not indegree[nid]]
    visited = 0
    while ready:
        nid = ready.pop()
        visited += 1
        for target in prerequisites[nid]:
            indegree[target] -= 1
            if not indegree[target]:
                ready.append(target)
    require(visited == len(nodes), 'Circular prerequisite chain')
    reached, stack = set(), ['g1']
    while stack:
        nid = stack.pop()
        if nid not in reached:
            reached.add(nid)
            stack.extend(decomposed[nid])
    require(all(n['id'] in reached for n in nodes if n['type'] == 'proposal'), 'Proposals must be reachable from goal through decompose')
    require(all(('g1', 'decompose', pid) in seen for pid in expected_ids), 'Each input proposal needs goal -> proposal decompose')
    owners = {'g1': None}
    for node in nodes[1:]:
        owners[node['id']] = node['id'] if node['type'] == 'proposal' else owners[node['exploration_parent_id']]
    cross_edges = [e for e in edges if owners[e['source']] is not None and owners[e['target']] is not None and owners[e['source']] != owners[e['target']]]
    cross_pairs = {tuple(sorted((owners[e['source']], owners[e['target']]))) for e in cross_edges}
    return {
        'cross_proposal_edges': len(cross_edges), 'connected_proposal_pairs': len(cross_pairs),
        'nodes': len(nodes), 'edges': len(edges), 'proposals': counts['proposal'],
        'max_depth': max(depths.values()), 'leaves': sum(not c for c in children.values()),
        'branching_nodes': sum(len(c) > 1 for c in children.values()),
        'max_children': max(map(len, children.values())), 'node_types': dict(counts),
    }
