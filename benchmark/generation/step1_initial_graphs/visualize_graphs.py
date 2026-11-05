#!/usr/bin/env python3
"""Render graph JSON files as SVG/PNG figures; the PNG is shown to the model during screening."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

from graph_io import validate

ABBR = {'goal': 'G', 'proposal': 'P', 'question': 'Q', 'hypothesis': 'H', 'plan': 'PL',
        'activity': 'A', 'evidence': 'E', 'decision': 'D', 'artifact': 'AR'}
COLORS = dict(zip(ABBR, ['#d4e5fb', '#dfd6fb', '#fff0bb', '#fde0e6', '#dce3ec', '#c9eee8', '#dceecb', '#ffddbd', '#e2def7']))
EDGE_COLORS = {'address': '#637b90', 'produce': '#00887d', 'depend_on': '#8960aa',
               'evaluate': '#2368b0', 'support': '#42852c', 'refute': '#bb4056', 'decompose': '#806539', 'revise': '#c07a19', 'revive': '#aa478f'}
matplotlib.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})


def layout(graph):
    """Place each subtree in a separate interval, preserving sibling order."""
    nodes = graph['nodes']
    children = {n['id']: [] for n in nodes}
    depths = {nodes[0]['id']: 0}
    for n in nodes[1:]:
        parent = n['exploration_parent_id']
        children[parent].append(n['id'])
        depths[n['id']] = depths[parent] + 1
    x = {}
    leaf = 0
    for n in nodes:
        if not children[n['id']]:
            x[n['id']] = leaf * 1.8
            leaf += 1
    for n in reversed(nodes):
        cs = children[n['id']]
        if cs:
            x[n['id']] = (x[cs[0]] + x[cs[-1]]) / 2
    return {nid: (value, -depths[nid] * 1.8) for nid, value in x.items()}


def render(graph, stem, out, tight=False):
    """Write <out>/<stem>.svg and <out>/<stem>.png; return the graph statistics.

    tight=True crops the figure to its content (used for evolved graphs in step 2).
    """
    stats = validate(graph)
    positions = layout(graph)
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#ffffff')
    ax.set_aspect('equal')
    ax.axis('off')
    radius = .24
    extents = list(positions.values())
    tree_pairs = set()
    # exploration tree: dashed gray lines
    for n in graph['nodes'][1:]:
        s, t = n['exploration_parent_id'], n['id']
        tree_pairs.add(frozenset((s, t)))
        ax.add_patch(FancyArrowPatch(positions[s], positions[t], arrowstyle='-', linestyle=(0, (4, 4)),
                                     color='#bcc6d1', linewidth=1.0, shrinkA=0, shrinkB=0, zorder=1))
    occupied = [(x - .32, y - .32, x + .32, y + .32) for x, y in positions.values()]

    def place_label(start, end, rad, relation):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(math.hypot(dx, dy), .001)
        half_width, half_height = len(relation) * .031 + .06, .095
        candidates = []
        for t in (.5, .4, .6, .3, .7, .25, .75):
            px = start[0] + t * dx + 2 * t * (1 - t) * rad * dy
            py = start[1] + t * dy - 2 * t * (1 - t) * rad * dx
            for offset in (0, .16, -.16, .32, -.32):
                x, y = px + offset * dy / length, py - offset * dx / length
                box = (x - half_width, y - half_height, x + half_width, y + half_height)
                overlaps = sum(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in occupied)
                candidates.append((overlaps * 100 + abs(t - .5) + abs(offset), x, y, box))
        _, x, y, box = min(candidates, key=lambda c: c[0])
        occupied.append(box)
        extents.extend(((box[0], box[1]), (box[2], box[3])))
        return x, y

    # semantic relations: labeled colored arrows
    pairs = {}
    for e in graph['edges']:
        s, t, relation = e['source'], e['target'], e['relation']
        start, end = positions[s], positions[t]
        pair = frozenset((s, t))
        ordinal = pairs.get(pair, 0)
        pairs[pair] = ordinal + 1
        rad = (.20 if pair in tree_pairs else .14) + .12 * ordinal
        color = EDGE_COLORS[relation]
        source_circle = Circle(start, radius, transform=ax.transData)
        target_circle = Circle(end, radius, transform=ax.transData)
        ax.add_patch(FancyArrowPatch(start, end, patchA=source_circle, patchB=target_circle,
                                     arrowstyle='-|>', mutation_scale=9, connectionstyle=f'arc3,rad={rad}',
                                     linewidth=1.05, color=color, shrinkA=2, shrinkB=2, zorder=2))
        dx, dy = end[0] - start[0], end[1] - start[1]
        midpoint = place_label(start, end, rad, relation)
        extents.append(((start[0] + end[0]) / 2 + rad * dy,
                        (start[1] + end[1]) / 2 - rad * dx))
        ax.text(*midpoint, relation, color=color, fontsize=6.5, ha='center', va='center',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1, 'alpha': .92}, zorder=4)
    # nodes: type abbreviation and order
    for n in graph['nodes']:
        x, y = positions[n['id']]
        ax.add_patch(Circle((x, y), radius, facecolor=COLORS[n['type']], edgecolor='#344760', linewidth=1.15, zorder=5))
        ax.text(x, y, f'{ABBR[n["type"]]}\n#{n["order"]}', ha='center', va='center',
                fontsize=7.3, linespacing=1.25, color='#20304a', zorder=6)
    min_x, max_x = min(p[0] for p in extents) - .7, max(p[0] for p in extents) + .7
    min_y, max_y = min(p[1] for p in extents) - .6, max(p[1] for p in extents) + .7
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    width, height = max_x - min_x, max_y - min_y
    fig.set_size_inches(max(6, width * .8), max(5, height * .8))
    fig.subplots_adjust(left=.01, bottom=.01, right=.99, top=.97)
    ax.set_title(f'{stem}   |   {stats["nodes"]} nodes · {stats["proposals"]} proposals · depth {stats["max_depth"]}', fontsize=12, pad=14)
    crop = {'bbox_inches': 'tight', 'pad_inches': 0.2} if tight else {}
    fig.savefig(out / f'{stem}.svg', format='svg', **crop)
    fig.savefig(out / f'{stem}.png', dpi=min(150, 3600 / max(fig.get_size_inches())), **crop)
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory with graph_*.json files')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(p for p in args.input.glob('graph_*.json') if not p.name.endswith('.input.json'))
    if not paths:
        parser.error(f'No graph_*.json files in {args.input}')
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for path in paths:
        stats = render(json.loads(path.read_text(encoding='utf-8')), path.stem, args.output)
        report.append({'file': path.name, **stats})
        print(f'Rendered {path.stem}: {stats}', flush=True)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
