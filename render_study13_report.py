"""Render the Study 13 report using only independently replayable public records."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from string import Template

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / 'study13_report.md.in'
REPORT = ROOT / 'RT_VISIBILITY_REPORT.md'
COST = ROOT / 'results/study13-cost-projection'
GEOMETRY = ROOT / 'results/study13-geometry-projection'
DISPATCH = ROOT / 'results/study13-matched-dispatch'
COST_PIN = '693aba784a51af0f7c5705045c3627d948020245b172f835f440269de1c587c7'
GEOMETRY_PIN = '889950bd9508b6d5a6d46cbb702fd1499534aa84505e6e27e09145ddfe8949e4'
DISPATCH_PINS = {
    'spread': '2c095e0800fe804cd93ff2320c9be925ec10d99164495f115674cef88f05c023',
    'clustered': '428724f978eff7ce8a90771f12004d10d0c9b08b20ef70d9a54b6db253006746',
    'boundary': '7e64f84c27da8f21caa3fcc657e164def39a13066a51afdecd8edfe0032e10f8',
}
ORDERS = ('ABDC', 'BCAD', 'CDBA', 'DACB')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned_json(path, pin):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == pin, f'pin mismatch: {path.name}')
    return json.loads(raw)


def table(rows):
    return '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def ms(ns):
    return f'{ns / 1e6:.3f}'


def us(ns):
    return f'{ns / 1e3:.3f}'


def span(samples, units=ms):
    return f'{units(median(samples))} [{units(min(samples))}, {units(max(samples))}]'


def cost_tables(cost):
    pairs = cost['pairs']
    need([p['pair'] for p in pairs] == [f'p{i:02}' for i in range(1, 9)], 'pair census differs')
    groups = Counter((p['scene'], p['order']) for p in pairs)
    need(groups == Counter({(s, o): 2 for s in ('spread', 'clustered')
                            for o in ('cuda-first', 'vulkan-first')}), 'pair balance differs')
    rows, budgets, buffers = [], [], []
    ledger = {s: [] for s in ('spread', 'clustered')}
    ledger_manifest = pinned_json(COST / 'evidence/manifest.json', cost['evidence_manifest_sha256'])
    for pair in pairs:
        routes = {r['route']: r for r in pair['routes']}
        need(len(pair['routes']) == 2 and set(routes) == {'cuda', 'vulkan'}, 'wrong pair routes')
        cu, vk = (routes[k]['child_wall_ns'] for k in ('cuda', 'vulkan'))
        rows.append((pair['pair'], pair['scene'], pair['order'], ms(cu), ms(vk), ms(vk - cu)))
        name = pair['pair'] + '.json'
        record = pinned_json(COST / 'evidence' / name, ledger_manifest['files'][name]['public_sha256'])['record']
        native = record['launches']
        cc = native['cuda']['record']['stdout']['record']['cost']
        vc = native['vulkan']['record']['cost']['record']
        ledger[pair['scene']].append((routes, cc, vc))
    for scene, entries in ledger.items():
        def sample(f):
            return [f(*entry) for entry in entries]
        def invariant(f):
            values = sample(f)
            need(len(set(values)) == 1, f'inconsistent byte geometry for {scene}')
            return f'{values[0]:,}'
        stages = [
            ('CUDA CPU BVH build', lambda r, c, v: c['scene_build_ns']),
            ('CUDA scene upload', lambda r, c, v: c['scene_upload_ns']),
            ('CUDA first batch service', lambda r, c, v: c['batches'][0]['service_ns']),
            ('CUDA reused batch service', lambda r, c, v: c['batches'][1]['service_ns']),
            ('Vulkan initialization', lambda r, c, v: v['host_stages_ns']['target_init']),
            ('Vulkan first batch stage sum', lambda r, c, v: v['batches'][0]['measured_stage_sum_ns']),
            ('Vulkan reused batch stage sum', lambda r, c, v: v['batches'][1]['measured_stage_sum_ns']),
            ('CUDA unattributed wall', lambda r, c, v: r['cuda']['unattributed_child_wall_ns']),
            ('Vulkan unattributed wall', lambda r, c, v: r['vulkan']['unattributed_child_wall_ns']),
        ]
        budgets += [(scene, label, span(sample(f))) for label, f in stages]
        b = [lambda r, c, v: c['owned_device_bytes'], lambda r, c, v: c['scene_h2d_bytes'],
             lambda r, c, v: c['batches'][0]['rays_h2d_bytes'] + c['batches'][0]['poison_h2d_bytes'],
             lambda r, c, v: c['batches'][0]['d2h_bytes'],
             lambda r, c, v: v['vulkan_owned_buffer_totals']['requested_bytes'],
             lambda r, c, v: v['vulkan_owned_buffer_totals']['allocation_bytes'],
             lambda r, c, v: v['mapped_io_bytes']['scene_vertices_once'] + v['mapped_io_bytes']['scene_instance_once'],
             lambda r, c, v: v['mapped_io_bytes']['rays_per_batch'] + v['mapped_io_bytes']['canary_per_batch'],
             lambda r, c, v: v['mapped_io_bytes']['readback_per_batch']]
        buffers.append((scene, *(invariant(f) for f in b)))
    return table(rows), table(budgets), table(buffers)


def render():
    cost = pinned_json(COST / 'replay-receipt.json', COST_PIN)
    geometry = pinned_json(GEOMETRY / 'receipt.json', GEOMETRY_PIN)
    need(cost['comparison_eligible'] is geometry['comparison_eligible'] is False, 'eligibility changed')
    dispatch = []
    for scene, pin in DISPATCH_PINS.items():
        x = pinned_json(DISPATCH / f'{scene}.replay.json', pin)
        c, v = (x['readbacks'][r] for r in ('cuda', 'vulkan'))
        dispatch.append((scene, f"{x['triangles']:,}", f"{x['active_total']:,}", f"{c['words_checked']:,}",
                         c['mismatches'], v['mismatches']))
    cost_rows, stage_rows, byte_rows = cost_tables(cost)
    geo = geometry['geometry']['cases']
    need(set(geo) == {'same-topology', 'different-topology'}, 'geometry census differs')
    geo_rows = []
    for name in ('same-topology', 'different-topology'):
        x = geo[name]
        c, v = (x['routes'][r] for r in ('cuda', 'vulkan'))
        geo_rows.append((name, ' → '.join(map(str, x['primitive_counts'])), c['operations']['after'],
                         v['operations']['after'], len(x['changed_lanes']), c['words_checked'] + v['words_checked'],
                         ms(c['recorded_wall_ns']), ms(v['recorded_wall_ns'])))
    q = geometry['quartet']
    need(tuple(x['order'] for x in q['quartets']) == ORDERS
         and set(q['specimens']) == {f'{o}-{a}' for o in ORDERS for a in o}, 'quartet census differs')
    deltas = []
    for contrast in ('B-A', 'D-C', 'C-A', 'D-B'):
        row = [contrast]
        for field, unit in (('after_edit_ns', us), ('whole_route_ns', ms)):
            values = [x['paired_deltas_ns'][field][contrast] for x in q['quartets']]
            row += [span(values, unit), f'{sum(n > 0 for n in values)}/{len(values)}']
        deltas.append(row)
    return Template(TEMPLATE.read_text(encoding='utf-8')).substitute(
        dispatch=table(dispatch), cost=cost_rows, stages=stage_rows, buffers=byte_rows,
        geometry=table(geo_rows), deltas=table(deltas), cost_words=f"{cost['words_checked']:,}",
        quartet_words=f"{q['active_words_checked'] + q['canaries_checked']:,}",
        quartet_canaries=f"{q['canaries_checked']:,}", cost_pin=COST_PIN, geometry_pin=GEOMETRY_PIN)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    a = p.parse_args()
    try:
        payload = render().encode('utf-8')
        if a.check:
            need(REPORT.read_bytes() == payload, 'generated report differs')
        else:
            REPORT.write_bytes(payload)
        print(f'{REPORT.name}: {len(payload)} bytes')
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
