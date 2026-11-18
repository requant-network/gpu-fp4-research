"""Study 13 public projections: offline exact CPU replay, no producer execution."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median
import stat
from struct import Struct

EPOCHS = ('before', 'after')
CASES = {'same-topology': (('cuda', 'vulkan'), 'refit', (2, 2), 'update_eligible'),
         'different-topology': (('vulkan', 'cuda'), 'rebuild', (2, 3), 'rebuild_required')}
CHANGES = {'same-topology': [[0, 0, 1, 0], [0, 1, 0, 1], [1, 3, 0, 1], [1, 4, 1, 0]],
           'different-topology': [[0, 3, 0, 1], [1, 1, 0, 1]]}
ORDERS = ('ABDC', 'BCAD', 'CDBA', 'DACB')
ARMS = {'A': 'refit', 'B': 'rebuild', 'C': 'same-topology', 'D': 'same-topology-build'}
CONTRASTS = ('B-A', 'D-C', 'C-A', 'D-B')
GEOM_SCOPE = 'perf_counter_ns_before_subprocess_run_through_wait_exit'
WHOLE_SCOPE = 'fresh-child-spawn-through-wait-v1'
CUDA_SCOPE = 'after-transition-to-second-full-readback-v1'
VK_SCOPE = 'after-scene-to-second-full-readback-v1'
COMPLETE = 'readbacks_collected_not_replayed'
ARCHIVE = 'aa2016c57f32be4b2217adda46e677975093a9294f2a9c502e89b0e4ca9518ed'
ACQUISITIONS = dict(zip(CASES, ('100d33dfc1ee233e49041640daba5e7a5f7fc9c5557a6bf232e9dcd0a6c5bfe2',
                               'a24e1b34b6d75e9c07b27c58946836f7238593c2db56bb3955dfdf98e790d3a9')))
MATRIX = '4522e739d2cb1d75091656055f5aace7317e4d502cfd5848fd55fa38ba33cf3d'
PLAN = 'cbf26e496ca5df5ed0ef9e75c57180d844f744af4cf15dcb53122b572c431539'
FIXTURES = {'before': '9d1eec42f294871c2ce822dad8097423cb0a5a6bb025de03c2330da48f1733f7',
            'same-topology': '67261e941704a93a072704604536fe44af58f9a7c9482949f579a8b2d7ccf8fc',
            'different-topology': '11d3c04a977a96a4c6daf594111c0b6610fb9237da55d639b8ddc14a2eacbc50'}
SHADER = '896174d5c320010130ab8f661ab69342dfea4072dcdf3f538c82886e41a7b5f5'
ORACLE = '573de4d69c51402665ed81484ae83c89245ac5628811252c57a6a22a9f26718a'
GEOM_PRODUCERS = {'cuda': 'cfb6ed7340256df6b76da2c5b9bfcec896a1637a970a236feb375d3fc87cc7ca',
                  'vulkan': '56d961792ce211971070ba018953881a0305d552b44562075033b6987e72f74a'}
ARM_PRODUCERS = {arm: ('5a5c6ec108d7cff3ba8cd3ec431ca34b55744dea24d63922d9147b98364eff61' if arm in 'AB'
                      else '157b17d1173e1941fbef607e1b7757ac462d37638dc5ebc47106151a718d47ff') for arm in ARMS}
TOP_NS = ('fixture_preflight_ns', 'host_query_pack_ns', 'device_select_ns', 'query_alloc_ns', 'query_release_ns', 'raw_write_ns')
TRANS_NS = ('host_geometry_ns', 'scene_alloc_ns', 'scene_h2d_ns')
BATCH_NS = ('rays_h2d_ns', 'poison_h2d_ns', 'dispatch_sync_ns', 'readback_d2h_ns')
HEADER, TRI, RAY, RAW = Struct('<8s6IQ'), Struct('<9f'), Struct('<8f'), Struct('<128I')
EVIDENCE = ('geometry.json', 'quartet.json')
DISCLAIMERS = dict(execution_attested=False, physical_identity_attested=False, gpu_evidence=False,
                   comparison_eligible=False, ranking_performed=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA-256')
    return value


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('ascii')


def parse(data):
    def pairs(items):
        obj = dict(items)
        require(len(obj) == len(items), 'duplicate JSON field')
        return obj

    def invalid(_):
        raise ValueError('nonfinite JSON constant')

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def keys(obj, names, label):
    require(type(obj) is dict and obj.keys() == set(names), f'{label}: unexpected field or census')


def match(obj, expected, label, extra=()):
    keys(obj, (*expected, *extra), label)
    for k, v in expected.items():
        require(type(obj[k]) is type(v) and obj[k] == v, f'{label}: {k} mismatch')


def ns(n, positive=False):
    require(type(n) is int and int(positive) <= n < 2**63, 'invalid recorded duration')
    return n


def record(row, label, original=None):
    keys(row, ('original_sha256', 'public_sha256', 'data'), label)
    old = pin(row['original_sha256'])
    require(original is None or old == original, f'{label}: original digest mismatch')
    require(digest(canonical(row['data'])) == pin(row['public_sha256']), f'{label}: public record digest mismatch')
    return row['data']


def payload(row, exact_size=None, expected_pin=None):
    keys(row, ('hex', 'sha256'), 'payload')
    text = row['hex']
    require(type(text) is str and 0 < len(text) <= 16384 and re.fullmatch('(?:[0-9a-f]{2})+', text), 'invalid payload hex')
    data = bytes.fromhex(text)
    require(exact_size is None or len(data) == exact_size, 'wrong payload length')
    actual = digest(data)
    require(actual == pin(row['sha256']) and expected_pin in (None, actual), 'payload SHA-256 mismatch')
    return data


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def intersects(start, end, tri):
    """Open segment vs closed, two-sided, nondegenerate triangle, exactly.

    Use Gram-system barycentric numerators and clip three affine inequalities.
    For coplanar segments keep the entire plane interval before clipping.
    """
    a, b, c = [tuple(map(F, vertex)) for vertex in tri]
    start, end = tuple(map(F, start)), tuple(map(F, end))
    if start == end:
        return False
    u, v = sub(b, a), sub(c, a)
    uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
    det = uu * vv - uv * uv
    if not det:
        return False
    n = tuple(u[(i + 1) % 3] * v[(i + 2) % 3] - u[(i + 2) % 3] * v[(i + 1) % 3] for i in range(3))
    p, q = sub(start, a), sub(end, a)
    d0, d1 = dot(p, n), dot(q, n)
    lo, hi = F(0), F(1)
    if d0 != d1:
        lo = hi = d0 / (d0 - d1)
        if not 0 < lo < 1:
            return False
    elif d0:
        return False

    def bary(w):
        wu, wv = dot(w, u), dot(w, v)
        s, t = vv * wu - uv * wv, uu * wv - uv * wu
        return s, t, det - s - t

    for s, t in zip(bary(p), bary(q), strict=True):
        slope = t - s
        if slope > 0:
            lo = max(lo, -s / slope)
        elif slope < 0:
            hi = min(hi, -s / slope)
        elif s < 0:
            return False
    return lo <= hi and hi > 0 and lo < 1


def decode_fixture(data, ntri):
    end = HEADER.size + ntri * TRI.size
    require(len(data) == end + 128 * RAY.size and HEADER.unpack_from(data) ==
            (b'S13VIS1\0', 0, 0, ntri, 5, 2, 64, 19), 'invalid fixture geometry/header')
    tris = [tuple(tuple(map(F, t[i:i + 3])) for i in (0, 3, 6)) for t in TRI.iter_unpack(data[HEADER.size:end])]
    return tris, data[end:]


def reference(scene, case):
    keys(scene, (*EPOCHS, 'shader'), 'shared input snapshots')
    payload(scene['shader'], expected_pin=SHADER)
    refs, triangles, rays, fps = {}, {}, {}, {}
    for e, n in zip(EPOCHS, CASES[case][2], strict=True):
        fps[e] = FIXTURES['before' if e == 'before' else case]
        triangles[e], rays[e] = decode_fixture(payload(scene[e], expected_pin=fps[e]), n)
        words = []
        for i, r in enumerate(RAY.iter_unpack(rays[e])):
            if i % 64 >= 5:
                words.append(0xfeedface)
                continue
            x, y, z, lo, dx, dy, dz, hi = map(F, r)
            require(lo < hi, 'active ray interval must increase')
            a, b = (tuple(o + t * d for o, d in zip((x, y, z), (dx, dy, dz), strict=True)) for t in (lo, hi))
            words.append(int(any(intersects(a, b, tri) for tri in triangles[e])))
        refs[e] = tuple(words)
    original, edited = (triangles[e] for e in EPOCHS)
    require(rays['before'] == rays['after'], 'rays changed during geometry edit')
    if case == 'same-topology':
        expected = [tuple((x + 16, y, z) for x, y, z in original[0]), *original[1:]]
    else:
        expected = [*original, ((56, 0, 0), (64, 0, 0), (56, 8, 0))]
    require(edited == expected, 'wrong triangle edit')
    changes = [[i // 64, i % 64, a, b] for i, (a, b) in enumerate(zip(*refs.values(), strict=True)) if a != b]
    require(changes == CHANGES[case], 'unexpected exact changed lanes')
    return refs, fps, changes


def check_reads(reads, refs, label):
    keys(reads, EPOCHS, label)
    counts = dict(words_checked=0, active_words_checked=0, canaries_checked=0)
    for e in EPOCHS:
        raw = payload(reads[e], RAW.size)
        for i, (got, want) in enumerate(zip(RAW.unpack(raw), refs[e], strict=True)):
            require(got == want, f'{label}/{e}: exact word mismatch at batch {i // 64} lane {i % 64} (got {got:#010x}, expected {want:#010x})')
            counts['words_checked'] += 1
            counts['active_words_checked' if i % 64 < 5 else 'canaries_checked'] += 1
    return counts


def check_vk(row, pair, fps, reads, counts, update, flag):
    keys(row, EPOCHS, 'Vulkan epoch census')
    for e, n in zip(EPOCHS, counts, strict=True):
        m = record(row[e], f'Vulkan {e} metadata')
        in_place = e == 'after' and update
        match(m, dict(schema='study13-update-host-v1', pair=pair, epoch=e, fixture_sha256=fps[e], shader_sha256=SHADER,
                      blas_mode='UPDATE' if in_place else 'BUILD', blas_flags=flag, blas_in_place=int(in_place),
                      tlas_mode='BUILD', primitives=n, words=128, invalid_active=0, invalid_tail=0,
                      raw_sha256=reads[e]['sha256']), f'Vulkan {e} metadata')
    return {e: row[e]['data']['blas_mode'] for e in EPOCHS}


def stage_sum(row, names):
    return sum(ns(row[k]) for k in names)


def check_cuda(row, fps, counts, action, quartet=False):
    fields = dict(schema='s13_cuda_same_edit_cost_v1' if quartet else 's13_cuda_update_host_v1', active=5, lanes=64,
                  structurally_valid=True, query_alloc_bytes=2304, peak_owned_device_bytes=2304 + 32 + 48 * max(counts))
    if quartet:
        fields['sample_kind'] = 'host_steady_clock'
    match(row, fields, 'CUDA ledger', (*TOP_NS, 'epochs'))
    outside = stage_sum(row, TOP_NS)
    require(type(row['epochs']) is list and len(row['epochs']) == 2, 'CUDA epoch census')
    for e, n, ep in zip(EPOCHS, counts, row['epochs'], strict=True):
        reuse = e == 'after' and action == 'refit'
        size = 32 + 48 * n
        expected = dict(epoch=e, operation='build' if e == 'before' else action, triangles=n, nodes=1, node_bytes=32,
                        primitive_bytes=48 * n, scene_h2d_bytes=size, scene_alloc_bytes=0 if reuse else size)
        stages = TRANS_NS + (('prior_scene_release_ns',) if quartet else ())
        if quartet:
            expected['fixture_sha256'] = fps[e]
        match(ep, expected, f'CUDA {e} ledger', (*stages, 'scene_release_ns', 'batches'))
        elapsed = stage_sum(ep, stages)
        require(not reuse or ep['scene_alloc_ns'] == 0, 'CUDA refit allocation')
        release = ns(ep['scene_release_ns'])
        if quartet:
            require(e == 'after' or release == 0, 'unexpected quartet baseline release')
            require((e == 'after' and action == 'rebuild') or ep['prior_scene_release_ns'] == 0, 'unexpected quartet prior-scene release')
        require(type(ep['batches']) is list and len(ep['batches']) == 2, 'CUDA batch census')
        for j, b in enumerate(ep['batches']):
            match(b, dict(batch=j, rays_h2d_bytes=2048, poison_h2d_bytes=256, readback_d2h_bytes=256), 'CUDA transfer ledger', BATCH_NS)
            elapsed += stage_sum(b, BATCH_NS)
        outside += release
        if e == 'before':
            outside += elapsed
    return elapsed, outside


def geometry(projection):
    match(projection, dict(schema='study13-public-geometry-v1'), 'geometry projection', ('cases',))
    keys(projection['cases'], CASES, 'geometry case census')
    result = {}
    for case, (order, action, counts, required) in CASES.items():
        item = projection['cases'][case]
        keys(item, ('acquisition', 'snapshots', 'routes'), case)
        refs, fps, changes = reference(item['snapshots'], case)
        acq = record(item['acquisition'], case, ACQUISITIONS[case])
        match(acq, dict(case=case, order=list(order), scope=GEOM_SCOPE, status='collected', mode='native',
                        inputs={**fps, 'shader': SHADER, 'oracle': ORACLE}, producers=GEOM_PRODUCERS,
                        gpu_evidence=False, comparison_eligible=False), case, ('launch_sha256',))
        keys(acq['launch_sha256'], order, 'geometry launch pins')
        keys(item['routes'], order, 'geometry route census')
        routes = {}
        for route in order:
            label = f'{case}/{route}'
            row = item['routes'][route]
            keys(row, ('launch', 'readbacks', 'cost' if route == 'cuda' else 'metadata', 'completion') if route == 'vulkan'
                 else ('launch', 'readbacks', 'cost'), label)
            launch = record(row['launch'], label, pin(acq['launch_sha256'][route]))
            reads = row['readbacks']
            check = check_reads(reads, refs, label)
            inputs = {**acq['inputs'], route: GEOM_PRODUCERS[route]}
            bindings = {e + '.raw': reads[e]['sha256'] for e in EPOCHS}
            artifact = 'cost' if route == 'cuda' else 'completion'
            bindings['stdout.log'] = row[artifact]['original_sha256']
            bindings['stderr.log'] = digest(b'')
            if route == 'vulkan':
                keys(row['metadata'], EPOCHS, label)
                bindings.update({e + '.meta': row['metadata'][e]['original_sha256'] for e in EPOCHS})
            match(launch, dict(route=route, case=case, exit_code=0, input_sha256_before=inputs, input_sha256_after=inputs,
                               artifacts=bindings, stderr_empty=True), label, ('wall_ns',))
            wall = ns(launch['wall_ns'], True)
            if route == 'cuda':
                end, outside = check_cuda(record(row['cost'], label), fps, counts, action)
                require(wall >= end + outside, 'geometry CUDA stages exceed wall')
                ops = {'before': 'build', 'after': action}
                check.update(accounted_ns=end + outside, unattributed_ns=wall - end - outside)
            else:
                match(record(row['completion'], label), {'status': COMPLETE}, label)
                ops = check_vk(row['metadata'], case, fps, reads, counts, case == 'same-topology', int(case == 'same-topology'))
            routes[route] = dict(**check, operations=ops, exit_code=0, recorded_wall_ns=wall,
                                 raw_sha256={e: reads[e]['sha256'] for e in EPOCHS})
        result[case] = dict(order=list(order), required_action=required, primitive_counts=list(counts), changed_lanes=changes,
                            fixture_sha256=fps, routes=routes)
    return result


def quartet(projection):
    match(projection, dict(schema='study13-public-quartet-v1'), 'quartet projection', ('matrix', 'plan', 'snapshots', 'specimens'))
    matrix = record(projection['matrix'], 'matrix', MATRIX)
    plan = record(projection['plan'], 'plan', PLAN)
    refs, fps, changes = reference(projection['snapshots'], 'same-topology')
    groups = [{'order': order, 'specimens': [f'{order}-{arm}' for arm in order]} for order in ORDERS]
    match(plan, dict(orders=list(ORDERS), routes=ARMS, producers=ARM_PRODUCERS, fixtures={**fps, 'shader': SHADER}), 'declared plan')
    match(matrix, dict(mode='host-records', producers=ARM_PRODUCERS, quartets=groups), 'matrix', ('specimen_sha256',))
    ids = [name for g in groups for name in g['specimens']]
    keys(projection['specimens'], ids, 'quartet specimen census')
    keys(matrix['specimen_sha256'], ids, 'quartet specimen pins')
    samples = {}
    for ident in ids:
        arm = ident[-1]
        route, cuda = ARMS[arm], arm in 'AB'
        row = projection['specimens'][ident]
        keys(row, ('observation', 'wait', 'timing', 'readbacks', 'cost', 'stdout_matches_cost') if cuda else
             ('observation', 'wait', 'timing', 'readbacks', 'metadata', 'completion'), ident)
        reads = row['readbacks']
        check = check_reads(reads, refs, ident)
        bindings = {e + '.raw': reads[e]['sha256'] for e in EPOCHS}
        bindings.update({'timing.meta': row['timing']['original_sha256'], 'stderr.log': digest(b'')})
        if cuda:
            cost_pin = row['cost']['original_sha256']
            require(row['stdout_matches_cost'] is True, 'CUDA stdout/ledger mismatch')
            bindings.update({'cost.json': cost_pin, 'stdout.json': cost_pin})
        else:
            keys(row['metadata'], EPOCHS, ident)
            bindings.update({e + '.meta': row['metadata'][e]['original_sha256'] for e in EPOCHS})
            bindings['stdout.log'] = row['completion']['original_sha256']
            match(record(row['completion'], ident), {'status': COMPLETE}, ident)
        obs = record(row['observation'], ident, pin(matrix['specimen_sha256'][ident]))
        match(obs, dict(specimen=ident, route=route, scope=WHOLE_SCOPE, wait_exit=0,
                        binary_sha256=ARM_PRODUCERS[arm], artifacts=bindings, stderr_empty=True), ident, ('whole_route_ns',))
        wall = ns(obs['whole_route_ns'], True)
        match(record(row['wait'], ident), dict(wait_exit=0, whole_route_ns=wall), ident + '/wait')
        timing = record(row['timing'], ident)
        scope = CUDA_SCOPE if cuda else VK_SCOPE
        fields = dict(route=route, scope=scope, sample_kind='host_steady_clock')
        if cuda:
            after_stage, outside = check_cuda(record(row['cost'], ident), fps, (2, 2), route, True)
            fields.update(stage_sum_ns=after_stage, cost_sha256=cost_pin, **{e + '_raw_sha256': reads[e]['sha256'] for e in EPOCHS})
            match(timing, fields, ident + '/timing', ('after_ns', 'unattributed_ns'))
            after = ns(timing['after_ns'], True)
            remainder = ns(timing['unattributed_ns'])
            require(after_stage + remainder == after and after + outside <= wall, f'{ident}: after-stage/whole-route timing enclosure')
            check.update(stage_sum_ns=after_stage, after_unattributed_ns=remainder, outside_after_ledger_ns=outside)
            ops = dict(before='build', after=route)
        else:
            ops = check_vk(row['metadata'], route, fps, reads, (2, 2), arm == 'C', 1)
            fields.update({e + '_meta_sha256': row['metadata'][e]['original_sha256'] for e in EPOCHS})
            match(timing, fields, ident + '/timing', ('after_ns',))
            after = ns(timing['after_ns'], True)
            require(after <= wall, f'{ident}: after-stage/whole-route timing enclosure')
        samples[ident] = dict(**check, arm=arm, route=route, operations=ops, wait_exit=0, after_edit_ns=after,
                              after_scope=scope, whole_route_ns=wall, raw_sha256={e: reads[e]['sha256'] for e in EPOCHS})
    metrics = ('after_edit_ns', 'whole_route_ns')
    for group in groups:
        order = group['order']
        group['paired_deltas_ns'] = {k: {c: samples[f'{order}-{c[0]}'][k] - samples[f'{order}-{c[2]}'][k]
                                         for c in CONTRASTS} for k in metrics}
    summary = {}
    for k in metrics:
        summary[k] = {}
        for c in CONTRASTS:
            vals = [g['paired_deltas_ns'][k][c] for g in groups]
            summary[k][c] = dict(n=len(vals), min_ns=min(vals), median_ns=median(vals), max_ns=max(vals))
    return dict(changed_lanes=changes, fixture_sha256=fps, recorded_quartets=len(groups), specimens=samples,
                quartets=groups, recorded_paired_summary_ns=summary, whole_route_scope=WHOLE_SCOPE)


def regular(path, directory=False):
    s = path.lstat()
    reparse = getattr(s, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)
    require(not reparse and (stat.S_ISDIR(s.st_mode) if directory else stat.S_ISREG(s.st_mode) and s.st_nlink == 1),
            f'{path.name}: expected unlinked regular {"directory" if directory else "file"}')


def read(path, limit=1 << 20):
    regular(path)
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, f'{path.name}: oversized file')
    return data


def replay(root, manifest_pin):
    regular(root, directory=True)
    manifest_bytes = read(root / 'public-manifest.json', 4096)
    require(digest(manifest_bytes) == pin(manifest_pin), 'manifest digest mismatch (external pin required)')
    m = parse(manifest_bytes)
    match(m, dict(schema='study13-public-manifest-v1', archive_sha256=ARCHIVE), 'manifest', ('evidence', 'replay_sha256'))
    require(digest(read(Path(__file__))) == pin(m['replay_sha256']), 'replay source digest mismatch')
    keys(m['evidence'], EVIDENCE, 'manifest evidence census')
    folder = root / 'evidence'
    regular(folder, directory=True)
    require({p.name for p in folder.iterdir()} == set(EVIDENCE), 'unexpected evidence directory census')
    objects = []
    for name in EVIDENCE:
        data = read(folder / name)
        require(digest(data) == pin(m['evidence'][name]), f'{name}: evidence digest mismatch')
        objects.append(parse(data))
    g, q = geometry(objects[0]), quartet(objects[1])
    counts = ('words_checked', 'active_words_checked', 'canaries_checked')
    g_rows = [row for case in g.values() for row in case['routes'].values()]
    g_totals = {k: sum(row[k] for row in g_rows) for k in counts}
    q_totals = {k: sum(row[k] for row in q['specimens'].values()) for k in counts}
    return dict(schema='study13-public-exact-replay-v1', status='pass', manifest_sha256=manifest_pin, **DISCLAIMERS,
                geometry=dict(cases=g, scope=GEOM_SCOPE, **g_totals), quartet=dict(**q, **q_totals),
                **{k: g_totals[k] + q_totals[k] for k in counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--check', action='store_true', help='compare deterministic receipt.json; never rewrite it')
    args = parser.parse_args()
    try:
        result = replay(args.root, args.manifest_sha256)
        if args.check:
            require(canonical(result) == read(args.root / 'receipt.json'), 'receipt bytes differ')
            print(f'PASS: {result["words_checked"]} words, {result["canaries_checked"]} canaries; receipt matches')
        else:
            print(canonical(result).decode('ascii'), end='')
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
