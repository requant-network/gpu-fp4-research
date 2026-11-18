"""Offline CPU replay of the Study 13 public matched-cost projection."""
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import sys

ROUTES = ('cuda', 'vulkan')
ROWS = (
    ('p01', 'spread', 'cuda-first'), ('p02', 'clustered', 'vulkan-first'),
    ('p03', 'spread', 'vulkan-first'), ('p04', 'clustered', 'cuda-first'),
    ('p05', 'clustered', 'cuda-first'), ('p06', 'spread', 'vulkan-first'),
    ('p07', 'clustered', 'vulkan-first'), ('p08', 'spread', 'cuda-first'),
)
BOUNDARY = 'perf_counter_ns_before_subprocess_run_through_wait_exit'
ARCHIVE = 'aa2016c57f32be4b2217adda46e677975093a9294f2a9c502e89b0e4ca9518ed'
PLAN = '9cb676b3076953eb8ba345fa50d778a43a1dffc4cad4d449b3c822fec2a6f184'
LEDGER = '80ed525dcf06e68d78ea956370ad959b48c6361a9329caab027875d2562646a2'
BUILD = '32a445c1c0ee8b400d7514e61866341c436e619ca8a04f87e481fe9669633971'
ORACLE = '573de4d69c51402665ed81484ae83c89245ac5628811252c57a6a22a9f26718a'
WIRE = '76d968b50a7bf46c818f24ab10d457c9d80ffab9a0eb9ab54d4c72b39879e279'
VERIFY = '6f6357336009543a9a902ae280cff603d0057234440b5387235dd32b874ac026'
COST_SOURCE = '163ac620a597795f3cf19b395c07b29b546b1e0ec756f74c19549cddb65fab2b'
PAYLOAD_PINS = {
    'spread': 'de79db9859d146d1ae31aae0e56b487fa3d5bf5ff520a0fd5232e3cd3baa2be9',
    'clustered': 'f65559049ec415f5220d622bb02961e6be418f88a1239a72972be35898cc7407',
    'shader': '896174d5c320010130ab8f661ab69342dfea4072dcdf3f538c82886e41a7b5f5',
}
PRODUCERS = {
    'cuda': {'id': 'cuda_visibility_cost', 'sha256': '2ed544fcabd2e45e343b177f71ac91722599898a8b501de30905f757b1e6f7af'},
    'vulkan': {'id': 'vk_visibility_cost', 'sha256': '2ac008b42b5124fbcfddee6d606fd5e1af4a13457ac66dbcbe52d0bd13711a6c'},
}
SOURCES = {'oracle': ORACLE, 'wire': WIRE, 'verify': VERIFY, 'cost_validator': COST_SOURCE}
FIXTURES = {k: f'{k}.s13.b64' for k in ('spread', 'clustered')}
SHADER = 'shader.spv.b64'
FILES = {'schedule.json', 'builds.json', 'collection.json', SHADER, *FIXTURES.values()} | {
    f'{p}.json' for p, _, _ in ROWS} | {f'{p}.{r}.raw.b64' for p, _, _ in ROWS for r in ROUTES}
SCHEMA = 'study13-public-matched-cost-v1'
C_SETUP = 'fixture_read_ns scene_build_ns device_select_ns host_buffers_ns device_alloc_ns scene_upload_ns device_release_ns raw_write_ns'
C_BATCH = 'pack_ns upload_ns dispatch_ns return_ns'
C_BYTES = 'node_bytes primitive_bytes scene_h2d_bytes owned_device_bytes'
C_IO = 'rays_h2d_bytes poison_h2d_bytes d2h_bytes'
V_SETUP = 'input_preflight target_init setup_excluding_scene_upload vertex_upload instance_upload collect_overhead teardown raw_write'
V_BATCH = 'ray_write_ns canary_write_ns record_ns submit_wait_reset_ns readback_ns'
V_PAYLOAD = 'shader fixture_vertices fixture_rays instance_record canary_staging retained_words raw_serialization'
V_IO = 'scene_vertices_once scene_instance_once rays_per_batch canary_per_batch readback_per_batch'
V_BUFFERS = 'vertices instances scratch blas tlas rays output'
C_KEYS = 'schema mode structurally_valid triangles active lanes batches ' + C_SETUP + ' ' + C_BYTES
V_KEYS = 'schema timer fixture host_stages_ns host_stage_total_ns logical_host_payload_bytes mapped_io_bytes vulkan_owned_buffers vulkan_owned_buffer_totals batches'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid digest')
    return value


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def parse(blob):
    def distinct(pairs):
        row = dict(pairs)
        need(len(row) == len(pairs), 'duplicate JSON key')
        return row

    def reject_constant(value):
        raise ValueError('nonfinite JSON value')

    return json.loads(blob, object_pairs_hook=distinct, parse_constant=reject_constant)


def obj(value, fields, label):
    names = set(fields.split())
    need(type(value) is dict and set(value) == names, f'{label}: unexpected or missing fields')
    return value


def uint(value):
    need(type(value) is int and 0 <= value < 2**64, 'invalid nonnegative integer')
    return value


def numbers(row, keys):
    return [uint(row[k]) for k in keys.split()]


def equals(value, expected, label):
    need(canonical(value) == canonical(expected), f'{label}: value mismatch')


def regular(path, directory=False):
    s = path.lstat()
    flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)
    mode = stat.S_ISDIR if directory else stat.S_ISREG
    need(not getattr(s, 'st_file_attributes', 0) & flag and mode(s.st_mode)
         and (directory or s.st_nlink == 1), 'public evidence must be regular files, not aliases')


def encode(blob):
    text = base64.b64encode(blob)
    return b'\n'.join(text[i:i+76] for i in range(0, len(text), 76)) + b'\n'


def unpack(blob):
    raw = base64.b64decode(blob.replace(b'\n', b''), validate=True)
    need(encode(raw) == blob, 'noncanonical text payload')
    return raw


def bind_record(value, label):
    obj(value, 'original_sha256 public_sha256 record', label)
    pin(value['original_sha256'])
    need(digest(canonical(value['record'])) == pin(value['public_sha256']), f'{label}: public record digest mismatch')
    return value['record']


def load_bundle(bundle, external_pin):
    regular(bundle, directory=True)
    need({p.name for p in bundle.iterdir()} == FILES | {'manifest.json'}, 'public file census mismatch')
    for name in FILES | {'manifest.json'}:
        regular(bundle / name)
    raw = (bundle / 'manifest.json').read_bytes()
    need(digest(raw) == pin(external_pin), 'external manifest pin mismatch')
    m = obj(parse(raw), 'schema archive_sha256 files', 'manifest')
    need(raw == canonical(m), 'noncanonical manifest')
    equals([m['schema'], m['archive_sha256']], [SCHEMA, ARCHIVE], 'manifest identity')
    need(type(m['files']) is dict and set(m['files']) == FILES, 'manifest file census mismatch')
    data = {}
    for name in sorted(FILES):
        entry = obj(m['files'][name], 'original_sha256 public_sha256', name)
        blob = (bundle / name).read_bytes()
        need(digest(blob) == pin(entry['public_sha256']), f'{name}: public file digest mismatch')
        value = unpack(blob) if name.endswith('.b64') else parse(blob)
        actual = digest(value) if name.endswith('.b64') else value['original_sha256']
        need(actual == pin(entry['original_sha256']), f'{name}: original digest binding mismatch')
        if name.endswith('.json'):
            need(blob == canonical(value), f'{name}: noncanonical JSON')
            bind_record(value, name)
        data[name] = value
    return data


def cost_account(route, row, fixture, shader_length):
    t, a, l, b = len(fixture.triangles), fixture.active, fixture.lanes, len(fixture.batches)
    if route == 'cuda':
        obj(row, C_KEYS, 'CUDA cost')
        equals([row['schema'], row['mode'], row['structurally_valid'], row['triangles'], row['active'], row['lanes']],
               ['s13_cuda_cost_v1', 'performance', True, t, a, l], 'CUDA cost identity')
        nodes, primitives, scene, owned = numbers(row, C_BYTES)
        count, remainder = divmod(nodes, 32)
        need(remainder == 0 and count % 2 and 2*((t+3)//4)-1 <= count <= 2*t-1, 'CUDA BVH nodes invalid')
        need((primitives, scene, owned) == (48*t, nodes+48*t, nodes+48*t+36*l), 'CUDA byte ledger mismatch')
        total = sum(numbers(row, C_SETUP))
        need(type(row['batches']) is list and len(row['batches']) == b, 'CUDA batch census mismatch')
        for i, r in enumerate(row['batches']):
            obj(r, 'batch service_ns ' + C_IO + ' ' + C_BATCH, 'CUDA batch')
            need(numbers(r, 'batch ' + C_IO) == [i, 32*l, 4*l, 4*l], 'CUDA batch bytes or index mismatch')
            service = uint(r['service_ns'])
            need(sum(numbers(r, C_BATCH)) <= service, 'CUDA parts exceed service')
            total += service
        return total
    obj(row, V_KEYS, 'Vulkan cost')
    equals([row['schema'], row['timer']], ['study13.vulkan.matched.host_cost.v1', 'host_steady_clock_ns'], 'Vulkan timer')
    equals(row['fixture'], dict(profile=fixture.profile, seed=fixture.seed, triangles=t, batches=b,
                               active_per_batch=a, padded_per_batch=l), 'Vulkan fixture')
    total = sum(numbers(obj(row['host_stages_ns'], V_SETUP, 'Vulkan stages'), V_SETUP))
    need(type(row['batches']) is list and len(row['batches']) == b, 'Vulkan batch census mismatch')
    for i, r in enumerate(row['batches']):
        obj(r, 'index scope measured_stage_sum_ns ' + V_BATCH, 'Vulkan batch')
        stage = sum(numbers(r, V_BATCH))
        equals([r['index'], r['scope'], r['measured_stage_sum_ns']],
               [i, 'build_and_dispatch' if i == 0 else 'reuse_and_dispatch', stage], 'Vulkan batch ledger')
        total += stage
    need(uint(row['host_stage_total_ns']) == total, 'Vulkan stage total mismatch')
    for name, labels, expect in (
        ('logical_host_payload_bytes', V_PAYLOAD, [shader_length, 36*t, 32*l*b, 64, 4*l, 4*l*b, 4*l*b]),
        ('mapped_io_bytes', V_IO, [36*t, 64, 32*l, 4*l, 4*l]),
    ):
        need(numbers(obj(row[name], labels, name), labels) == expect, f'{name}: bytes mismatch')
    inv = obj(row['vulkan_owned_buffers'], V_BUFFERS, 'Vulkan buffer inventory')
    req_total = alloc_total = 0
    for name, buf in inv.items():
        obj(buf, 'requested_bytes allocation_bytes mapped host_coherent', 'Vulkan buffer')
        req, alloc = numbers(buf, 'requested_bytes allocation_bytes')
        mapped = name in {'vertices', 'instances', 'rays', 'output'}
        need(buf['mapped'] is mapped and type(buf['host_coherent']) is bool and alloc >= req
             and (req > 0 or name == 'scratch' and alloc == 0), 'invalid Vulkan buffer inventory')
        req_total += req
        alloc_total += alloc
    need(inv['vertices']['requested_bytes'] >= 36*t and inv['instances']['requested_bytes'] >= 64
         and inv['rays']['requested_bytes'] == 32*l and inv['output']['requested_bytes'] == 4*l,
         'Vulkan buffer geometry mismatch')
    equals(row['vulkan_owned_buffer_totals'], dict(requested_bytes=req_total, allocation_bytes=alloc_total), 'Vulkan buffer totals')
    return total


def schedule():
    return dict(schema=SCHEMA, rows=[list(x) for x in ROWS], wall_boundary=BOUNDARY,
                fixture_sha256={k: PAYLOAD_PINS[k] for k in FIXTURES}, shader_sha256=PAYLOAD_PINS['shader'], sources=SOURCES)


def build_table():
    return dict(routes=PRODUCERS, shader_sha256=PAYLOAD_PINS['shader'])


def order_routes(order):
    return list(ROUTES if order == 'cuda-first' else ROUTES[::-1])


def status(route, f):
    row = dict(status='raw_written' if route == 'cuda' else 'readback_collected',
               triangles=len(f.triangles), active_per_batch=f.active, padded_per_batch=f.lanes,
               batches=len(f.batches), words_retained=f.lanes * len(f.batches))
    if route == 'vulkan':
        row.update(backend='vulkan_ray_query', invalid_active_words=0, invalid_tail_words=0)
    return row


def replay(bundle, external_pin):
    data = load_bundle(bundle, external_pin)
    # A trusted sibling implementation, never a source file supplied by evidence.
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exact_geometry import audit, bind, compare, decode, expected, first_hit

    for name, source, record in (('schedule.json', PLAN, schedule()), ('builds.json', BUILD, build_table())):
        need(data[name]['original_sha256'] == source, f'{name}: source identity mismatch')
        equals(data[name]['record'], record, name)
    collection = data['collection.json']
    need(collection['original_sha256'] == LEDGER, 'collection source ledger mismatch')
    run = obj(collection['record'], 'wrapper_exit wrapper_exit_sha256 phase phase_sha256 pairs', 'collection')
    equals([run['wrapper_exit'], run['wrapper_exit_sha256'], run['phase'], run['phase_sha256']],
           [0, digest(b'0\n'), 'collected-awaiting-independent-replay', digest(b'collected-awaiting-independent-replay\n')], 'collection status')
    need(type(run['pairs']) is list and len(run['pairs']) == len(ROWS), 'collection row census mismatch')
    shader = data[SHADER]
    need(digest(shader) == PAYLOAD_PINS['shader'] and shader.startswith(b'\x03\x02\x23\x07'), 'shader binding mismatch')
    fixtures, refs, audits = {}, {}, {}
    for name, filename in FIXTURES.items():
        raw = data[filename]
        need(digest(raw) == PAYLOAD_PINS[name], 'fixture source binding mismatch')
        f = fixtures[name] = decode(raw)
        need((f.profile, f.mode) == (name, 'performance'), 'fixture scene mismatch')
        refs[name] = expected(f, first_hit)
        audits[name] = audit(f, refs[name])
    pairs, acquisition_pins = [], set()
    for (pair, scene, order), cr in zip(ROWS, run['pairs'], strict=True):
        wrapped = data[f'{pair}.json']
        acq = obj(wrapped['record'], 'pair scene order status mode wall_boundary fixture_sha256 shader_sha256 oracle_sha256 launches', 'acquisition')
        equals([acq['pair'], acq['scene'], acq['order'], acq['status'], acq['mode'], acq['wall_boundary'],
                acq['fixture_sha256'], acq['shader_sha256'], acq['oracle_sha256']],
               [pair, scene, order_routes(order), 'collected', 'linux-native-recorded', BOUNDARY,
                PAYLOAD_PINS[scene], PAYLOAD_PINS['shader'], ORACLE], 'acquisition identity and order')
        orig = wrapped['original_sha256']
        need(orig not in acquisition_pins, 'duplicate original acquisition')
        acquisition_pins.add(orig)
        equals(cr, dict(pair=pair, original_sha256=orig, public_sha256=wrapped['public_sha256'],
                        acquire_exit=0, acquire_exit_sha256=digest(b'0\n'), acquire_stderr_sha256=digest(b''),
                        acquire_stdout_sha256=digest(b'collected\n')), 'collection acquisition binding')
        obj(acq['launches'], 'cuda vulkan', 'route census')
        routes = []
        for route in acq['order']:
            name = f'{pair}/{route}'
            w = acq['launches'][route]
            launch = obj(bind_record(w, name), 'pair route producer fixture_sha256 shader_sha256 child_exit child_wall_ns wall_boundary raw_sha256 stderr_sha256 stdout' + (' cost' if route == 'vulkan' else ''), name)
            equals([launch['pair'], launch['route'], launch['producer'], launch['fixture_sha256'], launch['shader_sha256'],
                    launch['child_exit'], launch['wall_boundary'], launch['stderr_sha256']],
                   [pair, route, PRODUCERS[route], PAYLOAD_PINS[scene], PAYLOAD_PINS['shader'], 0, BOUNDARY, digest(b'')], name + ' identity')
            raw = data[f'{pair}.{route}.raw.b64']
            need(digest(raw) == pin(launch['raw_sha256']), f'{name}: raw provenance mismatch')
            f = fixtures[scene]
            exact = compare(f, refs[scene].words, bind(f, raw))
            need(exact['status'] == 'pass', f'{name}: exact CPU visibility mismatch ({exact["mismatches"]} of {exact["words_checked"]})')
            stdout = bind_record(launch['stdout'], name + ' stdout')
            if route == 'cuda':
                obj(stdout, 'summary cost', 'CUDA stdout')
                equals(stdout['summary'], status(route, f), 'CUDA stdout summary')
                cost = stdout['cost']
            else:
                equals(stdout, status(route, f), 'Vulkan stdout summary')
                cost = bind_record(launch['cost'], name + ' Vulkan sidecar')
            accounted = cost_account(route, cost, f, len(shader))
            wall = uint(launch['child_wall_ns'])
            need(0 < wall and accounted <= wall, f'{name}: stages exceed child wall or invalid wall')
            routes.append(dict(route=route, producer=launch['producer'], launch_original_sha256=w['original_sha256'],
                               launch_public_sha256=w['public_sha256'], raw_sha256=launch['raw_sha256'],
                               exact=exact, canaries_checked=(f.lanes-f.active)*len(f.batches), child_exit=0,
                               child_wall_ns=wall, accounted_child_wall_ns=accounted,
                               unattributed_child_wall_ns=wall-accounted))
        pairs.append(dict(pair=pair, scene=scene, order=order, routes=routes))
    all_routes = [r for p in pairs for r in p['routes']]
    return dict(schema=SCHEMA + '-replay', evidence_manifest_sha256=external_pin, status='pass',
                pairs_checked=len(pairs), routes_checked=len(all_routes), words_checked=sum(r['exact']['words_checked'] for r in all_routes),
                canaries_checked=sum(r['canaries_checked'] for r in all_routes),
                hardware_attestation=False, execution_attested=False, comparison_eligible=False,
                wall_boundary=BOUNDARY, fixtures=audits, pairs=pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('manifest_sha256', help='independent SHA-256 pin, not read from the bundle')
    a = p.parse_args()
    try:
        sys.stdout.buffer.write(canonical(replay(a.bundle, a.manifest_sha256)))
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as e:
        p.exit(1, f'Public replay rejected: {e}\n')


if __name__ == '__main__':
    main()
