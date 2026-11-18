"""Compare the two independently replayed SIFT matrices against their plain controls."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

from verify_compression_genuine_cost import PINNED, ReplayError, audit_case, require, sha
from verify_compression_genuine_output import expected_routes

REUSES = (1, 10, 100)
COST_PINS = {
    'ordered-copy': 'e4d46e501198e39237f632c1feed4e9f88d2745deea063509d57649f22f9b7f1',
    'stock-serial': '844bff410412c4a299dc25f72175ff4588441d81f35fc85b25eeda2cb43bd31c',
}


def compare_cell(rows):
    plain = [row for row in rows if row['route']['codec'] == 'plain']
    compressed = [row for row in rows if row['route']['codec'] in ('ndzip', 'xbp')]
    require(len(plain) == 2 and {row['route']['queue_depth'] for row in plain} == {1, 2},
            'Missing or duplicate plain baseline')
    require(bool(compressed) and len(plain) + len(compressed) == len(rows), 'Missing compressed route')
    result = {}
    for reuse in REUSES:
        def time(row):
            return row['service_ns'] + Fraction(row['fixed_ns'], reuse)

        def order(row):
            route = row['route']
            return time(row), route['codec'], route['policy'], route['queue_depth']

        def encode(x):
            return [x.numerator, x.denominator]

        baseline = min(plain, key=order)
        best = min(compressed, key=order)
        result[reuse] = {'plain_queue_depth': baseline['route']['queue_depth'],
                         'plain_amortized_ns': encode(time(baseline)),
                         'plain_wire_bytes': baseline['wire_bytes'],
                         'fastest_compressed': {**best['route'], 'wire_bytes': best['wire_bytes'],
                                                'amortized_ns': encode(time(best))},
                         'strict_win': time(best) < time(baseline),
                         'strict_wins_by_codec': {codec: any(time(r) < time(baseline) for r in compressed
                                                           if r['route']['codec'] == codec)
                                                  for codec in ('ndzip', 'xbp')}}
    return result


def read_record(path, pin):
    raw = path.read_bytes()
    require(sha(raw) == pin, f'Pinned input differs: {path.name}')
    return json.loads(raw)


def evaluate(collection_root, receipts, reference):
    manifest_pin, output_pin = PINNED[reference]
    stem = f'compression-genuine-{reference}-full'
    record = read_record(collection_root / stem / 'manifest.json', manifest_pin)
    output = read_record(receipts / (stem + '-output-replay.json'), output_pin)
    cost = read_record(receipts / (stem + '-cost-replay.json'), COST_PINS[reference])
    require(record['ndzip_reference'] == output['ndzip_reference'] == cost['reference'] == reference
            and record['complete'] is True and record['suite'] == output['suite'] == 'full'
            and manifest_pin == output['collection_manifest_sha256'] == cost['collection_sha256']
            and output_pin == cost['output_replay_sha256'], 'Prior receipts disagree')
    schedule = expected_routes('full', reference)
    require(len(schedule) == len(record['cases']) == cost['cases'] == output['cases'] == 200,
            'Incomplete route matrix')
    groups = defaultdict(list)
    acc = dict.fromkeys(('named_case_ns', 'observation_count', 'restored_byte_observations',
                         'wire_bytes_observed', 'expected_digest_bytes_observed'), 0)
    for i, (case, (identity, route)) in enumerate(zip(record['cases'], schedule, strict=True)):
        require(case['index'] == i and tuple(case[k] for k in ('cohort', 'dtype', 'frame_bytes', 'seed')) == identity
                and case['route'] == route, 'Route order differs')
        row = audit_case(case)
        for key in acc:
            acc[key] += row[key]
        service = sum(trial['host_ns'] for trial in case['trials'])
        fixed = row['named_case_ns'] - service
        groups[identity].append({'route': route, 'service_ns': Fraction(service, 2),
                                 'fixed_ns': fixed, 'wire_bytes': row['wire_bytes_observed'] // 2,
                                 'input_bytes': case['total_bytes']})
    require(all(cost[key] == value for key, value in acc.items())
            and output['observations'] == acc['observation_count']
            and output['restored_byte_observations'] == acc['restored_byte_observations'], 'Accounting census differs')
    require(len(groups) == 20 and all(len(rows) == 10 for rows in groups.values()), 'Incomplete comparison cells')
    cells = []
    wire = defaultdict(list)
    for (cohort, dtype, frame_bytes, seed), rows in groups.items():
        cells.append({'cohort': cohort, 'dtype': dtype, 'frame_bytes': frame_bytes, 'seed': seed,
                      'frontier': compare_cell(rows)})
        for row in rows:
            if row['route']['codec'] != 'plain':
                wire[row['route']['codec']].append(Fraction(row['wire_bytes'], row['input_bytes']))

    def show(frac):
        return [frac.numerator, frac.denominator]

    return {'reference': reference, 'collection_manifest_sha256': manifest_pin, 'output_replay_sha256': output_pin,
            'cost_replay_sha256': COST_PINS[reference], 'cells': cells, 'observations': acc['observation_count'],
            'strict_winning_cells': {r: {'any': sum(x['frontier'][r]['strict_win'] for x in cells),
                                         **{k: sum(x['frontier'][r]['strict_wins_by_codec'][k] for x in cells)
                                            for k in ('ndzip', 'xbp')}} for r in REUSES},
            'encoded_to_original_wire_ratio': {k: {'min': show(min(v)), 'max': show(max(v)),
                                                     'smaller_routes': sum(x < 1 for x in v), 'routes': len(v)}
                                                for k, v in sorted(wire.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('collections', type=Path)
    p.add_argument('--receipts', required=True, type=Path)
    p.add_argument('--summary', required=True, type=Path)
    a = p.parse_args()
    try:
        rows = [evaluate(a.collections, a.receipts, ref) for ref in COST_PINS]
        result = {'schema': 'study14-sift-paired-comparison-v1', 'epochs': rows, 'reuse_counts': REUSES,
                  'verifier_sha256': sha(Path(__file__).read_bytes()),
                  'scope': ('Observed two-repeat mean host service plus other named case stages divided by reuse. '
                            'Ex-post same-cell plain controls; excludes unattributed process time. '
                            'Not a selector, cold process wall, or statistical performance claim.')}
        with a.summary.open('x', encoding='ascii', newline='\n') as output:
            output.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        p.exit(1, f'NOT ELIGIBLE: {error}\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'cells'} for r in rows], sort_keys=True))


if __name__ == '__main__':
    main()
