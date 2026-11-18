"""Check compact Study 14 SIFT receipts, not private frame witnesses."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTIER = 'sift-frontier.json'
REFS = ('ordered-copy', 'stock-serial')
CODECS = ('ndzip', 'xbp')
REUSES = ('1', '10', '100')
CELLS = {(cohort, dtype, size, seed) for cohort, seed in (('a', 71), ('b', 171))
         for dtype in ('<f4', '<f8') for size in (2**18, 2**20, 2**22, 2**24, 2**26)}
VERIFIERS = {'compression_pair_models.py', 'verify_compression_genuine_cost.py',
             'verify_compression_genuine_output.py', 'verify_compression_genuine_preparation.py'}
PINS = {
    FRONTIER: 'b16d76859c37dc2b4b4a9b91304f6dc6a39353ca74808972cc3e394379dd389a',
    'compression-genuine-ordered-copy-full-cost-replay.json': 'e4d46e501198e39237f632c1feed4e9f88d2745deea063509d57649f22f9b7f1',
    'compression-genuine-ordered-copy-full-output-replay.json': '8fb6a4d9dc38f05779d5b77586dd902660bde3dcc6946df010bb56b05e8d70f1',
    'compression-genuine-stock-serial-full-cost-replay.json': '844bff410412c4a299dc25f72175ff4588441d81f35fc85b25eeda2cb43bd31c',
    'compression-genuine-stock-serial-full-output-replay.json': '1174efb682d5765ba20480f67477a757d3591f8102ec76cdea9ccd7ffb819ee3',
}


def need(condition, why):
    if not condition:
        raise ValueError(why)


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def load_pinned(path):
    raw = path.read_bytes()
    need(not path.is_symlink() and sha256(raw).hexdigest() == PINS[path.name],
         f'receipt digest mismatch: {path.name}')
    return json.loads(raw)


def positive_fraction(values):
    need(isinstance(values, list) and len(values) == 2
         and all(type(x) is int and x > 0 for x in values), 'invalid positive rational')
    value = Fraction(*values)
    need([value.numerator, value.denominator] == values, 'noncanonical rational')
    return value


def check_epoch(epoch):
    cells = epoch['cells']
    actual = {(row['cohort'], row['dtype'], row['frame_bytes'], row['seed']) for row in cells}
    need(len(cells) == len(CELLS) and actual == CELLS, 'cell census differs')
    wins = {reuse: dict.fromkeys(('any', *CODECS), 0) for reuse in REUSES}
    for cell in cells:
        need(set(cell['frontier']) == set(REUSES), 'reuse census differs')
        for reuse, entry in cell['frontier'].items():
            best = entry['fastest_compressed']
            policy = best['policy']
            need(best['codec'] in CODECS and policy in ('online', 'precompressed')
                 and best['queue_depth'] in (1, 2) and entry['plain_queue_depth'] in (1, 2),
                 'unknown route')
            plain = positive_fraction(entry['plain_amortized_ns'])
            compressed = positive_fraction(best['amortized_ns'])
            by_codec = entry['strict_wins_by_codec']
            need(set(by_codec) == set(CODECS) and all(type(v) is bool for v in by_codec.values())
                 and entry['strict_win'] is (compressed < plain)
                 and by_codec[best['codec']] is entry['strict_win']
                 and any(by_codec.values()) is entry['strict_win'], 'strict-win arithmetic differs')
            wins[reuse]['any'] += int(entry['strict_win'])
            for codec in CODECS:
                wins[reuse][codec] += int(by_codec[codec])
    need(epoch['strict_winning_cells'] == wins, 'strict-win census differs')
    ratios = epoch['encoded_to_original_wire_ratio']
    need(set(ratios) == set(CODECS), 'codec census differs')
    for codec, totals in ratios.items():
        low, high = positive_fraction(totals['min']), positive_fraction(totals['max'])
        expected = 80 if codec == 'xbp' else 0
        need(low <= high and (high < 1 if expected else low >= 1)
             and totals['routes'] == 80 and totals['smaller_routes'] == expected,
             'wire ratio receipt inconsistent')
    return [wins[r]['any'] for r in REUSES]


def audit(receipts, source_root):
    need({p.name for p in receipts.iterdir()} == set(PINS), 'receipt inventory differs')
    data = load_pinned(receipts / FRONTIER)
    need(data['schema'] == 'study14-sift-paired-comparison-v1'
         and data['reuse_counts'] == [1, 10, 100]
         and [epoch['reference'] for epoch in data['epochs']] == list(REFS), 'frontier scope differs')
    need(hash_file(source_root / 'compression_sift_frontier.py') == data['verifier_sha256'],
         'frontier verifier differs')
    reports = []
    for epoch in data['epochs']:
        ref = epoch['reference']
        out_name = f'compression-genuine-{ref}-full-output-replay.json'
        cost_name = f'compression-genuine-{ref}-full-cost-replay.json'
        out, cost = (load_pinned(receipts / name) for name in (out_name, cost_name))
        need(out['schema'] == 'compression-genuine-output-replay-v1' and out['suite'] == 'full'
             and cost['schema'] == 'compression-genuine-accounting-replay-v1', 'receipt schema differs')
        need(out['ndzip_reference'] == cost['reference'] == ref
             and out['collection_manifest_sha256'] == cost['collection_sha256'] == epoch['collection_manifest_sha256']
             and PINS[out_name] == cost['output_replay_sha256'] == epoch['output_replay_sha256']
             and PINS[cost_name] == epoch['cost_replay_sha256'], 'receipt binding differs')
        need(set(cost['verifiers']) == VERIFIERS and
             all(hash_file(source_root / name) == value for name, value in cost['verifiers'].items())
             and out['verifier_sha256'] == cost['verifiers']['verify_compression_genuine_output.py']
             and out['prep_verifier_sha256'] == cost['verifiers']['verify_compression_genuine_preparation.py'],
             'accounting source pins differ')
        need(out['cases'] == cost['cases'] == 200 and out['unique_witnesses'] == 4092
             and out['observations'] == cost['observation_count'] == epoch['observations'] == 81840
             and out['restored_byte_observations'] == cost['restored_byte_observations'],
             'output and accounting census differs')
        need(cost['named_case_ns'] > 0 and cost['other_child_ns'] >= 0
             and cost['process_wall_ns'] == cost['named_case_ns'] + cost['other_child_ns'],
             'whole process accounting differs')
        reports.append({'reference': ref, 'cases': cost['cases'], 'observations': out['observations'],
                        'strict_wins': check_epoch(epoch)})
    return {'schema': 'study14-sift-compact-receipt-audit-v1', 'epochs': reports,
            'cases': sum(r['cases'] for r in reports),
            'observations': sum(r['observations'] for r in reports),
            'raw_fingerprint_replay': False, 'all_candidate_frontier_recomputed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipts', type=Path, nargs='?', default=ROOT / 'results/sift-full-accounting-receipts')
    args = parser.parse_args()
    try:
        report = audit(args.receipts, ROOT)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
