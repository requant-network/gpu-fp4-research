from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import replay as r

ROOT = Path(__file__).resolve().parent
PIN = '0cfed9098f220a0b9f18c2dcde813ce9bfa968304ba53366e5fe45aa2c9660a5'


class ExactArithmetic(unittest.TestCase):
    def test_open_interval_and_closed_triangle_edges(self):
        intersects = r.intersects
        tri = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        self.assertTrue(intersects((0, 0, -1), (0, 0, 1), tri))
        self.assertFalse(intersects((0, 0, -1), (0, 0, 0), tri))
        self.assertFalse(intersects((4, 4, -1), (4, 4, 1), tri))
        self.assertTrue(intersects((-1, 0, 0), (1, 0, 0), tri))
        self.assertFalse(intersects((1, 1, 0), (1, 1, 0), tri))
        self.assertFalse(intersects((0, 0, -1), (0, 0, 1), (tri[0], tri[1], tri[1])))


class PublicReplay(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        shutil.copytree(ROOT / 'evidence', self.root / 'evidence')
        shutil.copyfile(ROOT / 'public-manifest.json', self.root / 'public-manifest.json')

    def load(self, kind):
        return r.parse((self.root / 'evidence' / f'{kind}.json').read_bytes())

    def reseal(self, kind, obj):
        # New internally consistent public file bindings, simulating a forged publisher pin.
        name = f'{kind}.json'
        b = r.canonical(obj)
        (self.root / 'evidence' / name).write_bytes(b)
        manifest = self.root / 'public-manifest.json'
        m = r.parse(manifest.read_bytes())
        m['evidence'][name] = r.digest(b)
        b = r.canonical(m)
        manifest.write_bytes(b)
        return r.digest(b)

    @staticmethod
    def rehash(row):
        row['public_sha256'] = r.digest(r.canonical(row['data']))

    def test_frozen_receipt_and_counts(self):
        result = r.replay(self.root, PIN)
        self.assertEqual(r.canonical(result), (ROOT / 'receipt.json').read_bytes())
        self.assertEqual((result['geometry']['words_checked'], result['geometry']['canaries_checked']), (1024, 944))
        self.assertEqual((result['quartet']['words_checked'], result['quartet']['canaries_checked']), (4096, 3776))
        self.assertEqual(len(result['quartet']['specimens']), 16)
        self.assertTrue(all(result[k] is False for k in r.DISCLAIMERS))

    def test_paired_deltas_follow_the_same_order_and_timer(self):
        from fractions import Fraction
        result = r.replay(self.root, PIN)['quartet']
        samples = result['specimens']
        for metric in ('after_edit_ns', 'whole_route_ns'):
            for contrast in ('B-A', 'D-C', 'C-A', 'D-B'):
                deltas = []
                for g in result['quartets']:
                    l, rr = (g['order'] + '-' + arm for arm in contrast.split('-'))
                    d = samples[l][metric] - samples[rr][metric]
                    self.assertEqual(g['paired_deltas_ns'][metric][contrast], d)
                    deltas.append(d)
                a, b, c, d = sorted(deltas)
                self.assertEqual(result['recorded_paired_summary_ns'][metric][contrast],
                                 dict(n=4, min_ns=a, median_ns=Fraction(b + c, 2), max_ns=d))

    def test_wrong_external_pin(self):
        with self.assertRaisesRegex(ValueError, 'external pin required'):
            r.replay(self.root, '0' * 64)

    def test_extra_file(self):
        (self.root / 'evidence' / 'unknown.json').touch()
        with self.assertRaisesRegex(ValueError, 'evidence directory census'):
            r.replay(self.root, PIN)

    def test_missing_file(self):
        (self.root / 'evidence' / 'geometry.json').unlink()
        with self.assertRaisesRegex(ValueError, 'evidence directory census'):
            r.replay(self.root, PIN)

    def test_rehashed_quartet_order(self):
        q = self.load('quartet')
        q['matrix']['data']['quartets'][0]['specimens'].reverse()
        self.rehash(q['matrix'])
        with self.assertRaisesRegex(ValueError, 'matrix: quartets mismatch'):
            r.replay(self.root, self.reseal('quartet', q))

    def test_rehashed_quartet_route(self):
        q = self.load('quartet')
        q['specimens']['ABDC-A']['observation']['data']['route'] = 'rebuild'
        self.rehash(q['specimens']['ABDC-A']['observation'])
        with self.assertRaisesRegex(ValueError, 'ABDC-A: route mismatch'):
            r.replay(self.root, self.reseal('quartet', q))

    def test_rehashed_timing_scope(self):
        q = self.load('quartet')
        row = q['specimens']['ABDC-D']['timing']
        row['data']['scope'] = 'build-only'
        self.rehash(row)
        with self.assertRaisesRegex(ValueError, 'ABDC-D/timing: scope mismatch'):
            r.replay(self.root, self.reseal('quartet', q))

    def test_rehashed_vulkan_topology_control(self):
        g = self.load('geometry')
        row = g['cases']['different-topology']['routes']['vulkan']['metadata']['after']
        row['data']['blas_mode'] = 'UPDATE'
        self.rehash(row)
        with self.assertRaisesRegex(ValueError, 'Vulkan after metadata: blas_mode mismatch'):
            r.replay(self.root, self.reseal('geometry', g))

    def test_rehashed_wrong_epoch_readback(self):
        q = self.load('quartet')
        row = q['specimens']['ABDC-A']
        row['readbacks']['after'] = row['readbacks']['before']
        self.rebind_cuda_after(row)
        with self.assertRaisesRegex(ValueError, 'ABDC-A/after: exact word mismatch at batch 0 lane 0'):
            r.replay(self.root, self.reseal('quartet', q))

    def test_coherently_rehashed_geometry_active_word(self):
        g = self.load('geometry')
        row = g['cases']['same-topology']['routes']['cuda']
        self.flip_word(row['readbacks']['after'], 2)
        row['launch']['data']['artifacts']['after.raw'] = row['readbacks']['after']['sha256']
        self.rehash(row['launch'])
        with self.assertRaisesRegex(ValueError, 'same-topology/cuda/after: exact word mismatch at batch 0 lane 2'):
            r.replay(self.root, self.reseal('geometry', g))

    def test_coherently_rehashed_quartet_canary(self):
        q = self.load('quartet')
        row = q['specimens']['ABDC-A']
        self.flip_word(row['readbacks']['after'], 127)
        self.rebind_cuda_after(row)
        with self.assertRaisesRegex(ValueError, 'ABDC-A/after: exact word mismatch at batch 1 lane 63'):
            r.replay(self.root, self.reseal('quartet', q))

    @staticmethod
    def flip_word(obj, lane):
        b = bytearray.fromhex(obj['hex'])
        b[4 * lane] ^= 1
        obj.update(hex=b.hex(), sha256=r.digest(b))

    def rebind_cuda_after(self, row):
        sha = row['readbacks']['after']['sha256']
        row['observation']['data']['artifacts']['after.raw'] = sha
        row['timing']['data']['after_raw_sha256'] = sha
        for field in ('observation', 'timing'):
            self.rehash(row[field])


if __name__ == '__main__':
    unittest.main()
