import gzip
from pathlib import Path
import tempfile
import unittest

from spmm_records import load_record


class SpmmRecordTests(unittest.TestCase):
    def test_compressed_records_and_reproduction_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.json'
            archive = path.with_name(path.name + '.gz')
            archive.write_bytes(gzip.compress(b'{"complete":true,"seed":1}\n', mtime=0))
            self.assertEqual(load_record(path), {'complete': True, 'seed': 1})
            self.assertEqual(load_record(archive), {'complete': True, 'seed': 1})
            path.write_text('{"complete":true,"seed":2}\n', encoding='utf-8')
            self.assertEqual(load_record(path)['seed'], 2)
            archive.write_bytes(b'not gzip')
            with self.assertRaises(gzip.BadGzipFile):
                load_record(archive)


if __name__ == '__main__':
    unittest.main()
