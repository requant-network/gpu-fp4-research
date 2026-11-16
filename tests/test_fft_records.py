import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fft_records import load_witness


class FftRecordTests(unittest.TestCase):
    def test_exact_assembly_corruption_and_reproduced_file_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, values=np.arange(10, dtype=np.float32))
            raw = buffer.getvalue()
            expected = hashlib.sha256(raw).hexdigest()
            parts = []
            for i, data in enumerate((raw[:100], raw[100:])):
                name = f"part{i}.bin"
                (root/name).write_bytes(data)
                parts.append(dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            (root/"fft-artifacts.json").write_text(json.dumps({"data.npz": dict(bytes=len(raw), sha256=expected, parts=parts)}))
            with load_witness(root/"data.npz", expected) as z:
                np.testing.assert_array_equal(z["values"], np.arange(10, dtype=np.float32))
            (root/"part0.bin").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                load_witness(root/"data.npz", expected)
            (root/"data.npz").write_bytes(raw)
            with load_witness(root/"data.npz", expected) as z:
                self.assertEqual(len(z["values"]), 10)


if __name__ == "__main__":
    unittest.main()
