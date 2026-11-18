"""Small content-addressed full-output witnesses for streaming movement experiments."""
import hashlib
from pathlib import Path
import numpy as np


class WitnessStore:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        (self.root / 'witnesses').mkdir(parents=True, exist_ok=True)

    def put(self, table):
        assert isinstance(table, np.ndarray) and table.dtype == np.dtype('u1')
        assert table.ndim == 2 and table.shape[1] == 32 and table.flags.c_contiguous
        digest = hashlib.sha256(table).hexdigest()
        relative = f'witnesses/{digest}.npy'
        path = self.root / relative
        assert not path.is_symlink()
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            assert existing.dtype == table.dtype and existing.shape == table.shape
            assert np.array_equal(existing, table), 'Existing witness bytes were altered'
        else:
            np.save(path, table, allow_pickle=False)
        return {'file': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'array_sha256': digest, 'array_bytes': table.nbytes, 'shape': list(table.shape)}
