"""Observed-byte census for plain-transfer failures; no hardware-cause attribution."""
import hashlib
import numpy as np
from compression_integrity import digest_chunks


def census(original,regenerated,restored,expected,returned,gpu,prepared_guard):
    assert original.dtype==regenerated.dtype==restored.dtype and original.shape==regenerated.shape==restored.shape
    word='<u4' if original.dtype.itemsize==4 else '<u8'
    old=regenerated.view(word).reshape(-1);now=restored.view(word).reshape(-1)
    different=np.flatnonzero(old!=now)
    return {'host_source_matches_prepared_guard':hashlib.sha256(original).hexdigest()==prepared_guard,
            'host_source_matches_regeneration':np.array_equal(original.view(word),regenerated.view(word)),
            'expected_matches_regeneration':np.array_equal(expected,digest_chunks(regenerated,4096)),
            'restored_matches_regeneration':np.array_equal(restored.view(word),regenerated.view(word)),
            'restored_matches_current_host':np.array_equal(restored.view(word),original.view(word)),
            'restored_different_words':int(different.size),
            'gpu_digest_matches_cpu_restored':np.array_equal(gpu,digest_chunks(restored,4096)),
            'returned_host_digest_matches_gpu':np.array_equal(returned,gpu),
            'first_differences':[{'word':int(i),'expected':int(old[i]),'restored':int(now[i])} for i in different[:16]]}
