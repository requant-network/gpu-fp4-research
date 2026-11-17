"""Byte-exact original FP32 input guards, separate from numerical tolerances."""
import hashlib
import re

import numpy as np


def input_sha256(values):
    assert isinstance(values,np.ndarray) and values.dtype==np.float32 and values.ndim==2 and values.flags.c_contiguous
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


def verify_input(values,expected,require_readonly=True):
    assert isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected)
    if require_readonly:assert not values.flags.writeable,'Original input is not read-only'
    actual=input_sha256(values)
    assert actual==expected,'Original input bytes changed'
    return actual


def freeze_input(values,expected):
    verify_input(values,expected,require_readonly=False)
    values.setflags(write=False)
    return verify_input(values,expected)
