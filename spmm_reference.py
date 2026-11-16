"""Lossless sparse/dense block decomposition for the pending SpMM investigation."""
import numpy as np


def csr(matrix):
    """Canonical sorted CSR; exact zeros are omitted without a drop tolerance."""
    if matrix.ndim != 2 or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("A finite FP32 matrix is required")
    rows, columns = np.nonzero(matrix)
    if len(columns) >= 2**31 or max(matrix.shape) >= 2**31:
        raise ValueError("The native protocol uses 32-bit sparse indices")
    rowptr = np.zeros(matrix.shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows, minlength=matrix.shape[0]), out=rowptr[1:])
    return {"rowptr": rowptr, "columns": columns.astype(np.int32),
            "values": np.ascontiguousarray(matrix[rows, columns])}


def partition(matrix, block_size, minimum_fill):
    """Move occupied blocks above a fill threshold into dense storage exactly.

    The denominator includes the entire padded block because the GPU pays for
    padded arithmetic/storage. Unselected entries remain in canonical CSR.
    """
    if matrix.ndim != 2 or matrix.dtype != np.float32 or not np.isfinite(matrix).all() or min(matrix.shape) < 1:
        raise ValueError("A nonempty finite FP32 matrix is required")
    if block_size not in (2, 4, 8, 16, 32, 64) or not 0 < minimum_fill <= 1:
        raise ValueError("Invalid block size or fill threshold")
    m, k = matrix.shape
    mb, kb = (m + block_size - 1) // block_size, (k + block_size - 1) // block_size
    padded = np.zeros((mb * block_size, kb * block_size), dtype=np.float32)
    padded[:m, :k] = matrix
    tiles = padded.reshape(mb, block_size, kb, block_size).transpose(0, 2, 1, 3)
    nonzeros = np.count_nonzero(tiles, axis=(2, 3))
    mask = (nonzeros > 0) & (nonzeros >= minimum_fill * block_size**2)
    block_rows, block_columns = np.nonzero(mask)
    values = np.ascontiguousarray(tiles[block_rows, block_columns])
    tiles[block_rows, block_columns] = 0
    remainder = csr(np.ascontiguousarray(padded[:m, :k]))
    block_rowptr = np.zeros(mb + 1, dtype=np.int32)
    np.cumsum(np.bincount(block_rows, minlength=mb), out=block_rowptr[1:])
    return {"shape": matrix.shape, "block_size": block_size, "minimum_fill": minimum_fill,
            "block_rows": block_rows.astype(np.int32), "block_columns": block_columns.astype(np.int32),
            "block_rowptr": block_rowptr, "block_values": values, **remainder}


def reconstruct(parts):
    """Exact lossless-decomposition witness, before any numerical quantization."""
    m, k = parts["shape"]
    block = parts["block_size"]
    result = np.zeros((m, k), dtype=np.float32)
    for row in range(m):
        start, end = parts["rowptr"][row:row+2]
        result[row, parts["columns"][start:end]] = parts["values"][start:end]
    for row, column, values in zip(parts["block_rows"], parts["block_columns"], parts["block_values"]):
        i, j = int(row) * block, int(column) * block
        height, width = min(block, m - i), min(block, k - j)
        result[i:i+height, j:j+width] += values[:height, :width]
    return result


def product(parts, right):
    """Independent FP64 product of the exact sparse/dense decomposition."""
    m, k = parts["shape"]
    if right.ndim != 2 or right.shape[0] != k or not np.isfinite(right).all():
        raise ValueError("Invalid dense right operand")
    result = np.zeros((m, right.shape[1]), dtype=np.float64)
    for row in range(m):
        start, end = parts["rowptr"][row:row+2]
        columns, values = parts["columns"][start:end], parts["values"][start:end]
        result[row] = values.astype(np.float64) @ right[columns].astype(np.float64)
    block = parts["block_size"]
    for row, column, values in zip(parts["block_rows"], parts["block_columns"], parts["block_values"]):
        i, j = int(row) * block, int(column) * block
        height, width = min(block, m - i), min(block, k - j)
        result[i:i+height] += values[:height, :width].astype(np.float64) @ right[j:j+width].astype(np.float64)
    return result
