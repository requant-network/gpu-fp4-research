"""CSR-based format preparation; the separate dense oracle checks losslessness."""
import numpy as np


def validate(source, shape):
    m, k = shape
    rp, ci, values = (source[key] for key in ("rowptr", "columns", "values"))
    if not 1 <= m <= 4096 or not 1 <= k <= 4096:
        raise ValueError("Study dimensions must be in [1,4096]")
    if rp.dtype != np.int32 or ci.dtype != np.int32 or values.dtype != np.float32 or any(x.ndim != 1 for x in (rp, ci, values)):
        raise ValueError("Expected one-dimensional int32 CSR indices and FP32 values")
    if len(rp) != m + 1 or rp[0] != 0 or rp[-1] != len(values) or len(ci) != len(values) or np.any(np.diff(rp) < 0):
        raise ValueError("Invalid CSR offsets/extents")
    if np.any(ci < 0) or np.any(ci >= k) or not np.isfinite(values).all():
        raise ValueError("Out-of-range CSR column or nonfinite value")
    rows = np.repeat(np.arange(m, dtype=np.int32), np.diff(rp))
    if len(ci) > 1 and np.any((rows[1:] == rows[:-1]) & (ci[1:] <= ci[:-1])):
        raise ValueError("CSR columns must be sorted and unique per row")
    return rows


def reorder(source, shape, order, seed):
    rows = validate(source, shape)
    if order == "natural":
        permutation = np.arange(shape[0], dtype=np.int32)
        return dict(source), permutation
    elif order == "degree":
        permutation = np.argsort(-np.diff(source["rowptr"]), kind="stable").astype(np.int32)
    elif order == "scrambled":
        permutation = np.random.Generator(np.random.PCG64(seed + 7919)).permutation(shape[0]).astype(np.int32)
    else:
        raise ValueError("Unknown row ordering")
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation), dtype=np.int32)
    new_rows = inverse[rows]
    sort = np.lexsort((source["columns"], new_rows))
    rp = np.zeros(shape[0] + 1, dtype=np.int32)
    np.cumsum(np.bincount(new_rows, minlength=shape[0]), out=rp[1:])
    return {"rowptr": rp, "columns": source["columns"][sort].copy(), "values": source["values"][sort].copy()}, permutation


def split(source, shape, block, minimum_fill):
    """Partition directly from CSR without allocating an m-by-k dense matrix."""
    rows = validate(source, shape)
    if block not in (2, 4, 8, 16, 32, 64) or not 0 <= minimum_fill <= 1:
        raise ValueError("Invalid block size or padded-block fill threshold")
    m, k = shape
    mb, kb = (m + block - 1) // block, (k + block - 1) // block
    keys = (rows.astype(np.int64) // block) * kb + source["columns"] // block
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    selected = counts >= minimum_fill * block**2
    dense_entry = selected[inverse]
    selected_map = np.cumsum(selected, dtype=np.int64) - 1
    dense_indices = selected_map[inverse[dense_entry]]
    values = np.zeros((int(selected.sum()), block, block), dtype=np.float32)
    values[dense_indices, rows[dense_entry] % block, source["columns"][dense_entry] % block] = source["values"][dense_entry]
    selected_keys = unique[selected]
    block_rows, block_columns = (selected_keys // kb).astype(np.int32), (selected_keys % kb).astype(np.int32)
    brp = np.zeros(mb + 1, dtype=np.int32)
    np.cumsum(np.bincount(block_rows, minlength=mb), out=brp[1:])
    rp = np.zeros(m + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows[~dense_entry], minlength=m), out=rp[1:])
    return {"shape": shape, "block_size": block, "minimum_fill": minimum_fill,
            "block_rows": block_rows, "block_columns": block_columns, "block_rowptr": brp, "block_values": values,
            "rowptr": rp, "columns": source["columns"][~dense_entry].copy(), "values": source["values"][~dense_entry].copy()}


def prepare(source, shape, method, order, seed):
    """Return FP32 source-format buffers; GPU encoding is charged separately."""
    source, permutation = reorder(source, shape, order, seed)
    kind, block = method["kind"], method.get("block", 1)
    if kind not in (0, 1, 2, 3, 4):
        raise ValueError("Unknown native method kind")
    if kind == 4:
        block = 16
    if kind == 0:
        block = 1
    m, k = shape
    mp, kp = ((m + block - 1) // block) * block, ((k + block - 1) // block) * block
    result = {"shape": shape, "padded_shape": (mp, kp), "kind": kind, "block": block, "permutation": permutation,
              "rowptr": np.zeros(1, dtype=np.int32), "columns": np.empty(0, dtype=np.int32), "values": np.empty(0, dtype=np.float32),
              "block_rowptr": np.zeros(1, dtype=np.int32), "block_columns": np.empty(0, dtype=np.int32),
              "block_values": np.empty(0, dtype=np.float32), "block_count": 0, "ell_columns": 0}
    if kind == 0:
        result.update(source)
    elif kind == 4:
        dense = np.zeros((mp, kp), dtype=np.float32)
        rows = np.repeat(np.arange(m), np.diff(source["rowptr"]))
        dense[rows, source["columns"]] = source["values"]
        result["values"] = dense.ravel()
    else:
        parts = split(source, shape, block, method.get("fill", 0) if kind == 1 else 0)
        count = len(parts["block_columns"])
        result["block_count"] = count
        if kind == 1:
            rp = np.full(mp + 1, len(parts["values"]), dtype=np.int32)
            rp[:m+1] = parts["rowptr"]
            result.update({"rowptr": rp, "columns": parts["columns"], "values": parts["values"],
                           "block_rowptr": parts["block_rowptr"], "block_columns": parts["block_columns"],
                           "block_values": parts["block_values"].ravel()})
        elif kind == 2:
            result.update({"rowptr": parts["block_rowptr"], "columns": parts["block_columns"], "values": parts["block_values"].ravel()})
        else:
            slots = int(np.diff(parts["block_rowptr"]).max())
            columns = np.full((mp // block, slots), -1, dtype=np.int32)
            values = np.zeros((mp, slots * block), dtype=np.float32)
            for index, (row, column, tile) in enumerate(zip(parts["block_rows"], parts["block_columns"], parts["block_values"])):
                slot = index - int(parts["block_rowptr"][row])
                columns[row, slot] = column
                values[row*block:(row+1)*block, slot*block:(slot+1)*block] = tile
            result.update({"columns": columns.ravel(), "values": values.ravel(), "ell_columns": slots * block})
    return result


def reconstruct_format(formatted):
    """Decode source storage without relying on its preparation implementation."""
    from spmm_reference import reconstruct
    m, k = formatted["shape"]
    mp, kp = formatted["padded_shape"]
    kind, block = formatted["kind"], formatted["block"]
    if kind in (0, 1):
        parts = {"shape": (mp, kp), "block_size": block, "rowptr": formatted["rowptr"],
                 "columns": formatted["columns"], "values": formatted["values"],
                 "block_rows": np.repeat(np.arange(mp // block), np.diff(formatted["block_rowptr"])) if kind == 1 else [],
                 "block_columns": formatted["block_columns"],
                 "block_values": formatted["block_values"].reshape(-1, block, block)}
        matrix = reconstruct(parts)
    elif kind == 4:
        matrix = formatted["values"].reshape(mp, kp).copy()
    else:
        matrix = np.zeros((mp, kp), dtype=np.float32)
        if kind == 2:
            for row in range(mp // block):
                for index in range(formatted["rowptr"][row], formatted["rowptr"][row+1]):
                    column = int(formatted["columns"][index])
                    tile = formatted["values"][index*block**2:(index+1)*block**2].reshape(block, block)
                    matrix[row*block:(row+1)*block, column*block:(column+1)*block] = tile
        else:
            slots = formatted["ell_columns"] // block
            columns = formatted["columns"].reshape(mp // block, slots)
            values = formatted["values"].reshape(mp, slots * block)
            for row in range(mp // block):
                for slot, column in enumerate(columns[row]):
                    if column >= 0:
                        matrix[row*block:(row+1)*block, column*block:(column+1)*block] = values[row*block:(row+1)*block, slot*block:(slot+1)*block]
    original = np.empty((m, k), dtype=np.float32)
    original[formatted["permutation"]] = matrix[:m, :k]
    return original
