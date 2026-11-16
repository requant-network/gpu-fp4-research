"""Independent CPU storage/encoding, product and quality checks for native SpMM."""
import numpy as np

from spmm_preflight import encode


def sparse_product(source, shape, right):
    m, k = shape
    if right.ndim != 2 or right.shape[0] != k:
        raise ValueError("Invalid sparse product shape")
    right = np.asarray(right, dtype=np.float64)
    values = source["values"].astype(np.float64)
    output = np.zeros((m, right.shape[1]), dtype=np.float64)
    for row in range(m):
        start, end = source["rowptr"][row:row+2]
        output[row] = values[start:end] @ right[source["columns"][start:end]]
    return output


def whole_product(source, shape, right, precision):
    rounded = dict(source, values=encode(source["values"], precision)[1])
    return sparse_product(rounded, shape, encode(right, precision)[1])


def hybrid_product(formatted, right, precision):
    mp, kp = formatted["padded_shape"]
    m, k = formatted["shape"]
    block = formatted["block"]
    padded = np.zeros((kp, right.shape[1]), dtype=np.float32)
    padded[:k] = right
    result = sparse_product(formatted, (mp, kp), padded)
    tiles = encode(formatted["block_values"], precision)[1].reshape(-1, block, block).astype(np.float64)
    rounded_b = encode(padded, precision)[1].astype(np.float64)
    for row in range(mp // block):
        for index in range(formatted["block_rowptr"][row], formatted["block_rowptr"][row+1]):
            column = int(formatted["block_columns"][index])
            result[row*block:(row+1)*block] += tiles[index] @ rounded_b[column*block:(column+1)*block]
    output = np.empty((m, right.shape[1]), dtype=np.float64)
    output[formatted["permutation"]] = result[:m]
    return output


def expected_buffers(formatted, right, precision, layout):
    mp, kp = formatted["padded_shape"]
    m, k = formatted["shape"]
    kind, block = formatted["kind"], formatted["block"]
    b_source = np.zeros((kp, right.shape[1]), dtype=np.float32)
    b_source[:k] = right
    if kind == 1:
        a = formatted["block_values"].reshape(-1, block, block).transpose(0, 2, 1).copy().ravel()
        b = np.zeros((formatted["block_count"], right.shape[1], block), dtype=np.float32)
        for index, column in enumerate(formatted["block_columns"]):
            b[index] = b_source[column*block:(column+1)*block].T
        b = b.ravel()
    else:
        a = formatted["values"]
        b = b_source.T.copy().ravel() if layout == 1 else b_source.ravel()
    values = (encode(a, precision)[0], encode(b, precision)[0], formatted["values"], b_source.ravel())
    return [np.ascontiguousarray(array).view(np.uint8).ravel() for array in values]


def metrics(output, original, model, opaque_fp32_ell=False):
    if output.shape != original.shape or model.shape != output.shape or output.dtype != np.float32:
        raise ValueError("Invalid SpMM output or reference shape/type")
    if not np.isfinite(output).all():
        raise AssertionError("Nonfinite SpMM output")
    difference = output.astype(np.float64) - original
    norm = float(np.linalg.norm(original))
    error = float(np.linalg.norm(difference))
    relative = error / norm if norm else (0. if error == 0 else float("inf"))
    model_error = float(np.linalg.norm(output.astype(np.float64) - model))
    model_norm = float(np.linalg.norm(model))
    model_relative = model_error / model_norm if model_norm else (0. if model_error == 0 else float("inf"))
    per_column = np.linalg.norm(difference, axis=0)
    column_norm = np.linalg.norm(original, axis=0)
    zero = column_norm == 0
    if np.any(per_column[zero] != 0):
        raise AssertionError("An exact-zero output column became nonzero")
    ratios = np.divide(per_column, column_norm, out=np.zeros_like(column_norm), where=~zero)
    if not opaque_fp32_ell and (not np.isfinite(model_relative) or model_relative > 2e-5):
        raise AssertionError(f"SpMM differs from its independent rounded-input product: {model_relative}")
    if not np.isfinite(relative) or not np.isfinite(ratios).all() or not np.isfinite(model_relative):
        raise AssertionError("Nonfinite SpMM error metric")
    return {"relative_frobenius_error": relative, "max_relative_column_error": float(ratios.max()),
            "relative_error_vs_product_model": model_relative,
            "product_model": "Opaque approximate FP32-storage Blocked-ELL; original FP64-product quality measured, internal quantizer not certified" if opaque_fp32_ell else "FP64 product of independently rounded inputs; hybrid sparse remainder stays FP32",
            "arithmetic_model_checked": not opaque_fp32_ell,
            "exact_zero_columns_checked": int(zero.sum())}
