"""Independent whole-pipeline quality and actual Tensor FFT stage checks."""
import numpy as np

from fft_reference import stage_input, stage_output
from reference import digest
from spmm_preflight import encode


def quality(output, spectrum, filtered, reference, parameters, input_norm):
    result = {"finite_output": bool(np.isfinite(output).all()), "finite_spectrum": bool(np.isfinite(spectrum).all()),
              "finite_filtered": bool(np.isfinite(filtered).all()),
              "nonfinite_output_entries": int(np.count_nonzero(~np.isfinite(output)))}
    for name, actual, gold in (("output", output, reference[2]), ("spectrum", spectrum, reference[0]), ("filtered", filtered, reference[1])):
        if np.isfinite(actual).all():
            difference = actual.astype(np.complex128)-gold
            denominator = float(np.linalg.norm(gold))
            result[f"{name}_relative_l2"] = float(np.linalg.norm(difference)/max(denominator, 1e-300))
            result[f"{name}_max_abs_error"] = float(np.max(np.abs(difference)))
        else:
            result[f"{name}_relative_l2"] = result[f"{name}_max_abs_error"] = None
    result["output_error_over_windowed_input"] = (float(np.linalg.norm(output-reference[2])/max(input_norm, 1e-300))
                                                   if result["finite_output"] else None)
    amplitude, weak = parameters["weak_amplitude"], parameters["weak_bin"]
    for key in ("weak_bin_relative_error", "quiet_bin_artifact_over_weak", "output_error_over_weak", "input_weak_amplitude_ratio"):
        result[key] = None
    if amplitude is not None:
        batch, n = output.shape
        scale = amplitude*n
        result["input_weak_amplitude_ratio"] = float(np.mean(abs(reference[0][:, weak]))/scale)
        if result["finite_spectrum"]:
            result["weak_bin_relative_error"] = float(np.max(abs(spectrum[:, weak]-reference[0][:, weak]))/scale)
            quiet = np.ones(n, dtype=bool)
            quiet[parameters["strong_bin"]-1:parameters["strong_bin"]+2] = False
            quiet[weak] = False
            result["quiet_bin_artifact_over_weak"] = float(np.max(abs(spectrum[:, quiet]-reference[0][:, quiet]))/scale)
        if result["finite_output"]:
            result["output_error_over_weak"] = float(np.linalg.norm(output-reference[2])/(amplitude*np.sqrt(batch*n)))
    return result


def check_output(output, reference_output, method, input_norm):
    if not np.isfinite(output).all():
        # A genuine FP16 transform overflow is a retained failed-quality outcome.
        if method["backend"] != 0 or method["precision"] != 2:
            raise AssertionError("Unexpected nonfinite native FFT output")
        return
    limit = (.08 if method["precision"] in (2, 3) else .02) if method["backend"] == 0 else (.08 if method["precision"] == 2 else .02)
    if method["precision"] == 0 or method["backend"] == 2:
        limit = 2e-5
    if method["backend"] == 0 and method["precision"] == 1:
        limit = 2e-6
    error = float(np.linalg.norm(output.astype(np.complex128)-reference_output))
    if error > limit*input_norm+1e-30:
        raise AssertionError(f"Native pipeline fails input-normalized arithmetic screening: {error} > {limit*input_norm}")


def tensor_stages(plan, method, coefficients, arrays, prefix):
    """Full runtime byte/arithmetic checks, plus replayable selected-column witnesses."""
    metadata = plan.metadata()
    batch, n = plan.shape
    precision = method["precision"]
    records = []
    encoded_basis = plan.snapshot(8)
    expected_basis = encode(coefficients["basis"], precision)[0]
    if encoded_basis.tobytes() != expected_basis.tobytes():
        raise AssertionError("GPU basis encoding differs from the independently rounded bytes")
    for inverse in (False, True):
        plan.call("ff_tc_begin", int(inverse))
        for index, stage in enumerate(metadata["stages"]):
            plan.call("ff_tc_step", index, int(inverse))
            before = plan.snapshot(9).view(np.complex64).reshape(batch, n)
            radix, previous = stage["radix"], stage["previous"]
            start = stage["phase_offset"]
            phase = coefficients[f"phases{method['radix']}"][start:start+radix*previous].reshape(radix, previous)
            packed_float = stage_input(before, previous, radix, phase, inverse)
            expected, decoded = encode(np.ascontiguousarray(packed_float.T), precision)
            raw = plan.snapshot(10)
            if raw.tobytes() != expected.tobytes():
                raise AssertionError("Tensor FFT phase/packing/rounding byte mismatch")
            size = stage["tile"]
            offset = stage["basis_offset"]+1792*inverse
            weight = expected_basis[offset:offset+size*size]
            if precision == 1:
                a = weight.astype(np.float32)
            elif precision == 2:
                a = (weight.astype(np.uint32)<<16).view(np.float32)
            else:
                a = weight
            a = a.reshape(size, size, order="F").astype(np.float64)
            b = decoded.T.astype(np.float64)
            product = plan.snapshot(11).view(np.float32).reshape(size, stage["padded"], order="F")
            gold = a@b
            bound = 2*size*np.finfo(np.float32).eps*(abs(a)@abs(b))+1e-35
            if not np.all(abs(product-gold) <= bound):
                raise AssertionError("Tensor FFT GEMM fails the independent FP64 accumulation envelope")
            after = plan.snapshot(12).view(np.complex64).reshape(batch, n)
            if not np.array_equal(after, stage_output(product, batch, n, previous, radix)):
                raise AssertionError("Tensor FFT unpacking/normalization mismatch")
            if precision and not stage["lt_numerical_flags"] & 2:
                raise AssertionError("Low-precision stage has no HMMA implementation flag")
            columns = np.unique(np.linspace(0, stage["columns"]-1, min(64, stage["columns"]), dtype=np.int64))
            t = np.arange(radix)[:, None]
            j = columns % previous
            g = (columns//previous) % (n//(radix*previous))
            bi = columns//(n//radix)
            positions = g[None, :]*radix*previous+j[None, :]+t*previous
            key = f"{prefix}_d{int(inverse)}_s{index}"
            arrays[key+"_columns"] = columns
            arrays[key+"_before"] = before[bi[None, :], positions].copy()
            arrays[key+"_packed"] = expected.reshape(stage["padded"], size)[columns].T.copy()
            arrays[key+"_product"] = product[:, columns].copy()
            arrays[key+"_after"] = after[bi[None, :], positions].copy()
            records.append({"prefix": key, "inverse": inverse, "index": index, "geometry": stage,
                            "before_sha256": digest(before), "packed_sha256": digest(raw),
                            "product_sha256": digest(np.ascontiguousarray(product.T)), "after_sha256": digest(after),
                            "checked_product_entries": product.size, "checked_packed_bytes": raw.size,
                            "max_product_abs_error": float(np.max(abs(product-gold))),
                            "full_columns_retained": len(columns) == stage["columns"]})
    return records
