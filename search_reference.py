"""Independent filtered nearest-neighbour and candidate-refinement contracts."""
import numpy as np


def validate(database, queries, k, allowed):
    if database.dtype != np.float32 or queries.dtype != np.float32 or database.ndim != 2 or queries.ndim != 2:
        raise ValueError("Expected two FP32 matrices")
    if not len(database) or not len(queries) or database.shape[1] != queries.shape[1] or database.shape[1] == 0:
        raise ValueError("Invalid vector-search dimensions")
    if not np.isfinite(database).all() or not np.isfinite(queries).all() or not isinstance(k, int) or k < 1:
        raise ValueError("Finite vectors and positive integer k are required")
    if allowed is None:
        allowed = np.ones(len(database), dtype=bool)
    if allowed.dtype != np.bool_ or allowed.shape != (len(database),):
        raise ValueError("Expected a global boolean allow-mask")
    return allowed


def reference_search(database, queries, k, allowed=None, chunk=4096):
    """Direct FP64 squared differences, streamed over the database; ID breaks ties."""
    allowed = validate(database, queries, k, allowed)
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError("Chunk size must be a positive integer")
    result_ids = np.full((len(queries), k), -1, dtype=np.int64)
    result_distances = np.full((len(queries), k), np.inf, dtype=np.float64)
    for qi, query in enumerate(queries):
        ids = np.empty(0, dtype=np.int64)
        distances = np.empty(0, dtype=np.float64)
        for start in range(0, len(database), chunk):
            selected = np.flatnonzero(allowed[start:start+chunk])+start
            difference = database[selected].astype(np.float64)-query.astype(np.float64)
            squared = np.einsum("ij,ij->i", difference, difference)
            candidate_ids = np.concatenate((ids, selected))
            candidate_distances = np.concatenate((distances, squared))
            order = np.lexsort((candidate_ids, candidate_distances))[:k]
            ids, distances = candidate_ids[order], candidate_distances[order]
        result_ids[qi, :len(ids)] = ids
        result_distances[qi, :len(ids)] = distances
    return result_distances, result_ids


def evaluate(database, queries, candidate_ids, truth_distances, truth_ids, allowed=None):
    """Strict recall plus tie-aware recall, using independently recomputed distances."""
    if candidate_ids.dtype != np.int64 or candidate_ids.ndim != 2 or candidate_ids.shape[0] != len(queries):
        raise ValueError("Expected query-by-candidate int64 IDs")
    k = truth_ids.shape[1]
    allowed = validate(database, queries, k, allowed)
    if truth_ids.shape != (len(queries), k) or truth_distances.shape != truth_ids.shape:
        raise ValueError("Invalid ground-truth shape")
    strict, tied, valid_counts, duplicates, invalid = [], [], [], 0, 0
    distances = np.full(candidate_ids.shape, np.inf, dtype=np.float64)
    for qi, ids in enumerate(candidate_ids):
        valid = (ids >= 0) & (ids < len(database))
        positions = np.flatnonzero(valid)
        valid[positions] &= allowed[ids[positions]]
        invalid += int(np.count_nonzero((ids != -1) & ~valid))
        selected = ids[valid]
        unique = np.unique(selected)
        duplicates += len(selected)-len(unique)
        difference = database[selected].astype(np.float64)-queries[qi].astype(np.float64)
        distances[qi, valid] = np.einsum("ij,ij->i", difference, difference)
        truth = truth_ids[qi][truth_ids[qi] >= 0]
        count = len(truth)
        valid_counts.append(count)
        strict.append(len(np.intersect1d(unique, truth))/count if count else 1.)
        cutoff = truth_distances[qi, count-1] if count else -1.
        accepted = np.unique(ids[(distances[qi] <= cutoff) & valid])
        tied.append(min(len(accepted), count)/count if count else 1.)
    return {"recall": float(np.mean(strict)), "tie_aware_recall": float(np.mean(tied)),
            "duplicate_ids": int(duplicates), "invalid_or_filtered_ids": int(invalid),
            "truth_counts": valid_counts}, distances


def refine(database, queries, candidates, k, allowed=None):
    """Exact FP64 candidate rerank with duplicate removal and a global allow-mask."""
    allowed = validate(database, queries, k, allowed)
    if candidates.dtype != np.int64 or candidates.ndim != 2 or candidates.shape[0] != len(queries):
        raise ValueError("Invalid refinement candidate matrix")
    ids = np.full((len(queries), k), -1, dtype=np.int64)
    distances = np.full(ids.shape, np.inf, dtype=np.float64)
    for qi, row in enumerate(candidates):
        selected = np.unique(row[(row >= 0) & (row < len(database))])
        selected = selected[allowed[selected]]
        difference = database[selected].astype(np.float64)-queries[qi].astype(np.float64)
        squared = np.einsum("ij,ij->i", difference, difference)
        order = np.lexsort((selected, squared))[:k]
        ids[qi, :len(order)], distances[qi, :len(order)] = selected[order], squared[order]
    return distances, ids
