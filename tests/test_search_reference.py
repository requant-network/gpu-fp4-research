import unittest

import numpy as np

from search_reference import evaluate, reference_search, refine


class SearchReferenceTests(unittest.TestCase):
    def test_filter_chunk_boundaries_ties_and_candidate_refinement(self):
        data = np.array([[0, 0], [1, 0], [-1, 0], [4, 4], [0, 0]], dtype=np.float32)
        queries = np.array([[0, 0], [2, 0]], dtype=np.float32)
        allowed = np.array([False, True, True, True, True])
        gold_distances, gold_ids = reference_search(data, queries, 3, allowed, chunk=2)
        np.testing.assert_array_equal(gold_ids, [[4, 1, 2], [1, 4, 2]])
        np.testing.assert_array_equal(gold_distances, [[0, 1, 1], [1, 4, 9]])
        candidates = np.array([[2, 4, 1, 1, 0], [3, 2, 4, 1, -1]], dtype=np.int64)
        distances, ids = refine(data, queries, candidates, 3, allowed)
        np.testing.assert_array_equal(ids, gold_ids)
        np.testing.assert_array_equal(distances, gold_distances)
        result, _ = evaluate(data, queries, ids, gold_distances, gold_ids, allowed)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['tie_aware_recall'], 1)

    def test_tie_aware_recall_does_not_reward_duplicates_or_filtered_ids(self):
        data = np.array([[1], [-1], [1]], dtype=np.float32)
        query = np.zeros((1, 1), dtype=np.float32)
        distances, ids = reference_search(data, query, 1)
        result, _ = evaluate(data, query, np.array([[1]], dtype=np.int64), distances, ids)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['tie_aware_recall'], 1)
        distances, ids = reference_search(data, query, 2)
        result, _ = evaluate(data, query, np.array([[0, 0, 9]], dtype=np.int64), distances, ids)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['tie_aware_recall'], .5)
        self.assertEqual(result['duplicate_ids'], 1)
        self.assertEqual(result['invalid_or_filtered_ids'], 1)

    def test_empty_filter_missing_results_and_invalid_inputs(self):
        data = np.ones((3, 2), dtype=np.float32)
        query = np.zeros((1, 2), dtype=np.float32)
        allowed = np.zeros(3, dtype=bool)
        distances, ids = reference_search(data, query, 4, allowed)
        self.assertTrue(np.isinf(distances).all())
        np.testing.assert_array_equal(ids, -np.ones((1, 4), dtype=np.int64))
        for k in (0, -1):
            with self.assertRaises(ValueError):
                reference_search(data, query, k)
        with self.assertRaises(ValueError):
            reference_search(data.astype(np.float64), query, 1)


if __name__ == '__main__':
    unittest.main()
