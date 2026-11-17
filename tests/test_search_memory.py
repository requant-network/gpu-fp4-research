"""Faiss scratch reservation must cover two full distance tiles plus margin."""
import importlib.util
import unittest


class SearchMemoryTests(unittest.TestCase):
    def test_distance_tiles_and_small_case_minimum(self):
        self.assertIsNotNone(importlib.util.find_spec('search_memory'),'Shape-aware Faiss reservation is missing')
        from search_memory import faiss_reservation_bytes
        self.assertEqual(faiss_reservation_bytes(16384,1),64<<20)
        self.assertEqual(faiss_reservation_bytes(262144,128),2*262144*128*4+(16<<20))
        self.assertEqual(faiss_reservation_bytes(1000000,128),2*1000000*128*4+(16<<20))
        self.assertLess(faiss_reservation_bytes(1000000,128),1536<<20)
        with self.assertRaises(ValueError):faiss_reservation_bytes(0,128)
        with self.assertRaises(ValueError):faiss_reservation_bytes(1000000,0)


if __name__=='__main__':unittest.main()
