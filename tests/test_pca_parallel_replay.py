"""Partition merging rejects double counting and mismatched source/data epochs."""
import copy
import importlib.util
import unittest


class PcaParallelReplayTests(unittest.TestCase):
    def test_disjoint_verified_partitions_merge_without_double_counting(self):
        self.assertIsNotNone(importlib.util.find_spec('pca_parallel_replay'),'Parallel PCA replay is missing')
        from pca_parallel_replay import merge_replays
        def part(identity,checksum)->dict:
            return dict(scope='Bounded PCA subset replay; not full study completion',
                        counts=dict(cases=1,dataset_collections=1,collection_directories=1,environment_snapshots=1,guarded_cases=0),
                        producer_source_sha256={'producer.py':'b'*64},verifier_source_sha256={'checker.py':'c'*64},
                        binary_sha256='d'*64,collection_sha256={identity:checksum},producer_epoch_collections={identity:'original'},
                        additional_producer_source_sha256={},limits='Unit-test fixture, not experimental output')
        a=part('pca-full-a/x-collection.json','a'*64)
        b=part('pca-full-b/y-collection.json','e'*64)
        expected={**a['collection_sha256'],**b['collection_sha256']}
        combined=merge_replays([a,b],expected,complete=False)
        self.assertEqual(combined['counts']['cases'],2)
        self.assertEqual(combined['counts']['collection_directories'],2)
        with self.assertRaises(AssertionError):merge_replays([a,a],a['collection_sha256'],complete=False)
        wrong=copy.deepcopy(b);wrong['verifier_source_sha256']['checker.py']='f'*64
        with self.assertRaises(AssertionError):merge_replays([a,wrong],expected,complete=False)


if __name__=='__main__':unittest.main()
