"""CPU-only checks for exact mixing, corpus continuity and source exhaustion."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "mixed_batch_loader", Path(__file__).parents[1] / "mixed_batch_loader.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MixedBatchLoader = module.MixedBatchLoader


class Source:
    def __init__(self, name, count):
        self.name, self.count, self.epoch = name, count, 0
        self.closed = False

    def __len__(self):
        return self.count

    def iter_prefetched_batches(self, num_workers=None):
        for i in range(self.count):
            yield self.name, self.epoch, i

    def reshuffle_for_next_epoch(self):
        self.epoch += 1

    def close(self):
        self.closed = True


class MixedBatchTest(unittest.TestCase):
    def test_exact_blocks_continuity_and_reshuffling(self):
        sources = [Source('train', 100), Source('mercury', 3)]
        mixed = MixedBatchLoader(sources, batches_per_epoch=10, seed=42)
        batches = []
        for _ in range(3):
            mixed.reshuffle_for_next_epoch()
            batches.extend(mixed.iter_prefetched_batches())
        for start in range(0, 30, 5):
            self.assertEqual(sum(b[0] == 'train' for b in batches[start:start+5]), 4)
        self.assertEqual([b[2] for b in batches if b[0] == 'train'], list(range(24)))
        self.assertEqual([(b[1], b[2]) for b in batches if b[0] == 'mercury'],
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        self.assertGreater(len({tuple(b[0] for b in batches[i:i+5]) for i in range(0, 30, 5)}), 1)
        self.assertEqual(mixed.source_counts, [24, 6])
        mixed.close()
        self.assertTrue(all(s.closed for s in sources))

    def test_reproducible_and_invalid_configuration(self):
        def run():
            return list(MixedBatchLoader([Source('a', 8), Source('b', 2)],
                                        batches_per_epoch=10, seed=9).iter_prefetched_batches())
        self.assertEqual(run(), run())
        with self.assertRaises(ValueError):
            MixedBatchLoader([Source('a', 0), Source('b', 1)])
        with self.assertRaises(ValueError):
            MixedBatchLoader([Source('a', 1), Source('b', 1)], batches_per_epoch=7)


if __name__ == '__main__':
    unittest.main()
