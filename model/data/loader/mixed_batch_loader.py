"""Mix complete source batches in shuffled, exact-ratio blocks.

An epoch is a bounded training window. Source iterators continue across windows;
only an exhausted source is reshuffled, so a large corpus is not restarted at
every adaptation epoch. A source batch remains intact (including its transforms).
"""
import random


class MixedBatchLoader:
    def __init__(self, loaders, batches_per_source=(4, 1), batches_per_epoch=1000,
                 seed=0):
        self.loaders = list(loaders)
        self.ratio = tuple(batches_per_source)
        if (not self.loaders or len(self.loaders) != len(self.ratio)
                or any(type(n) is not int or n <= 0 for n in self.ratio)):
            raise ValueError("Each source needs a positive integer batch count")
        if (type(batches_per_epoch) is not int or batches_per_epoch <= 0
                or batches_per_epoch % sum(self.ratio)):
            raise ValueError("batches_per_epoch must be a positive multiple of the ratio sum")
        if any(len(loader) == 0 for loader in self.loaders):
            raise ValueError("Cannot mix an empty source")
        self.num_batches = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self._iterators = [None] * len(self.loaders)
        self.source_counts = [0] * len(self.loaders)

    def __len__(self):
        return self.num_batches

    def reshuffle_for_next_epoch(self):
        self.epoch += 1

    def iter_prefetched_batches(self, num_workers=None):
        rng = random.Random(self.seed + self.epoch)
        block = [i for i, count in enumerate(self.ratio) for _ in range(count)]
        counts = [0] * len(self.loaders)
        for block_index in range(self.num_batches // len(block)):
            order = block.copy()
            rng.shuffle(order)
            if block_index == 0:
                print(f"[mix] epoch {self.epoch}: first block source order={order} (0=train, 1=adaptation)", flush=True)
            for source in order:
                loader = self.loaders[source]
                if self._iterators[source] is None:
                    self._iterators[source] = iter(loader.iter_prefetched_batches(num_workers))
                try:
                    batch = next(self._iterators[source])
                except StopIteration:
                    loader.reshuffle_for_next_epoch()
                    self._iterators[source] = iter(loader.iter_prefetched_batches(num_workers))
                    try:
                        batch = next(self._iterators[source])
                    except StopIteration as exc:
                        raise RuntimeError(f"Source {source} yielded no usable batches") from exc
                counts[source] += 1
                self.source_counts[source] += 1
                yield batch
            if block_index == 0:
                print(f"[mix] epoch {self.epoch}: first block consumed, batches={counts}", flush=True)
        print(f"[mix] epoch {self.epoch}: batches={counts}, total={self.source_counts}", flush=True)

    def close(self):
        for iterator in self._iterators:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
        self._iterators = [None] * len(self.loaders)
        for loader in self.loaders:
            loader.close()
