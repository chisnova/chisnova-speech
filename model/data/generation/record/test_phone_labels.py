"""Transport source pronunciations into actual training records without audio inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phone_labels import load_phone_labels


class PhoneLabelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'train.phone_labels.jsonl'
        self.wave = self.root / 'unused.wav'
        self.wave.write_bytes(b'synthetic feature fixture')
        self.index = {p: i for i, p in enumerate(['<pad>', '-', '<s>', '</s>', '_', 'AH', 'EY', 'EH', 'M'])}
        self.row = dict(schema_version=1, id='sample', text='a m', phones=['EY', '_', 'EH', 'M'],
                        source_jsonl='demo.jsonl', source_line_number=1,
                        source_jsonl_sha256='a' * 64, source_phone_vocab_sha256='b' * 64)
        self.row['audio_sha256'] = hashlib.sha256(self.wave.read_bytes()).hexdigest()

    def load(self):
        return load_phone_labels(self.path, {'sample': 'a m'}, self.index, {'sample': self.wave})

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_symbol_mapping_and_provenance(self):
        self.write([self.row])
        labels, provenance = self.load()
        self.assertEqual(labels, {'sample': [2, 6, 4, 7, 8, 3]})
        self.assertEqual(provenance['utterances'], 1)
        self.assertEqual(len(provenance['sha256']), 64)

    def test_invalid_or_mismatched_labels_fail_before_generation(self):
        variants = [dict(id='unknown'), dict(text='am'), dict(phones=['AH', '<s>']),
                    dict(phones=['EY', '_', '_', 'M']), dict(phones=['BOGUS']),
                    dict(source_jsonl_sha256='bad'), dict(schema_version=True)]
        for changes in variants:
            with self.subTest(changes=changes):
                row = copy.deepcopy(self.row)
                row.update(changes)
                self.write([row])
                with self.assertRaises(ValueError):
                    self.load()
        for rows in ([], [self.row, self.row]):
            self.write(rows)
            with self.assertRaises(ValueError):
                self.load()

    def test_stale_audio_sidecar_is_rejected(self):
        self.write([self.row])
        self.wave.write_bytes(b'regenerated recording')
        with self.assertRaisesRegex(ValueError, 'phone label audio mismatch'):
            self.load()

    def test_lmdb_uses_override_and_preserves_word_targets(self):
        # Audio features are fixed; the real serializer and LMDB label path run.
        import lmdb
        import numpy as np
        import pickle
        from data_gen import AsrDataGen, phone_dict
        self.write([self.row])
        wave_list = self.root / 'train.wlist'
        text_list = self.root / 'train.lab'
        wave_list.write_text('sample\t' + str(self.wave) + '\n')
        text_list.write_text('sample\ta m\n')
        vocab = Path(__file__).resolve().parents[2] / 'mercury.model'
        gen = AsrDataGen(str(wave_list), str(text_list), str(vocab), phone_labels=self.path)
        gen.getFbankFeature = lambda path: np.zeros((3, 80), dtype=np.float32)
        output = self.root / 'lmdb'
        gen.saveLMDB(str(output), map_size=1 << 20)
        env = lmdb.open(str(output / 'shard_00000.lmdb'), readonly=True, lock=False)
        try:
            with env.begin(write=False) as txn:
                record = pickle.loads(txn.get(b'sample'))
            self.assertEqual(record['phone_ids'].tolist(), [2] + [phone_dict[p] for p in self.row['phones']] + [3])
            self.assertEqual(record['word_ids'].tolist(), gen.getWordpieceFeature('a m'))
            self.assertEqual(gen.makePhoneFeature('a m')[1], phone_dict['AH'])
            self.assertIn('phone_labels', json.loads((output / 'meta.json').read_text()))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
