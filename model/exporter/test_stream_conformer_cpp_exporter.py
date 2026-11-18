"""Manifest configuration tests without loading a trained checkpoint."""
from types import SimpleNamespace
import unittest

from exporter.stream_conformer_cpp_exporter import StreamConformerCppExporter


class RecognitionWeightTest(unittest.TestCase):
    def make_exporter(self, config):
        exporter = StreamConformerCppExporter.__new__(StreamConformerCppExporter)
        exporter.model_config = {'arch': 'stream_conformer', 'vocab_path': 'data/mercury.vocab', **config}
        exporter.model = SimpleNamespace(vocab_size=1024, ctc_time_reduction=2,
                                        phn_decoder=SimpleNamespace(tout_emb=SimpleNamespace(out_features=47)))
        exporter.inference_wrapper = SimpleNamespace(blank_id=0, bos_id=2, eos_id=3)
        return exporter

    def test_default_and_explicit_weights(self):
        for config, expected in [({}, 0.5), ({'recognition_weight': 0}, 0), ({'recognition_weight': 1}, 1)]:
            with self.subTest(config=config):
                manifest = self.make_exporter(config).build_manifest({})['model_path']
                self.assertEqual(manifest['recognition_weight'], expected)

    def test_proper_noun_floor(self):
        for config, expected in [({}, 90.0), ({'proper_noun_floor': 0}, 0.0), ({'proper_noun_floor': 88}, 88.0)]:
            with self.subTest(config=config):
                manifest = self.make_exporter(config).build_manifest({})['model_path']
                self.assertEqual(manifest['proper_noun_floor'], expected)
                self.assertEqual(manifest['proper_noun_list'], 'proper_nouns.txt')
                self.assertEqual(manifest['ungraded_word_list'], 'ungraded_words.txt')
        for value in (-1, 101, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_exporter({'proper_noun_floor': value}).build_manifest({})

    def test_invalid_weights(self):
        for value in (-0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_exporter({'recognition_weight': value}).build_manifest({})


if __name__ == '__main__':
    unittest.main()
