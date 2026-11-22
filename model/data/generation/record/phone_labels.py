"""Validated pronunciation sidecars; no audio/ML dependencies."""
import hashlib
import json
from pathlib import Path
import re


def audio_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_phone_labels(path, expected_texts, phone_index, wave_paths):
    """Require exact utterance coverage before any LMDB output is opened."""
    raw = Path(path).read_bytes()
    targets = {}
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get('schema_version')) is not int or row['schema_version'] != 1:
                raise ValueError('unsupported phone label schema')
            identity = row.get('id')
            if not isinstance(identity, str) or identity not in expected_texts or identity in targets:
                raise ValueError('unknown or duplicate phone label ID: {!r}'.format(identity))
            if row.get('text') != expected_texts[identity]:
                raise ValueError('phone label text mismatch: ' + identity)
            phones = row.get('phones')
            if not isinstance(phones, list) or not phones or any(
                    not isinstance(p, str) or p not in phone_index or phone_index[p] < 4 for p in phones):
                raise ValueError('invalid phone symbols: ' + identity)
            if phones[0] == '_' or phones[-1] == '_' or any(a == b == '_' for a, b in zip(phones, phones[1:])):
                raise ValueError('invalid phone boundaries: ' + identity)
            if not isinstance(row.get('source_jsonl'), str) or not row['source_jsonl'] or \
                    type(row.get('source_line_number')) is not int or row['source_line_number'] < 1:
                raise ValueError('missing source pronunciation provenance: ' + identity)
            for field in ('source_jsonl_sha256', 'source_phone_vocab_sha256', 'audio_sha256'):
                if not isinstance(row.get(field), str) or not re.fullmatch(r'[0-9a-f]{64}', row[field]):
                    raise ValueError('invalid ' + field + ': ' + identity)
            if audio_digest(wave_paths[identity]) != row['audio_sha256']:
                raise ValueError('phone label audio mismatch: ' + identity)
            targets[identity] = [2] + [phone_index[p] for p in phones] + [3]
        except (ValueError, TypeError) as exc:
            raise ValueError('{}:{}: {}'.format(path, number, exc)) from exc
    missing = set(expected_texts) - set(targets)
    if missing:
        raise ValueError('Missing phone labels for {} utterances: {}'.format(len(missing), ', '.join(sorted(missing)[:5])))
    return targets, {'file': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                     'schema_version': 1, 'utterances': len(targets)}
