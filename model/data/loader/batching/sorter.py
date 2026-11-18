"""Sorter — 발화를 셔플하고 chunk 안에서 길이순으로 정렬한다."""
import numpy as np


class Sorter:
    """발화 순서 결정 — 셔플 -> chunk 로 분할 -> chunk 안에서 feat_len 오름차순.

    전체를 길이순으로 정렬하면 패딩은 최소가 되지만 에폭마다 배치 구성이 똑같아
    학습에 나쁘고, 셔플만 하면 배치 안에 길이가 뒤섞여 패딩이 폭증한다.
    chunk 단위 정렬이 그 절충이다 — 배치 구성은 에폭마다 달라지면서(셔플이 먼저
    돌므로) 배치 안 길이는 고르게 모인다.

    chunk 가 배치 크기에 비해 작으면 한 배치가 chunk 를 거의 다 먹어 정렬 효과가
    사라진다. AsrDataLoader 의 chunk="auto" 가 resolve_auto_chunk 로 잡아준다.

    이 기법의 표준 명칭 (검색용 — 다시 조사하지 않도록 적어 둔다)
      sortish sampling      fastai SortishSampler, HF legacy seq2seq (원조 계보)
      length grouping       HF transformers Trainer 의 group_by_length=True
                            (chunk 를 megabatch 라 부른다)
      pool / pooling        torchtext BucketIterator.pool, OpenNMT-py pool_factor
    "bucketing" 은 문헌에서 두 가지를 가리키므로 주의 — torchtext 계열은 이 방식이지만
    TensorFlow/NeMo/SpeechBrain 의 bucket 은 절대 길이 경계로 나눈 정적 버킷이다.

    음성 프레임워크는 이 방식을 쓰지 않는다는 점을 알아 둘 것. ESPnet/SpeechBrain/
    NeMo/lhotse/fairseq 는 전부 고정 batch_size 를 피하고 프레임·토큰 총량으로 배치
    크기를 가변시킨다(동적 배칭). chunk 정렬은 batch_size 고정을 전제하는 텍스트 계열
    기법이라, 우리처럼 고정으로 쓰면 긴 발화가 몰린 배치가 그대로 메모리 피크가 된다
    — OOM 은 AsrDataLoader 의 max_feat_len 으로 막아야 한다.

    그리고 이건 품질이 아니라 속도를 위한 기법이다. Kocmi & Bojar 2017(RANLP)은 NMT
    에서 길이 버킷팅의 BLEU 개선이 표준편차 범위 안이라고 결론냈다.
    """

    MIN_CHUNK = 2048          # chunk 가 1개뿐이면 전역정렬이 되어 무작위성이 사라진다.
                              # HF 도 같은 이유로 megabatch 가 최소 4개는 나오게 잡는다
                              # (mega_batch_mult = min(N // (bs*4), 50)).
    CHUNK_PER_BATCH = 64      # 관습값: fastai·HF 는 50배, torchtext·PyTorch-NLP 는 100배.
                              # 64 는 그 사이다. 어느 쪽도 근거를 문서화하지 않았고(전부
                              # 매직 넘버), 실측상 배치가 chunk 를 절반 이상 먹으면
                              # 배치 안 길이가 뒤섞여 패딩효율이 급락한다.

    def __init__(self, chunk=2048, shuffle=True):
        self.chunk = int(chunk)
        if self.chunk <= 0:
            raise ValueError("chunk 는 양수여야 합니다: {!r}".format(chunk))
        self.shuffle = shuffle

    @classmethod
    def resolve_auto_chunk(cls, batch_size):
        """batch_size 에 맞는 chunk 크기를 정한다 (chunk="auto" 용)."""
        return max(cls.MIN_CHUNK, int(cls.CHUNK_PER_BATCH * batch_size))

    def make_order(self, feat_len, utt_indices, rng=None):
        """utt_indices -> 셔플 후 chunk 안에서 길이 정렬한 발화 순서 배열."""
        rng = rng or np.random.default_rng(0)
        utt_order = np.array(utt_indices, dtype=np.int64, copy=True)
        if self.shuffle:
            rng.shuffle(utt_order)
        for chunk_start in range(0, utt_order.shape[0], self.chunk):
            chunk_indices = utt_order[chunk_start:chunk_start + self.chunk]
            chunk_indices[:] = chunk_indices[
                np.argsort(feat_len[chunk_indices], kind="stable")]
        return utt_order

    def __repr__(self):
        return "Sorter(chunk={:,}, shuffle={})".format(self.chunk, self.shuffle)
