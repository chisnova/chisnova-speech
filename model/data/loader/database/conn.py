"""Conn — 레코드 저장소 커넥션 인터페이스."""


class Conn:
    """저장소 하나로의 커넥션. 카탈로그를 내보내고, utt_id 로 레코드를 꺼낸다.

    배치 쪽(batching 패키지)은 이 인터페이스와 아래 카탈로그 계약만 알고 구현을
    모른다 — LMDB 든 다른 무엇이든 이만큼만 지키면 갈아끼울 수 있다.

    구현이 채워야 하는 것
      feat_dim                      레코드 feat 의 채널 수(int). 배치 바이트 예산이
                                    레코드를 읽기 전에 필요로 하므로 커넥션이 안다.
      load_manifest()               카탈로그 반환 (아래 계약)
      read_record(utt_id, locator)  레코드 dict 반환. 없으면 KeyError.
      close()                       커넥션 정리.

    레코드 계약 (저장소가 무엇이든 이 형태로 돌려줘야 한다)
      {"utt_id": str, "feat": (T, feat_dim) 배열,
       "word_ids": (Lw,) 배열, "phone_ids": (Lp,) 배열}

    카탈로그(manifest) 계약 — load_manifest() 가 돌려주는 객체
      utt_ids   : (N,) bytes 배열   저장소의 키
      locators  : (N,) 정수 배열    read_record 에 그대로 넘길 위치 힌트
      feat_len  : (N,) 정수 배열    프레임 수
      word_len  : (N,) 정수 배열    wordpiece id 개수 (SOS/EOS 포함)
      phone_len : (N,) 정수 배열    phone id 개수 (SOS/EOS 포함)
      __len__() : 발화 수

      역할 분담 — 저장소는 "무엇이 있는지" 목록만 내고, 그 목록을 어떤 순서로
      늘어놓고(shuffle/sort) 어떻게 자를지(chunk/batch)는 전부 batching 이 정한다.
      저장소는 배치를 모르고, 배치는 저장소를 모른다.

      배치는 이 길이 배열만으로 짜고 실제 레코드는 배치가 정해진 뒤에 꺼낸다.
      길이를 알아내려고 저장소를 훑지 않는 것이 이 로더의 전제이므로, 카탈로그는
      저장소가 값싸게 낼 수 있어야 한다(LMDB 는 manifest.jsonl + npz 캐시로 낸다).
      numpy 배열이어야 하는 이유: 927만 발화를 파이썬 list 로 들면 fork 워커에서
      refcount 때문에 COW 가 깨져 메모리가 워커 수만큼 늘어난다.

      여러 번 불려도 같은 카탈로그를 돌려줘야 한다 — 커넥션이 들고 있으라는 뜻이다.
      매번 새로 만들면 호출한 만큼 메모리가 늘어난다(train 기준 1회 370MB).

      이 목록은 하한선이다 — 구현이 편의 메서드를 더 두는 것은 자유지만(LmdbManifest
      의 get_utt_id 처럼) batching 은 위 5개 배열과 __len__ 만 쓴다. 그리고 발화
      하나씩이 아니라 배치/전체 단위로 팬시 인덱싱해서 쓴다(feat_len[utt_order],
      utt_ids[utt_indices]). 행 단위 접근자를 계약에 넣지 않는 이유이자, 카탈로그가
      numpy 배열이어야 하는 두 번째 이유다.

    locator
      저장소가 정한 위치 힌트다 — LMDB 면 샤드 번호, 다른 저장소면 파일 오프셋이나
      파티션 키가 될 수 있다. 카탈로그가 발화마다 실어 나르고, 배치 쪽은 내용을
      해석하지 않고 그대로 넘기기만 한다. 위치를 스스로 아는 저장소는 0 을 채우면 된다.
    """

    feat_dim = None      # 구현이 __init__ 에서 채운다

    def load_manifest(self):
        raise NotImplementedError(
            "{} 는 load_manifest() 를 구현해야 합니다".format(type(self).__name__))

    def read_record(self, utt_id, locator=None):
        raise NotImplementedError(
            "{} 는 read_record(utt_id, locator) 를 구현해야 합니다".format(
                type(self).__name__))

    def close(self):
        """열어둔 자원 정리 — 없으면 그대로 둬도 된다."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __repr__(self):
        return "{}(feat_dim={})".format(type(self).__name__, self.feat_dim)
