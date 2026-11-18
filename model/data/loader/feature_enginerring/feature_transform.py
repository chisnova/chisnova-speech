"""FeatureTransform — 특징 변환의 기반 클래스."""
import torch


class FeatureTransform:
    """feature (B,T,D) 를 받아 변환한 feature 를 돌려준다. 그게 전부다.

        feat = transform(feat)

    서브클래스는 __call__ 하나만 구현하면 된다.

    지켜야 할 것
      - 모양 (B,T,D) 은 바꾸지 않는다. 배치와 feat_mask 는 이미 확정돼 있다.
      - 패딩 프레임은 0 으로 남긴다. Collator 가 패딩을 0 으로 채우므로 어디가
        패딩인지는 feat 만 보면 알 수 있다 — get_padding_row_mask(feat) 참고.
      - in-place 로 바꿔도 되고 새 텐서를 만들어도 되지만, 호출자는 반환값을 쓴다.

    난수는 torch 전역 RNG 를 쓴다 — 재현이 필요하면 torch.manual_seed() 를 부르면 된다.
    """

    _PARAMS = ()          # __repr__ 에 찍을 속성 이름

    def __call__(self, feat):
        raise NotImplementedError(
            "{} 는 __call__(feat) 을 구현해야 합니다".format(type(self).__name__))

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            ", ".join("{}={}".format(k, getattr(self, k)) for k in self._PARAMS))


# -- 서브클래스 공용 헬퍼 ----------------------------------------------------
def get_padding_row_mask(feat):
    """패딩 프레임 (B,T) bool. 패딩은 전 채널이 0 이므로 feat 만으로 알 수 있다."""
    return feat.abs().sum(-1) == 0


def sample_random_span(extent, max_width, max_ratio=1.0):
    """extent(B,) 안에서 구간 [start, start+width) 를 발화별로 하나씩 뽑는다.

    max_ratio 는 폭의 extent 대비 상한 — 짧은 발화가 통째로 지워지는 걸 막는다.
    반환: (start, width) 각각 (B,) long
    """
    max_width_per_utt = (extent.float() * max_ratio).long().clamp(
        min=0, max=int(max_width))
    width = (torch.rand_like(max_width_per_utt, dtype=torch.float)
             * (max_width_per_utt + 1)).long().clamp(max=max_width_per_utt)
    start_room = (extent - width).clamp(min=0)
    start = (torch.rand_like(start_room, dtype=torch.float)
             * (start_room + 1)).long().clamp(max=start_room)
    return start, width


def make_span_mask(start, width, size):
    """(start, width) -> (B, size) bool 마스크 (구간 안이 True)."""
    positions = torch.arange(size, device=start.device).unsqueeze(0)
    span_start = start.unsqueeze(1)
    return (positions >= span_start) & (positions < span_start + width.unsqueeze(1))
