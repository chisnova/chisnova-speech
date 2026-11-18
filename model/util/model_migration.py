"""구 checkpoint 형식을 현재 모델 구조로 변환하는 유틸.

attention 리팩터링(per-head Qdense/Kdense/Vdense → fused q_proj/k_proj/v_proj)
이전에 저장된 checkpoint를 로드할 때 사용한다.

- model.load_state_dict()를 직접 호출하는 경로는 각 attention 모듈에 등록된
  load pre-hook이 자동으로 변환하므로 별도 처리가 필요 없다.
- trainer처럼 checkpoint dict를 수동으로 매칭(partial load)하는 경로는
  매칭 전에 이 함수를 호출해 key를 fused 형식으로 바꿔야 한다.
"""


def fuse_legacy_attention_state_dict(model, state_dict):
    """구 per-head attention key를 fused 형식으로 in-place 변환해 반환한다."""
    for name, module in model.named_modules():
        fuse = getattr(module, "_fuse_legacy_weights", None)
        if fuse is not None:
            prefix = name + "." if name else ""
            fuse(state_dict, prefix)
    return state_dict
