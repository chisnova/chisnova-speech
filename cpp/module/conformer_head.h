#ifndef MERCURY_CONFORMER_HEAD
#define MERCURY_CONFORMER_HEAD

#include <string>
#include <memory>
#include <cmath>
#include "tensor/tensor.h"
#include "layer/softmax.h"

// ConformerHead: Conformer 단일 Attention Head (relative positional attention 코어)
// - Q/K/V/Pos projection 은 ConformerMultiHead 의 fused Linear 가 담당하고,
//   이 클래스는 projection 된 per-head 슬라이스만 받아 attention 을 계산한다
// - relative_shift로 position 정보를 attention score에 반영
//   (Python ConformerMultiHead.forward 의 per-head 수식과 동일)

namespace MERCURY {
    template<typename T>
    class ConformerHead {
    public:
        ConformerHead(int _head_dim);
        ~ConformerHead() = default;

        std::shared_ptr<Tensor<T>> relative_shift(const std::shared_ptr<Tensor<T>>& pos_score);
        // query/key/value/pos: (L, head_dim) — projection 완료된 per-head 슬라이스
        std::shared_ptr<Tensor<T>> forwardProjected(std::shared_ptr<Tensor<T>> query,
                                                    std::shared_ptr<Tensor<T>> key,
                                                    std::shared_ptr<Tensor<T>> value,
                                                    std::shared_ptr<Tensor<T>> pos);

        int head_dim;
        T inv_norm;   // 1 / sqrt(head_dim)
    };
} // namespace MERCURY

#endif
