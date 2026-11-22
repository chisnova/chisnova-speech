#ifndef MERCURY_CONFORMER_FEEDFORWARD
#define MERCURY_CONFORMER_FEEDFORWARD

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/silu.h"
#include "layer/layernorm.h"
#include "layer/linear.h"

// ConformerFeedForward: Conformer Feed-Forward Module
// - LayerNorm -> Linear1 -> SiLU -> Linear2
// - Conformer 블록에서 half-step residual의 일부로 사용됨

namespace MERCURY {
    template<typename T>
    class ConformerFeedForward {
    public:
        ConformerFeedForward(JsonObject feedforward_path, const std::string& base_path);
        ~ConformerFeedForward() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        std::shared_ptr<LayerNorm<T>> layernorm;
        std::shared_ptr<Linear<T>> linear1;
        std::shared_ptr<Linear<T>> linear2;
    };
} // namespace MERCURY

#endif
