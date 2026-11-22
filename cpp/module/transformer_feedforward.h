#ifndef MERCURY_TRANSFORMER_FEEDFORWARD
#define MERCURY_TRANSFORMER_FEEDFORWARD

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "layer/linear.h"
#include "layer/gelu.h"
#include "json_loader/json_object.h"

// TransformerFeedForward: Transformer Feed-Forward Network
// - Linear1 -> GELU -> Linear2
// - Transformer 블록의 position-wise FFN

namespace MERCURY {
    template<typename T>
    class TransformerFeedForward {
    public:
        TransformerFeedForward(JsonObject ffn_path, const std::string& base_path);
        ~TransformerFeedForward() = default;

        std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);

        std::shared_ptr<Linear<T>> linear1;
        std::shared_ptr<Linear<T>> linear2;
    };
} // namespace MERCURY

#endif
