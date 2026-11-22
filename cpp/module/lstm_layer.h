#ifndef MERCURY_ASR_LSTM
#define MERCURY_ASR_LSTM

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "layer/linear.h"
#include "json_loader/json_object.h"

// LSTMLayer: LSTM (Long Short-Term Memory) 레이어
// - input gate, forget gate, cell gate, output gate를 통해 시퀀스 처리
// - hidden/cell state를 유지하며 시계열 데이터의 장기 의존성 학습

namespace MERCURY {
    template<typename T>
    class LSTMLayer {
    public:
        LSTMLayer(JsonObject lstm_path, const std::string& base_path);
        ~LSTMLayer() = default;

        void setState(const std::shared_ptr<Tensor<float>>& _hidden,
                      const std::shared_ptr<Tensor<float>>& _cell);
        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        std::shared_ptr<Linear<T>> input_dense;
        std::shared_ptr<Linear<T>> hidden_dense;
        std::shared_ptr<Tensor<T>> hidden;
        std::shared_ptr<Tensor<T>> cell;
        int input_dim;
        int output_dim;
    };
} // namespace MERCURY

#endif
