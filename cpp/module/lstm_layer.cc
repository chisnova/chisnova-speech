#include "module/lstm_layer.h"

namespace MERCURY {
    template<typename T>
    LSTMLayer<T>::LSTMLayer(JsonObject lstm_path, const std::string& base_path){
        input_dense = std::make_shared<Linear<T>>(lstm_path.parseObject("input_dense"), base_path);
        hidden_dense = std::make_shared<Linear<T>>(lstm_path.parseObject("hidden_dense"), base_path);
        input_dim = int(lstm_path.parserValue("input_dim"));
        output_dim = int(lstm_path.parserValue("output_dim"));
        hidden = std::make_shared<Tensor<T>>();
        cell = std::make_shared<Tensor<T>>();
        hidden->value.resize(1, output_dim);
        hidden->value.setZero();
        cell->value.resize(1, output_dim);
        cell->value.setZero();
    }

    template<typename T>
    void LSTMLayer<T>::setState(const std::shared_ptr<Tensor<float>>& _hidden, const std::shared_ptr<Tensor<float>>& _cell) {
        hidden->value = _hidden->value;
        cell->value = _cell->value;
    }

    // forward: LSTM gate 연산
    // 최적화: 4개 gate 텐서 할당 제거 → total_sum의 block view를 직접 사용
    // 기존: make_shared<Tensor> x4 + 값 복사 → Eigen array view로 직접 계산
    template<typename T>
    std::shared_ptr<Tensor<T>> LSTMLayer<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        auto output = std::make_shared<Tensor<T>>();

        // input/hidden linear projection 후 합산
        std::shared_ptr<Tensor<T>> input_cal = input_dense->forward(input);
        std::shared_ptr<Tensor<T>> hidden_cal = hidden_dense->forward(hidden);

        // in-place 합산: 별도 total_sum 할당 대신 input_cal 재사용
        input_cal->value.noalias() += hidden_cal->value;

        // gate를 Eigen block view로 직접 참조 → 4개 make_shared 제거
        // middleCols는 view이므로 복사 없음
        auto i_gate = (1.0f + (-input_cal->value.middleCols(output_dim * 0, output_dim).array()).exp()).cwiseInverse();
        auto f_gate = (1.0f + (-input_cal->value.middleCols(output_dim * 1, output_dim).array()).exp()).cwiseInverse();
        auto g_gate = input_cal->value.middleCols(output_dim * 2, output_dim).array().tanh();
        auto o_gate = (1.0f + (-input_cal->value.middleCols(output_dim * 3, output_dim).array()).exp()).cwiseInverse();

        // cell state 업데이트: c = f * c_prev + i * g
        cell->value.array() = cell->value.array() * f_gate + i_gate * g_gate;
        // hidden state 업데이트: h = o * tanh(c)
        hidden->value.array() = o_gate * cell->value.array().tanh();

        output->value = hidden->value;
        return output;
    }

    template class LSTMLayer<float>;
    template class LSTMLayer<int>;
};
