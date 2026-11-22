#include "module/conformer_stream_conv_module.h"

namespace MERCURY {
    template<typename T>
    ConformerStreamConvModule<T>::ConformerStreamConvModule(JsonObject conv_module_path, const std::string& base_path) {
        layernorm = std::make_shared<LayerNorm<T>>(conv_module_path.parseObject("layer_norm"), base_path);
        pointwise1= std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("pointwise1"), base_path);
        depthwise= std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("depthwise"), base_path);
        layernorm2 = std::make_shared<LayerNorm<T>>(conv_module_path.parseObject("layer_norm2"), base_path);
        pointwise2 = std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("pointwise2"), base_path);
        pre = std::make_shared<Tensor<T>>();
        pre->value.resize(0,0);
        ksize = int(conv_module_path.parserValue("ksize"));
    }

    template<typename T>
    void ConformerStreamConvModule<T>::clear() {
        pre = nullptr;
        pre = std::make_shared<Tensor<T>>();
        pre->value.resize(0,0);
    }

    // makePad: causal padding 생성
    // 최적화: noalias로 block copy 최적화
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamConvModule<T>::makePad(const std::shared_ptr<Tensor<T>>& input) {
        auto out = std::make_shared<Tensor<T>>();
        const int rows = input->shape(0);
        const int cols = input->shape(1);
        out->value.resize(rows, ksize - 1 + cols);
        out->value.setZero();
        out->value.rightCols(cols).noalias() = input->value;

        if(pre->value.size()==0){
            return out;
        }
        // 이전 프레임의 마지막 ksize-1 열을 왼쪽 padding으로 사용
        out->value.leftCols(ksize-1).noalias() = pre->value.rightCols(ksize-1);
        return out;
    }

    // forward: streaming conv module
    // 최적화: transpose를 가능한 한 줄이고 noalias 활용
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerStreamConvModule<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        std::shared_ptr<Tensor<float>> output;
        output = layernorm->forward(input);
        output = output->transpose();
        output = pointwise1->forward(output);
        output = GLU().forward2(output);

        // causal padding + 캐시 업데이트
        // 캐시에는 "직전 청크" 가 아니라 "패딩까지 포함한 스트림의 마지막 ksize-1 프레임"
        // 을 둔다 — 직전 청크만 들고 있으면 청크가 ksize-1 보다 짧을 때 좌컨텍스트가
        // 모자란다 (Python ConformerStreamConvModule.stream 과 동일한 정책)
        std::shared_ptr<Tensor<T>> padded = makePad(output);
        pre->value = padded->value.rightCols(ksize - 1);

        output = depthwise->forward(padded);
        output = output->transpose();
        output = layernorm2->forward(output);
        output = output->transpose();
        output = SILU().forward(output);
        output = pointwise2->forward(output);
        output = output->transpose();
        return output;
    }
    template class ConformerStreamConvModule<float>;
    // template class ConformerConvModule<int>;
};
