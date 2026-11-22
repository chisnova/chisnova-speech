#ifndef MERCURY_CONFORMER_STREAM_CONVMODULE
#define MERCURY_CONFORMER_STREAM_CONVMODULE

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/convolution1d.h"
#include "layer/glu.h"
#include "layer/silu.h"
#include "layer/layernorm.h"

// ConformerStreamConvModule: Streaming Conformer Convolution Module
// - 비스트리밍 ConformerConvModule의 streaming 버전
// - Depthwise Conv에 causal padding을 위해 이전 프레임 캐시(pre) 유지
// - BatchNorm 대신 LayerNorm 사용 (streaming 환경에서 BatchNorm 불가)

namespace MERCURY {
    template<typename T>
    class ConformerStreamConvModule {
    public:
        ConformerStreamConvModule(JsonObject conv_module_path, const std::string& base_path);
        ~ConformerStreamConvModule() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);
        std::shared_ptr<Tensor<T>> makePad(const std::shared_ptr<Tensor<T>>& input);
        void clear();

        std::shared_ptr<LayerNorm<T>> layernorm;
        std::shared_ptr<Convolution1D<T>> pointwise1;
        std::shared_ptr<Convolution1D<T>> depthwise;
        std::shared_ptr<Convolution1D<T>> pointwise2;
        std::shared_ptr<LayerNorm<T>> layernorm2;
        std::shared_ptr<Tensor<T>> pre;  // 이전 프레임 캐시 (causal padding용)
        int ksize;
    };
} // namespace MERCURY

#endif
