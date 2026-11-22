#ifndef MERCURY_CONFORMER_CONVMODULE
#define MERCURY_CONFORMER_CONVMODULE

#include <string>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/convolution1d.h"
#include "layer/glu.h"
#include "layer/silu.h"
#include "layer/batchnorm1d.h"
#include "layer/layernorm.h"

// ConformerConvModule: Conformer Convolution Module
// - LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> SiLU -> Pointwise Conv
// - 입력을 transpose하여 channel-first로 변환 후 convolution 적용

namespace MERCURY {
    template<typename T>
    class ConformerConvModule {
    public:
        ConformerConvModule(JsonObject conv_module_path, const std::string& base_path);
        ~ConformerConvModule() = default;

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input);

        std::shared_ptr<LayerNorm<T>> layernorm;
        std::shared_ptr<Convolution1D<T>> pointwise1;
        std::shared_ptr<Convolution1D<T>> depthwise;
        std::shared_ptr<Convolution1D<T>> pointwise2;
        std::shared_ptr<BatchNorm1D<T>> batchnorm;
    };
} // namespace MERCURY

#endif
