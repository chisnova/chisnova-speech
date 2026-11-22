#include "module/conformer_conv_module.h"

namespace MERCURY {
    template<typename T>
    ConformerConvModule<T>::ConformerConvModule(JsonObject conv_module_path, const std::string& base_path) {
        layernorm = std::make_shared<LayerNorm<T>>(conv_module_path.parseObject("layer_norm"), base_path);
        pointwise1= std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("pointwise1"), base_path);
        depthwise= std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("depthwise"), base_path);
        batchnorm = std::make_shared<BatchNorm1D<T>>(conv_module_path.parseObject("batch_norm"), base_path);
        pointwise2 = std::make_shared<Convolution1D<T>>(conv_module_path.parseObject("pointwise2"), base_path);
    }

    // forward: Conv module pipeline
    // LayerNorm → transpose → Pointwise1 → GLU → Depthwise → BatchNorm → SiLU → Pointwise2 → transpose
    template<typename T>
    std::shared_ptr<Tensor<T>> ConformerConvModule<T>::forward(const std::shared_ptr<Tensor<T>>& input) {
        std::shared_ptr<Tensor<float>> output;
        output = layernorm->forward(input);
        output = output->transpose();
        output = pointwise1->forward(output);
        output = GLU().forward2(output);
        output = depthwise->forward(output);
        output = batchnorm->forward(output);
        output = SILU().forward(output);
        output = pointwise2->forward(output);
        output = output->transpose();
        return output;
    }
    template class ConformerConvModule<float>;
    // template class ConformerConvModule<int>;
};
