#ifndef MERCURY_ASR_BAYESIAN_CMVN_H_
#define MERCURY_ASR_BAYESIAN_CMVN_H_

#include "tensor/tensor.h"
#include <functional>
#include <numeric>
#include <algorithm>

namespace MERCURY {

template<typename T>
class BayesianCMVNChunk {
    public:
        BayesianCMVNChunk(std::shared_ptr<Tensor<T>> mu0, std::shared_ptr<Tensor<T>> std);
        void reset();
        std::shared_ptr<Tensor<T>> updateChunk(const std::shared_ptr<Tensor<T>>& feat);

    private:
        std::shared_ptr<Tensor<T>> org_mean;    // original mean
        std::shared_ptr<Tensor<T>> running_mean;    // running mean
        float kappa;   // mean prior virtual frame count  
        float alpha;   // variance prior degrees of freedom 
        std::shared_ptr<Tensor<T>> org_var;  // original var
        std::shared_ptr<Tensor<T>> beta;  // posterior β
        float epsilon; // small constant for numerical stability
};

} // namespace MERCURY

#endif // MERCURY_ASR_BAYESIAN_CMVN_H_