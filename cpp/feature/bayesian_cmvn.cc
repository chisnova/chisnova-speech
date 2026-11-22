#include "feature/bayesian_cmvn.h"

namespace MERCURY {

    template<typename T>
    BayesianCMVNChunk<T>::BayesianCMVNChunk(std::shared_ptr<Tensor<T>> mu0, std::shared_ptr<Tensor<T>> var) {          // posterior μ
        org_mean = mu0;
        org_var = var;
        reset();
    }

    template<typename T>
    void BayesianCMVNChunk<T>::reset() {
        running_mean = nullptr;
        beta = nullptr;
        kappa = 800.0f;
        alpha = 2.5f;
        epsilon = 1e-8f;

        running_mean = std::make_shared<Tensor<T>>();
        beta = std::make_shared<Tensor<T>>();
        running_mean->value = org_mean->value;
        beta->value = (alpha - 1.0f) * org_var->value;

    }

    template<typename T>
    std::shared_ptr<Tensor<T>> BayesianCMVNChunk<T>::updateChunk(const std::shared_ptr<Tensor<T>>& x) {
        std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>();
        
        
        // Initialize output tensor with proper size
        output->value.resize(x->shape(0), x->shape(1));
        
        int chunk_N = x->value.rows();
        
        // Compute chunk statistics safely
        std::shared_ptr<Tensor<T>> chunk_mean = std::make_shared<Tensor<T>>();
        std::shared_ptr<Tensor<T>> chunk_ss = std::make_shared<Tensor<T>>();
        chunk_mean->value = x->value.colwise().mean();
        chunk_ss->value = ((x->value.rowwise() - chunk_mean->value).array().square()).colwise().sum();
        
        
        // Update parameters step by step to avoid complex expressions
        float kappa_new = kappa + chunk_N;
        std::shared_ptr<Tensor<T>> mu_new = std::make_shared<Tensor<T>>();
        mu_new->value = (kappa * running_mean->value + chunk_N * chunk_mean->value) / kappa_new;

        float alpha_new = alpha + chunk_N / 2.0f;
        
        // Break down complex expression into steps
        // std::shared_ptr<Tensor<T>> beta_base = std::make_shared<Tensor<T>>();
        // beta_base->value = beta->value;

        // std::shared_ptr<Tensor<T>> chunk_contribution = std::make_shared<Tensor<T>>();
        // chunk_contribution->value = 0.5f * chunk_ss->value.array();

        std::shared_ptr<Tensor<T>> mean_diff = std::make_shared<Tensor<T>>();
        mean_diff->value = (chunk_mean->value - running_mean->value).array();
        std::shared_ptr<Tensor<T>> prior_contribution = std::make_shared<Tensor<T>>();
        prior_contribution->value = 0.5f * (kappa * chunk_N / kappa_new) * mean_diff->value.array().square();
        
        std::shared_ptr<Tensor<T>> beta_new = std::make_shared<Tensor<T>>();
        beta_new->value = beta->value.array() + 0.5f * chunk_ss->value.array() + prior_contribution->value.array();
        std::shared_ptr<Tensor<T>> var_post = std::make_shared<Tensor<T>>();
        var_post->value = beta_new->value / (alpha_new - 1.0f);

        // Update state
        running_mean->value = mu_new->value;
        kappa = kappa_new; 
        alpha = alpha_new;
        beta->value = beta_new->value;
        
        // Compute output safely
        std::shared_ptr<Tensor<T>> normalized_input = std::make_shared<Tensor<T>>();
        normalized_input->value = (x->value.rowwise() - running_mean->value);
        std::shared_ptr<Tensor<T>> std_dev = std::make_shared<Tensor<T>>();
        std_dev->value = (var_post->value.array() + epsilon).sqrt();
        
        // Fix dimension mismatch: std_dev is (1, features), need to broadcast to each row
        // Use colwise division instead of rowwise
        output->value = normalized_input->value.array().rowwise() / std_dev->value.array();

        return output;
    } 
    template class BayesianCMVNChunk<float>;
}
