#include "layer/linear.h"
#include "utils/thread_pool.h"

namespace MERCURY {

template<typename T>
Linear<T>::Linear(JsonObject linear_path, std::string base_path) {
    auto full_weight = std::make_shared<Tensor<T>>();

    // JSON에서 가중치/편향 파일 경로 및 shape [in_features, out_features] 파싱
    std::string weight_path = base_path + "/" + linear_path.parserString("weight");
    std::string bias_path = base_path + "/" + linear_path.parserString("bias");
    std::vector<int> shape = linear_path.parserIntArray("shape");

    // 가중치 행렬 W [in_features x out_features] 로드
    full_weight->loadValue(weight_path, shape);
    // 편향 벡터 b [1 x out_features] 로드
    bias = std::make_shared<Tensor<T>>();
    bias->loadValue(bias_path, {1, shape[1]});

    // 큰 차원 기준으로 병렬 전략 결정: in < out이면 열 병렬, 아니면 행 병렬
    isColumParallel = (shape[0] <= shape[1]);
    int max_len = shape[0] > shape[1] ? shape[0] : shape[1];

    // 샤드 크기가 512 미만이 되도록 병렬 수(1/2/4) 결정
    num_parallel = getNumParallel(max_len);

    if (isColumParallel) {
        // 열 병렬: W를 열 방향으로 num_parallel 등분
        shard_size = shape[1] / num_parallel;
        for (int i = 0; i < num_parallel; ++i) {
            auto shard_weight = std::make_shared<Tensor<T>>();
            // W[:, i*shard_size : (i+1)*shard_size] 블록 추출
            shard_weight->value = full_weight->value.block(0, i * shard_size, shape[0], shard_size);
            weights.push_back(std::move(shard_weight));
        }
    } else {
        // 행 병렬: W를 행 방향으로 num_parallel 등분
        shard_size = shape[0] / num_parallel;
        for (int i = 0; i < num_parallel; ++i) {
            auto shard_weight = std::make_shared<Tensor<T>>();
            // W[i*shard_size : (i+1)*shard_size, :] 블록 추출
            shard_weight->value = full_weight->value.block(i * shard_size, 0, shard_size, shape[1]);
            weights.push_back(std::move(shard_weight));
        }
    }
}

template<typename T>
Linear<T>::~Linear() = default;

// 병렬 분할 수 결정: 샤드 크기가 512 미만이 되는 최소 분할 수 (최대 4)
template<typename T>
int Linear<T>::getNumParallel(int dim) {
    int num_p = 1;
    for (; num_p < 4; num_p *= 2) {
        int s = dim / num_p;
        // 균등 분할 가능하고 샤드가 충분히 작으면 중단
        if (dim % num_p == 0 && s < 512) {
            break;
        }
    }
    return num_p;
}

// 비동기 실행용 행렬 곱 래퍼: input * weight
template<typename T>
std::shared_ptr<Tensor<T>> Linear<T>::async_matmul(std::shared_ptr<Tensor<T>> input, std::shared_ptr<Tensor<T>> weight) {
    return input->matmul(weight);
}

// 열 병렬 순전파: 각 샤드별로 x * W_i 를 병렬 계산 후 수평 연결
// 결과: [batch, shard_size*num_parallel] = [batch, out_features]
template<typename T>
std::shared_ptr<Tensor<T>> Linear<T>::columParallelForward(std::shared_ptr<Tensor<T>> input) {
    auto out = std::make_shared<Tensor<T>>();
    out->value.resize(input->shape(0), shard_size * num_parallel);

    // 각 샤드에 대해 비동기 행렬 곱 실행
    std::vector<std::future<std::shared_ptr<Tensor<T>>>> future_vector;
    future_vector.reserve(num_parallel);

    for (int i = 0; i < num_parallel; ++i) {
        #ifdef THREAD_POOL_ENABLED
        future_vector.push_back(MERCURY::ThreadPool::submit(async_matmul, input, weights[i]));
        #else
        future_vector.push_back(std::async(std::launch::async, async_matmul, input, weights[i]));
        #endif
    }

    // 각 샤드 결과를 출력 텐서의 해당 열 블록에 배치 (수평 연결)
    const int in_rows = input->shape(0);
    for (int i = 0; i < num_parallel; ++i) {
        auto output = future_vector[i].get();
        out->value.block(0, i * shard_size, in_rows, shard_size).noalias() = output->value;
    }
    // In-place bias addition (broadcasting)
    out->value.array().rowwise() += bias->value.row(0).array();
    return out;
}

// 행 병렬 순전파: 입력을 열 방향으로 분할, 각 x_i * W_i 를 병렬 계산 후 합산
// x = [x_0 | x_1 | ... ], y = sum(x_i * W_i) + b
template<typename T>
std::shared_ptr<Tensor<T>> Linear<T>::rowParallelForward(std::shared_ptr<Tensor<T>> input) {
    std::vector<std::future<std::shared_ptr<Tensor<T>>>> future_vector;
    future_vector.reserve(num_parallel);

    const int in_rows = input->shape(0);
    for (int i = 0; i < num_parallel; ++i) {
        // Capture a materialized copy of the block (Eigen block expressions are not thread-safe
        // to evaluate concurrently if the source could be accessed from another thread).
        // Using MatrixXf directly avoids the shared_ptr<Tensor> allocation overhead.
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> block_copy =
            input->value.block(0, i * shard_size, in_rows, shard_size);
        #ifdef THREAD_POOL_ENABLED
        future_vector.push_back(MERCURY::ThreadPool::submit([block_copy = std::move(block_copy), w = weights[i]]() {
            auto out = std::make_shared<Tensor<T>>();
            out->value.noalias() = block_copy * w->value;
            return out;
        }));
        #else
        future_vector.push_back(std::async(std::launch::async, [block_copy = std::move(block_copy), w = weights[i]]() {
            auto out = std::make_shared<Tensor<T>>();
            out->value.noalias() = block_copy * w->value;
            return out;
        }));
        #endif
    }

    // 첫 번째 결과를 기준으로 나머지 샤드 결과를 누적 합산 (reduce-sum)
    auto out = future_vector[0].get();
    for (int i = 1; i < num_parallel; ++i) {
        auto output = future_vector[i].get();
        out->value.noalias() += output->value;
    }
    // In-place bias addition
    out->value.array().rowwise() += bias->value.row(0).array();
    return out;
}

// 순전파 진입점: 샤드가 1개면 직접 계산, 아니면 병렬 전략에 위임
template<typename T>
std::shared_ptr<Tensor<T>> Linear<T>::forward(std::shared_ptr<Tensor<T>> input) {
    if (num_parallel == 1) {
        // 단일 샤드: noalias()로 Eigen 임시 행렬 할당 방지 → 대형 행렬 곱 성능 향상
        auto out = std::make_shared<Tensor<T>>();
        out->value.noalias() = input->value * weights[0]->value;
        // In-place bias addition avoids temporary allocation
        out->value.array().rowwise() += bias->value.row(0).array();
        return out;
    }
    if (isColumParallel) {
        return columParallelForward(input);
    }
    return rowParallelForward(input);
}

template class Linear<float>;
template class Linear<int>;

} // namespace MERCURY
