// Linear 레이어: y = x * W + b (완전 연결 계층, Fully Connected Layer)
// 가중치를 자동으로 샤딩하여 병렬 행렬 곱 수행 (column-parallel / row-parallel)
#ifndef MERCURY_ASR_LINEAR
#define MERCURY_ASR_LINEAR

#include "tensor/tensor.h"
#include "json_loader/json_object.h"

#include <future>
#include <string>
#include <vector>

namespace MERCURY {

template<typename T>
class Linear {
public:
    Linear(JsonObject linear_path, std::string base_path);
    ~Linear();

    // 비동기 행렬 곱 래퍼 (ThreadPool 또는 std::async에서 호출)
    static std::shared_ptr<Tensor<T>> async_matmul(std::shared_ptr<Tensor<T>> input, std::shared_ptr<Tensor<T>> weight);
    // 순전파: 샤드 수에 따라 단일/병렬 경로 선택
    std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> input);
    // 열 병렬: W를 열 방향으로 분할, 결과를 수평 연결 (concat)
    std::shared_ptr<Tensor<T>> columParallelForward(std::shared_ptr<Tensor<T>> input);
    // 행 병렬: W를 행 방향으로 분할, 결과를 합산 (reduce-sum)
    std::shared_ptr<Tensor<T>> rowParallelForward(std::shared_ptr<Tensor<T>> input);
    // 차원 크기에 따라 병렬 분할 수 결정 (1, 2, 4 중 선택)
    int getNumParallel(int dim);

    std::vector<std::shared_ptr<Tensor<T>>> weights;  // 샤딩된 가중치 텐서 목록
    std::shared_ptr<Tensor<T>> bias;                   // 편향 벡터 [1, out_features]
    int num_parallel;                                   // 병렬 샤드 수
    int shard_size;                                     // 각 샤드의 크기
    bool isColumParallel;                               // true: 열 병렬, false: 행 병렬
};

} // namespace MERCURY

#endif