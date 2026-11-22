#ifndef MERCURY_TENSOR_H
#define MERCURY_TENSOR_H

// #define EIGEN_DEFAULT_ALIGN_BYTES 32
#define EIGEN_STRONG_INLINE inline
#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_NO_STATIC_ASSERT
#define EIGEN_USE_LAPACKE
#include "logger/logger_manager.h"
#include <utility>
#include <memory>
#include "third_party/eigen/Eigen/Dense"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <stdexcept>

namespace MERCURY {
    template<typename T>
    class Tensor {
        public:
        // [2026-04-07] Move 생성자/대입 연산자 추가 — Eigen::Matrix는 move를 네이티브 지원하므로
        // default로 위임. 기존에는 user-declared 소멸자로 인해 암묵적 move가 생성되지 않았음(Rule of Five)
        Tensor() = default;
        ~Tensor() = default;
        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        void loadValue(std::string path, std::vector<int> _shape);
        void setValue(std::vector<T> _value, std::vector<int> _shape);
        void printValue();
        void printShape();
        int shape(int idx);
        void clone(std::shared_ptr<Tensor<T>> other);
        std::shared_ptr<Tensor<T>> matmul(std::shared_ptr<Tensor<T>> other);
        std::shared_ptr<Tensor<T>> flatten();
        std::shared_ptr<Tensor<T>> reshape(int h, int w);
        std::shared_ptr<Tensor<T>> transpose();

        void setRows(int i, std::shared_ptr<Tensor<T>> other, int j);
        void setCols(int i, std::shared_ptr<Tensor<T>> other, int j);

        void appendRow(std::shared_ptr<Tensor<T>> other);
        void appendChunkRow(std::shared_ptr<Tensor<T>> other);
        void halfDimValue(std::shared_ptr<Tensor<T>> other1, std::shared_ptr<Tensor<T>> other2);

        // [2026-04-07] appendChunkRowWithCapacity 추가 — 기존 conservativeResize의 O(N^2) 문제 해결
        // 기존 appendChunkRow: 매 호출시 전체 데이터 재할당+복사 (conservativeResize)
        // 신규: capacity 기반 기하급수적 확장 (2배) + logical row 추적으로 amortized O(1) append
        // KV 캐시 스트리밍에서 시퀀스 길이 L에 대해 O(L^2) → O(L)으로 개선
        void appendChunkRowWithCapacity(std::shared_ptr<Tensor<T>> other, int& logical_rows);

        // [2026-04-07] trimToLogicalRows 추가 — capacity 기반 append 후 실제 사용 크기로 축소
        void trimToLogicalRows(int logical_rows);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> value;
    };
};

#endif
