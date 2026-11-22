// Isolate vendored Eigen FP32 GEMM from Tensor/Linear/thread-pool overhead.
// Experimental kernels are contiguous RowMajor, nonaliasing, assignment-only.
// They use the current input matrices every call; no weight prepacking/caching.
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Clock = std::chrono::steady_clock;
using Packet = Eigen::internal::packet_traits<float>::type;
constexpr int packet_size = Eigen::internal::packet_traits<float>::size;

// Keep Eigen's packing and microkernel intact, changing only the block policy.
// In a RowMajor product, the low-level kernel operates on B^T * A^T, so mc
// partitions the original output columns and nc the original output rows.
struct ManualBlocking : Eigen::internal::level3_blocking<float, float> {
    ManualBlocking(int m, int k, int n, int kc, int mc) {
        m_kc = std::min(k, kc);
        m_mc = std::min(n, mc);
        m_nc = m;
    }
};

template<typename Blocking>
void raw_product(const Matrix& a, const Matrix& b, Matrix& c, Blocking& blocking) {
    c.setZero();
    Eigen::internal::general_matrix_matrix_product<
        Eigen::Index, float, Eigen::RowMajor, false, float, Eigen::RowMajor,
        false, Eigen::RowMajor, 1>::run(
            a.rows(), b.cols(), a.cols(), a.data(), a.outerStride(),
            b.data(), b.outerStride(), c.data(), 1, c.outerStride(), 1.f, blocking);
}

// Keep R*P output packets in registers through the reduction over K. Reuse each
// loaded B packet across R rows, then store the completed output exactly once.
template<int R, int P>
inline void direct_tile(const float* a, const float* b, float* c, int k, int n) {
    using namespace Eigen::internal;
    int j = 0;
    for (; j + P * packet_size <= n; j += P * packet_size) {
        Packet accum[R][P];
        for (int r = 0; r < R; ++r)
            for (int p = 0; p < P; ++p) accum[r][p] = pset1<Packet>(0.f);
        for (int d = 0; d < k; ++d) {
            Packet weights[P];
            for (int p = 0; p < P; ++p)
                weights[p] = ploadu<Packet>(b + d * n + j + p * packet_size);
            for (int r = 0; r < R; ++r) {
                const Packet input = pset1<Packet>(a[r * k + d]);
                for (int p = 0; p < P; ++p)
                    accum[r][p] = pmadd(input, weights[p], accum[r][p]);
            }
        }
        for (int r = 0; r < R; ++r)
            for (int p = 0; p < P; ++p)
                pstoreu(c + r * n + j + p * packet_size, accum[r][p]);
    }
    // The audit includes odd sizes, so do not silently omit the vector tail.
    for (; j < n; ++j)
        for (int r = 0; r < R; ++r) {
            float sum = 0.f;
            for (int d = 0; d < k; ++d) sum += a[r * k + d] * b[d * n + j];
            c[r * n + j] = sum;
        }
}

template<int R, int P>
void direct_product(const Matrix& a, const Matrix& b, Matrix& c) {
    const int m = a.rows(), k = a.cols(), n = b.cols();
    int i = 0;
    for (; i + R <= m; i += R)
        direct_tile<R, P>(a.data() + i * k, b.data(), c.data() + i * n, k, n);
    for (; i < m; ++i)
        direct_tile<1, P>(a.data() + i * k, b.data(), c.data() + i * n, k, n);
}

struct Method {
    std::string name;
    std::function<void()> run;
    std::vector<double> times;
    double scaled_error = 0;
};

double validate(const Matrix& output, const Eigen::MatrixXd& reference) {
    const double error = ((output.cast<double>() - reference).array().abs()
                         / (1. + reference.array().abs())).maxCoeff();
    if (!output.allFinite() || !std::isfinite(error) || error > 5e-5)
        throw std::runtime_error("numerical check failed: " + std::to_string(error));
    return error;
}

int main(int argc, char** argv) try {
    const int rounds = argc > 1 ? std::stoi(argv[1]) : 9;
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 100;
    const int buffers = argc > 3 ? std::stoi(argv[3]) : 1;
    const int seed = argc > 4 ? std::stoi(argv[4]) : 42;
    if (rounds < 3 || iterations < 1 || buffers < 1)
        throw std::runtime_error("usage: benchmark [rounds>=3] [iterations>=1] [buffers>=1] [seed]");
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::cerr << "Eigen=" << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
              << EIGEN_MINOR_VERSION << " SIMD=" << Eigen::SimdInstructionSetsInUse()
              << " packet=" << packet_size << " threads=" << Eigen::nbThreads()
              << " rounds=" << rounds << " iterations=" << iterations
              << " buffers=" << buffers << " seed=" << seed
              << '\n';
    std::cout << "M,K,N,buffers,method,median_us,p95_batch_us,max_scaled_error\n";
    double checksum = 0;
    for (const auto shape : {std::pair<int, int>{256, 256}, {256, 768}, {256, 1024},
                            {1024, 256}, {1216, 256}, {255, 257}, {256, 3},
                            {8, 256}, {32, 256}}) {
        const int k = shape.first, n = shape.second;
        std::vector<Matrix> weights(buffers);
        for (auto& b : weights) {
            b.resize(k, n);
            for (int i = 0; i < b.size(); ++i) b.data()[i] = dist(rng);
        }
        for (const int m : {1, 2, 3, 4, 8, 12, 25, 38, 128}) {
            Matrix a(m, k), c(m, n);
            for (int i = 0; i < a.size(); ++i) a.data()[i] = dist(rng);
            int current_buffer = 0;
            std::vector<Method> methods;
            methods.push_back({"eigen_gemm", [&] { c.noalias() = a * weights[current_buffer]; }});
#ifdef BENCH_BLOCKING
            methods.push_back({"raw_default", [&] {
                Eigen::internal::gemm_blocking_space<Eigen::RowMajor, float, float,
                    Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> blocking(m, n, k, 1, true);
                raw_product(a, weights[current_buffer], c, blocking);
            }});
            for (const auto tile : {std::pair<int, int>{128, 96}, {256, 96},
                                    {512, 96}, {256, 384}}) {
                methods.push_back({"block_k" + std::to_string(tile.first)
                    + "_m" + std::to_string(tile.second), [&, tile] {
                    ManualBlocking blocking(m, k, n, tile.first, tile.second);
                    raw_product(a, weights[current_buffer], c, blocking);
                }});
            }
#else
            methods.push_back({"rows_gemv", [&] {
                for (int i = 0; i < m; ++i)
                    c.row(i).noalias() = a.row(i) * weights[current_buffer];
            }});
            methods.push_back({"columns_gemv", [&] {
                for (int j = 0; j < n; ++j)
                    c.col(j).noalias() = a * weights[current_buffer].col(j);
            }});
            methods.push_back({"lazy_product", [&] { c.noalias() = a.lazyProduct(weights[current_buffer]); }});
            methods.push_back({"direct_2x4p", [&] { direct_product<2, 4>(a, weights[current_buffer], c); }});
            methods.push_back({"direct_4x2p", [&] { direct_product<4, 2>(a, weights[current_buffer], c); }});
#endif
            // Validate every independent weight buffer, poisoning the output
            // first to catch accidental accumulation or incomplete tail stores.
            for (int buf = 0; buf < buffers; ++buf) {
                current_buffer = buf;
                const Eigen::MatrixXd reference = a.cast<double>() * weights[buf].cast<double>();
                for (auto& method : methods) {
                    c.setConstant(std::numeric_limits<float>::quiet_NaN());
                    method.run();
                    method.scaled_error = std::max(method.scaled_error, validate(c, reference));
                }
            }
            for (auto& method : methods)
                for (int i = 0; i < 20; ++i) {
                    current_buffer = i % buffers;
                    method.run();
                }
            for (int round = 0; round < rounds; ++round) {
                for (size_t j = 0; j < methods.size(); ++j) {
                    auto& method = methods[(j + round) % methods.size()];
                    const auto start = Clock::now();
                    for (int i = 0; i < iterations; ++i) {
                        current_buffer = i % buffers;
                        method.run();
                        // Force all stores to be observable, even with LTO.
                        asm volatile("" : : "g"(c.data()) : "memory");
                        checksum += c.data()[(i * 17) % c.size()];
                    }
                    method.times.push_back(std::chrono::duration<double, std::micro>(
                        Clock::now() - start).count() / iterations);
                }
            }
            for (auto& method : methods) {
                std::sort(method.times.begin(), method.times.end());
                std::cout << m << ',' << k << ',' << n << ',' << buffers << ',' << method.name << ','
                          << std::fixed << std::setprecision(4)
                          << method.times[method.times.size() / 2] << ','
                          << method.times[static_cast<size_t>(std::ceil(.95 * method.times.size())) - 1]
                          << ',' << std::scientific << method.scaled_error << '\n';
            }
        }
    }
    if (!std::isfinite(checksum)) throw std::runtime_error("non-finite checksum");
    std::cerr << "checksum=" << checksum << '\n';
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
