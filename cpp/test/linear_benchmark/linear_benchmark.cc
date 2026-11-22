// Compare production Linear::forward with experimental scheduling/layout choices.
// Run from the repository root; weights are loaded outside the timed region.
#include "layer/linear.h"
#include "utils/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>
#include <random>
#include <stdexcept>

using MERCURY::JsonObject;
using MERCURY::Linear;
using MERCURY::Tensor;
using MERCURY::ThreadPool;
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Output = std::shared_ptr<Tensor<float>>;
using Method = std::function<Output()>;
using Clock = std::chrono::steady_clock;

struct Sample {
    std::string path;
    JsonObject config;
    int count = 0;
};

// Select actual Linear nodes, excluding layer norm, embeddings and convolutions.
void collect(const nlohmann::json& node, const std::string& path,
             std::map<std::pair<int, int>, Sample>& samples) {
    if (!node.is_object() && !node.is_array()) return;
    if (node.is_object() && node.contains("weight") && node.contains("bias") &&
        node.contains("shape") && node["bias"].is_string()) {
        const auto shape = node["shape"].get<std::vector<int>>();
        const auto name = path.substr(path.find_last_of('.') + 1);
        if (shape.size() == 2 && (name.find("linear") != std::string::npos || name == "qkv_proj")) {
            auto& sample = samples[{shape[0], shape[1]}];
            if (sample.count++ == 0) {
                sample.path = path;
                sample.config.obj = node;
            }
        }
    }
    for (auto it = node.begin(); it != node.end(); ++it) {
        const auto key = node.is_object() ? it.key() : std::to_string(it - node.begin());
        collect(it.value(), path + "." + key, samples);
    }
}

// The caller computes the last partition; other workers write disjoint output
// regions. Captures own the output until every worker has finished.
template<class F>
void parallel(int parts, F work) {
    std::vector<std::future<void>> pending;
    pending.reserve(parts - 1);
    std::exception_ptr error;
    try {
        for (int i = 0; i + 1 < parts; ++i) pending.push_back(ThreadPool::submit(work, i));
        work(parts - 1);
    } catch (...) {
        error = std::current_exception();
    }
    for (auto& task : pending) {
        try { task.get(); }
        catch (...) { if (!error) error = std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
}

double validate(const Method& method, const Eigen::MatrixXd& reference) {
    auto out = method();
    if (out->value.rows() != reference.rows() || out->value.cols() != reference.cols())
        throw std::runtime_error("output shape mismatch");
    if (!out->value.allFinite()) throw std::runtime_error("non-finite output");
    const Eigen::MatrixXd error = out->value.cast<double>() - reference;
    const double scaled = (error.array().abs() / (1.0 + reference.array().abs())).maxCoeff();
    if (!std::isfinite(scaled) || scaled > 5e-5)
        throw std::runtime_error("numerical validation failed: " + std::to_string(scaled));
    return scaled;
}

int main(int argc, char** argv) try {
    const std::string model_dir = argc > 1 ? argv[1] :
        "android_mercury/app/src/main/assets/model_state";
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 200;
    if (iterations < 1) throw std::runtime_error("iterations must be positive");
    JsonObject manifest(model_dir + "/model.json");
    std::map<std::pair<int, int>, Sample> samples;
    collect(manifest.obj, "", samples);
    if (samples.empty()) throw std::runtime_error("no Linear nodes found in manifest");
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::cerr << "Eigen " << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.'
              << EIGEN_MINOR_VERSION << ", SIMD=" << Eigen::SimdInstructionSetsInUse()
              << ", Eigen threads=" << Eigen::nbThreads() << ", pool workers="
              << ThreadPool::getInstance().get_thread_count() << ", iterations=" << iterations << '\n';
    std::cout << "M,K,N,linear_count,method,median_us,p95_us,batch_median_us,max_scaled_error\n";
    for (auto& entry : samples) {
        auto& sample = entry.second;
        const int k = entry.first.first, n = entry.first.second;
        std::cerr << k << " -> " << n << " count=" << sample.count << " source=" << sample.path << '\n';
        Linear<float> baseline(sample.config, model_dir);
        Tensor<float> weight;
        weight.loadValue(model_dir + "/" + sample.config.parserString("weight"), {k, n});
        const Eigen::MatrixXf colmajor_weight = weight.value;
        const Matrix& bias = baseline.bias->value;
        const int col_parts = std::min(4, n);
        std::vector<Matrix> col_weights;
        for (int i = 0; i < col_parts; ++i) {
            const int begin = n * i / col_parts, end = n * (i + 1) / col_parts;
            col_weights.emplace_back(weight.value.middleCols(begin, end - begin));
        }
        for (const int m : {1, 4, 12, 25, 38, 128}) {
            auto input = std::make_shared<Tensor<float>>();
            input->value.resize(m, k);
            for (int i = 0; i < input->value.size(); ++i) input->value.data()[i] = dist(rng);
            Eigen::MatrixXd reference = input->value.cast<double>() * weight.value.cast<double>();
            reference.rowwise() += bias.row(0).cast<double>();
            std::vector<std::pair<std::string, Method>> methods;
            methods.emplace_back("current", [&] { return baseline.forward(input); });
            methods.emplace_back("existing_shards_direct", [&] {
                const int parts = baseline.num_parallel;
                if (baseline.isColumParallel) {
                    auto out = std::make_shared<Tensor<float>>();
                    out->value.resize(m, n);
                    parallel(parts, [&, out, input](int i) {
                        auto dst = out->value.middleCols(i * baseline.shard_size, baseline.shard_size);
                        dst.noalias() = input->value * baseline.weights[i]->value;
                    });
                    out->value.rowwise() += bias.row(0);
                    return out;
                }
                std::vector<Output> partial(parts);
                parallel(parts, [&, input](int i) {
                    auto out = std::make_shared<Tensor<float>>();
                    out->value.noalias() = input->value.middleCols(i * baseline.shard_size, baseline.shard_size)
                        * baseline.weights[i]->value;
                    partial[i] = std::move(out);
                });
                auto out = partial[0];
                for (int i = 1; i < parts; ++i) out->value += partial[i]->value;
                out->value.rowwise() += bias.row(0);
                return out;
            });
            methods.emplace_back("serial_rowmajor", [&] {
                auto out = std::make_shared<Tensor<float>>();
                out->value.noalias() = input->value * weight.value;
                out->value.rowwise() += bias.row(0);
                return out;
            });
            methods.emplace_back("serial_colmajor_weight", [&] {
                auto out = std::make_shared<Tensor<float>>();
                out->value.noalias() = input->value * colmajor_weight;
                out->value.rowwise() += bias.row(0);
                return out;
            });
            methods.emplace_back("columns_direct_4", [&] {
                auto out = std::make_shared<Tensor<float>>();
                out->value.resize(m, n);
                parallel(col_parts, [&, out, input](int i) {
                    const int begin = n * i / col_parts, width = col_weights[i].cols();
                    auto dst = out->value.middleCols(begin, width);
                    dst.noalias() = input->value * col_weights[i];
                    dst.rowwise() += bias.row(0).segment(begin, width);
                });
                return out;
            });
            methods.emplace_back("input_rows_direct_4", [&] {
                auto out = std::make_shared<Tensor<float>>();
                out->value.resize(m, n);
                const int parts = std::min(4, m);
                parallel(parts, [&, out, input](int i) {
                    const int begin = m * i / parts, rows = m * (i + 1) / parts - begin;
                    auto dst = out->value.middleRows(begin, rows);
                    dst.noalias() = input->value.middleRows(begin, rows) * weight.value;
                    dst.rowwise() += bias.row(0);
                });
                return out;
            });
            std::vector<double> errors;
            std::vector<std::vector<double>> timings(methods.size()), batches(methods.size());
            for (auto& method : methods) {
                errors.push_back(validate(method.second, reference));
                for (int i = 0; i < 20; ++i) method.second();
            }
            // Rotate method order on each round to reduce order/thermal bias.
            // Include output destruction and allocation in each measured call.
            double checksum = 0;
            for (int round = 0; round < 7; ++round) {
                for (size_t j = 0; j < methods.size(); ++j) {
                    const size_t idx = (j + round) % methods.size();
                    const auto batch_start = Clock::now();
                    for (int i = 0; i < iterations; ++i) {
                        const auto start = Clock::now();
                        {
                            auto out = methods[idx].second();
                            checksum += out->value(0, 0);
                        }
                        timings[idx].push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
                    }
                    batches[idx].push_back(std::chrono::duration<double, std::micro>(Clock::now() - batch_start).count() / iterations);
                }
            }
            if (!std::isfinite(checksum)) throw std::runtime_error("non-finite checksum");
            for (size_t j = 0; j < methods.size(); ++j) {
                auto& times = timings[j];
                auto& batch = batches[j];
                std::sort(times.begin(), times.end());
                std::sort(batch.begin(), batch.end());
                std::cout << m << ',' << k << ',' << n << ',' << sample.count << ',' << methods[j].first << ','
                          << std::fixed << std::setprecision(3) << times[times.size() / 2] << ','
                          << times[static_cast<size_t>((times.size() - 1) * .95)] << ',' << batch[batch.size() / 2]
                          << ',' << std::scientific << errors[j] << '\n';
            }
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "linear_benchmark: " << e.what() << '\n';
    return 1;
}
