#ifndef MECURY_ASR_LIBROSA_FEATURE
#define MECURY_ASR_LIBROSA_FEATURE
#include "tensor/tensor.h"
#include "librosa.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include "json_loader/json_object.h"
#include "feature/bayesian_cmvn.h"

namespace MERCURY {
    class LibrosaFeature {
        public:
        LibrosaFeature() {}
        LibrosaFeature(JsonObject &asr_model_path, std::string base_path) {
            // mean/std 는 CMVN 을 쓰는 구형 모델만 manifest 에 기록한다.
            // 신형(conformer_stream)은 모델 내부 feat_norm 이 정규화를 담당하므로 없다.
            if (asr_model_path.obj.contains("mean") && asr_model_path.obj.contains("std")) {
                loadFeatMeanStd(asr_model_path, base_path);
                std::shared_ptr<Tensor<float>> variance = std::make_shared<Tensor<float>>();
                variance->value = feat_std->value.array().square();
                bayesian = std::make_shared<BayesianCMVNChunk<float>>(feat_mean, variance);
            }
        }

        ~LibrosaFeature() {
            clear();
        }

        void clear() {
            if (bayesian != nullptr) {
                bayesian->reset();
            }
        }

        template<typename T>
        std::shared_ptr<Tensor<T>> global_cmvn(std::shared_ptr<Tensor<T>> feature) {
            std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
            out->value = feature->value;
            // auto mean = out->value.colwise().mean();
            // auto var = (out->value.rowwise() - mean).array().square().colwise().mean();
            // auto std = (var + static_cast<T>(1e-5)).sqrt();
            for(int i=0;i<out->value.rows();++i){
                out->value.row(i) = (out->value.row(i) - feat_mean->value).array() / feat_std->value.row(0).array();
            }
            return out;
        }

        template<typename T>
        std::shared_ptr<Tensor<T>> bayesian_cmvn(std::shared_ptr<Tensor<T>> feature) {
            std::shared_ptr<Tensor<T>> out;
            out = bayesian->updateChunk(feature);
            return out;
        }

        template<typename T>
        std::shared_ptr<Tensor<T>> cmvn(std::shared_ptr<Tensor<T>> feature) {
            std::shared_ptr<Tensor<T>> out = std::make_shared<Tensor<T>>();
            out->value = feature->value;
            auto mean = out->value.colwise().mean();
            auto var = (out->value.rowwise() - mean).array().square().colwise().mean();
            auto std = (var + static_cast<T>(1e-5)).sqrt();
            out->value.array() = (out->value.rowwise() - mean).array().rowwise() / std.array();
            return out;
        }

        std::shared_ptr<Tensor<float>> getFeature(float *audio, int size) {
            std::shared_ptr<Tensor<float>> ret = std::make_shared<Tensor<float>>();
            std::vector<float> float_wave;
            for(int i=0;i<size;++i) float_wave.push_back(audio[i]);
            std::vector<float> preemphasis_wave = preemphasis(float_wave);
            std::vector<std::vector<float>> mels = LogMelSpectrogram(preemphasis_wave);
            // std::vector<std::vector<float>> stack = stackFilter(mels);
            std::vector<float> flatten_mel = flatten(mels);
            ret->setValue(flatten_mel, {int(mels.size()), int(mels[0].size())});
            // ret = cmvn(ret);
            // ret = bayesian_cmvn<float>(ret);
            // ret = global_cmvn<float>(ret);
            return ret;
        }
        
        std::vector<float> preemphasis(std::vector<float> wave) {
            std::vector<float> out;
            out.push_back(wave[0]);
            for (int i=1;i<wave.size();++i) {
                float tmp = wave[i] - 0.97*wave[i-1];
                out.push_back(tmp);
            }
            return out;
        }

        void loadFeatMeanStd(JsonObject &asr_model_path, std::string base_path) {
            feat_mean = std::make_shared<Tensor<float>>();
            feat_std = std::make_shared<Tensor<float>>();

            JsonObject feat_mean_path = asr_model_path.parseObject("mean");
            feat_mean->loadValue(base_path + "/" + feat_mean_path.parserString("value"), feat_mean_path.parserIntArray("shape"));
            JsonObject feat_std_path = asr_model_path.parseObject("std");
            feat_std->loadValue(base_path + "/" + feat_std_path.parserString("value"), feat_std_path.parserIntArray("shape"));
        }

        std::shared_ptr<Tensor<float>> getFeatNorm(std::shared_ptr<Tensor<float>> feat) {
            std::shared_ptr<Tensor<float>> out = feat;  
            out->value = out->value.array().rowwise() - feat_mean->value.row(0).array(); 
            out->value = out->value.array().rowwise() / feat_std->value.row(0).array(); 
            return out;
        }

        private:
        std::vector<float> flatten(std::vector<std::vector<float>> stack){
            std::vector<float> ret;
            for(int i=0;i<stack.size();++i){
                for(int j=0;j<stack[0].size();++j){
                    ret.push_back(stack[i][j]);
                }
            }
            return ret;
        }

        std::vector<float> loadWave(std::vector<int16_t> audio){
            std::vector<float> x(audio.size());
            std::transform(audio.begin(), audio.end(), x.begin(), [](int16_t a) {return static_cast<float>(a) / 32767.f;});
            return x;
        }

        std::vector<std::vector<float>> LogMelSpectrogram(std::vector<float> audio) {
            return librosa::Feature::logMelSpectrogram(audio, sr, n_fft, n_hop, "hann", true, "constant", 2.f, n_mel, fmin, fmax);
        }

        std::vector<std::vector<float>> MFCC(std::vector<float> audio) {
            return librosa::Feature::mfcc(audio, sr, n_fft, n_hop, "hann", true, "edge", 2.f, n_mel, fmin, fmax, n_mel, true, 2);
        }

        float max(float a, float b){
            return a>b ? a:b;
        }

        std::vector<std::vector<float>> Power2DB(std::vector<std::vector<float>> mels) {
            // return librosa::Feature::power2db(mels);
            float top_db = 80;
            float amin=1e-10;
            float ref = 1.0;
            float log_max = -987654321.0;
            float mels_max = -987654321.0;

            for(int i=0;i<mels.size();++i){
                for(int j=0;j<mels[i].size();++j){
                    mels_max = max(mels[i][j], mels_max);
                }
            }

            std::vector<std::vector<float>> ret;
            for(int i=0;i<mels.size();++i){
                std::vector<float> tmp;
                for(int j=0;j<mels[i].size();++j){
                    float val = 10.0 * log10(max(amin, mels[i][j]));
                    val -= 10.0 * log10(max(amin, mels_max));
                    log_max = max(log_max, val);
                    tmp.push_back(val);
                }
                ret.push_back(tmp);
            }

            for(int i=0;i<ret.size();++i){
                for(int j=0;j<ret[i].size();++j){
                    ret[i][j] = max(ret[i][j], log_max - top_db);
                }
            }
            return ret;
        }

        std::vector<std::vector<float>> stackFilter(std::vector<std::vector<float>> feature) {
            std::vector<std::vector<float>> ret;
            int len = feature.size();
            int seq_len = int(feature.size()/16)*16;
            int remain = int(feature.size() % 16);
            int pad_len = 16 - int(feature.size() % 16);

            for(int i=0; i < seq_len ; i+=16){
                std::vector<float> tmp;
                for(int j=0; j < 16; ++j){
                    for(int k=0; k < feature[i].size();++k){
                        tmp.push_back(feature[i+j][k]);
                    }
                }
                ret.push_back(tmp);
            }

            std::vector<float> tmp;
            for(int j=0; j < remain; ++j){
                for(int k=0; k < feature[0].size();++k){
                    tmp.push_back(feature[seq_len + j][k]);
                }
            }

            for(int j=0; j < pad_len; ++j){
                for(int k=0; k < feature[0].size();++k){
                    tmp.push_back(feature[len-1][k]);
                }
            }
            ret.push_back(tmp);

            return ret;
        }

        int sr = 16000;
        int n_fft = 400;
        int n_hop = 160;
        int n_mel = 80;
        int fmin = 0;
        int fmax = 8000;

        std::shared_ptr<Tensor<float>> feat_mean;
        std::shared_ptr<Tensor<float>> feat_std;
        std::shared_ptr<BayesianCMVNChunk<float>> bayesian;
    };
};

#endif
