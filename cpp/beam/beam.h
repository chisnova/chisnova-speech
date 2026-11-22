#ifndef MECURY_ASR_BEAM
#define MECURY_ASR_BEAM

#include "tensor/tensor.h"
#include "module/decoder_cache.h"
#include <utility>

namespace MERCURY {
    class Beam{
    public:
        Beam(){
            sos = 2;
            eos = 3;
            cache = std::make_shared<DecoderCache>();
            cache->clear();
            score = 0.0;
            position = 0;
            isFinal = false;
            normscore=0;
            cur_score=0.0;
        }

        Beam(int _sos, int _eos){
            sos = _sos;
            eos = _eos;
            cache = std::make_shared<DecoderCache>();
            cache->clear();
            score = 0.0;
            position = 0;
            isFinal = false;
            normscore=0;
        }

        Beam(const Beam& other){
            clone(other);
        }

        // Move constructor - critical for beam search performance
        Beam(Beam&& other) noexcept
            : word_ids(std::move(other.word_ids))
            , score(other.score)
            , normscore(other.normscore)
            , position(other.position)
            , isFinal(other.isFinal)
            , sos(other.sos)
            , eos(other.eos)
            , cache(std::move(other.cache))
            , cur_score(other.cur_score) {}

        Beam& operator=(const Beam& other) {
            if (this != &other) {
                word_ids = other.word_ids;
                score = other.score;
                normscore = other.normscore;
                position = other.position;
                isFinal = other.isFinal;
                sos = other.sos;
                eos = other.eos;
                cur_score = other.cur_score;
                if (!cache) {
                    cache = std::make_shared<DecoderCache>();
                }
                cache->clone(other.cache);
            }
            return *this;
        }

        Beam& operator=(Beam&& other) noexcept {
            if (this != &other) {
                word_ids = std::move(other.word_ids);
                score = other.score;
                normscore = other.normscore;
                position = other.position;
                isFinal = other.isFinal;
                sos = other.sos;
                eos = other.eos;
                cache = std::move(other.cache);
                cur_score = other.cur_score;
            }
            return *this;
        }

        ~Beam(){
            cache = nullptr;
        }

        void workerIncremental(const Beam &org, int idx, float _score) {
            isFinal = org.isFinal;
            if(idx == eos){
                isFinal = true;
            }
            position = org.position + 1;
            score = org.score + _score;
            // Vector assignment + push_back instead of element-by-element copy
            word_ids = org.word_ids;
            word_ids.push_back(idx);
            cur_score = _score;

            if (cache == nullptr){
                cache = std::make_shared<DecoderCache>();
            }

            cache->clone(org.cache);
            normScore();
        }

        void stopIncremental(const Beam &org, float _score, std::shared_ptr<DecoderCache> stop_cache) {
            isFinal = org.isFinal;
            position = org.position;
            score = org.score + _score;
            cur_score = _score;

            // Vector assignment instead of element-by-element copy
            word_ids = org.word_ids;

            if (cache == nullptr){
                cache = std::make_shared<DecoderCache>();
            }
            cache->clone(stop_cache);
            normScore();
        }

        // Guard against division by zero on empty beams
        void normScore() {
            normscore = word_ids.empty() ? score : float(score/word_ids.size());
        }


        bool operator<(const Beam &org) const {
            return this->normscore < org.normscore;
        }

        void clone(const Beam &org) {
            isFinal = org.isFinal;
            position = org.position;
            score = org.score;
            normscore = org.normscore;
            cur_score = org.cur_score;
            // Vector assignment instead of element-by-element copy
            word_ids = org.word_ids;
            if (!cache) {
                cache = std::make_shared<DecoderCache>();
            }
            cache->clone(org.cache);
        }

        void makeInitBeam(int num_dec_layer){
            word_ids.push_back(sos);
            // word_score_context.push_back(1.0);
            cache->init(num_dec_layer, 4);
            score = 0.0;
            position = 0;
            normscore = 0.0;
        }
        
        void pushContext(int idx){
            word_ids.push_back(idx);
            if(idx == eos){
                isFinal = true;
            }
        }

        void print(){
            #ifdef DEBUG
            std::string ids_string = "[";
            for(int i=0;i<word_ids.size();++i) {
                ids_string += std::to_string(word_ids[i]) + std::string(", ");
            }
            int length = ids_string.length();
            ids_string = ids_string.substr(0, length-2) + std::string("]");

            LoggerManager::Log("DEBUG", " context: " + ids_string);
            LoggerManager::Log("DEBUG", " score : " + std::to_string(score));
            LoggerManager::Log("DEBUG", " cur_score : " + std::to_string(cur_score));
            LoggerManager::Log("DEBUG", " normscore : " + std::to_string(normscore));
            LoggerManager::Log("DEBUG", " position : " + std::to_string(position));
            LoggerManager::Log("DEBUG", " cache num layer: " + std::to_string(cache->num_layer));
            LoggerManager::Log("DEBUG", " cache num head: " + std::to_string(cache->nhead));
            LoggerManager::Log("DEBUG", " cache multihead_caches size: " + std::to_string(cache->multihead_caches.size()));
            LoggerManager::Log("DEBUG", " cache multihead_caches[0].heads size: " + std::to_string(cache->multihead_caches[0].heads.size()));
            int k_cache_size = cache->multihead_caches[0].heads[0].maskedKeys->value.size();
            int v_cache_size = cache->multihead_caches[0].heads[0].maskedVals->value.size();

            LoggerManager::Log("DEBUG", std::string(" cache multihead_caches[0].heads[0].maskedKeys->value.size(): [") + std::to_string(k_cache_size/64) + ", " + std::to_string(64) + "]");
            LoggerManager::Log("DEBUG", std::string(" cache multihead_caches[0].heads[0].maskedVals->value.size(): [") + std::to_string(v_cache_size/64) + ", " + std::to_string(64) + "]");
            #endif
        }

        void print(std::vector<std::string> vocab) {
            std::string ids_string = "[";
            for(int i=0;i<word_ids.size();++i) {
                ids_string += std::to_string(word_ids[i]) + std::string(", ");
            }

            std::string result="";
            for(int i=0;i<word_ids.size();++i){
                result += vocab[word_ids[i]];
            }

            int length = ids_string.length();
            ids_string = ids_string.substr(0, length-2) + std::string("]");
            std::cout << ids_string << std::endl;
            std::cout << result << std::endl;
            std::cout << "score : " << score << std::endl;
            std::cout << "position : " << position << std::endl;
            std::cout << "cache num layer: " << cache->num_layer << std::endl;
            if(cache->multihead_caches[0].heads[0].maskedKeys->value.size() > 0){
                std::cout << "cache dec layer1 head1 shape[0]: " << cache->multihead_caches[0].heads[0].maskedKeys->shape(0) << std::endl;
                std::cout << "cache dec layer1 head1 shape[1]: " << cache->multihead_caches[0].heads[0].maskedKeys->shape(1) << std::endl;
                std::cout << "================" << std::endl;
            }
        }
        std::vector<int> word_ids;
        // std::vector<int> phone_idx;
        // std::vector<float> word_score_context;
        // std::vector<float> phone_score_context;
        float score;
        float normscore;
        int position;
        bool isFinal;
        int sos;
        int eos;
        std::shared_ptr<DecoderCache> cache;
        float cur_score = 0.0f;
    };
};
#endif
