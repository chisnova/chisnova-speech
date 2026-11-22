#ifndef MERCURY_DECODER_CACHE
#define MERCURY_DECODER_CACHE

#include <vector>
#include <memory>
#include "tensor/tensor.h"
#include "json_loader/json_object.h"
#include "layer/linear.h"
#include "layer/layernorm.h"

// DecoderCache: 디코더의 key/value 캐시 관리 구조체
// - HeadCache: 단일 head의 masked key/value 캐시
// - MultiHeadCache: N개 head 캐시를 묶어 관리
// - DecoderCache: N개 layer의 MultiHeadCache를 관리 (beam search 등에서 복제 가능)

namespace MERCURY {
    // HeadCache: 단일 attention head의 key/value 캐시
    struct HeadCache {
        std::shared_ptr<Tensor<float>> maskedKeys;
        std::shared_ptr<Tensor<float>> maskedVals;

        HeadCache()
            : maskedKeys(std::make_shared<Tensor<float>>())
            , maskedVals(std::make_shared<Tensor<float>>()) {}

        HeadCache(const HeadCache& other)
            : maskedKeys(std::make_shared<Tensor<float>>())
            , maskedVals(std::make_shared<Tensor<float>>()) {
            clone(other);
        }

        // Move constructor - avoids deep copy of tensor data during beam search
        HeadCache(HeadCache&& other) noexcept
            : maskedKeys(std::move(other.maskedKeys))
            , maskedVals(std::move(other.maskedVals)) {}

        HeadCache& operator=(const HeadCache& other) {
            if (this != &other) {
                if (!maskedKeys) maskedKeys = std::make_shared<Tensor<float>>();
                if (!maskedVals) maskedVals = std::make_shared<Tensor<float>>();
                clone(other);
            }
            return *this;
        }

        HeadCache& operator=(HeadCache&& other) noexcept {
            if (this != &other) {
                maskedKeys = std::move(other.maskedKeys);
                maskedVals = std::move(other.maskedVals);
            }
            return *this;
        }

        ~HeadCache() = default;

        void clone(const HeadCache& head_cache) {
            maskedKeys->clone(head_cache.maskedKeys);
            maskedVals->clone(head_cache.maskedVals);
        }

        // Reuse existing tensor objects instead of allocating new ones
        void clear() {
            if (maskedKeys) {
                maskedKeys->value.resize(0, 0);
            } else {
                maskedKeys = std::make_shared<Tensor<float>>();
            }
            if (maskedVals) {
                maskedVals->value.resize(0, 0);
            } else {
                maskedVals = std::make_shared<Tensor<float>>();
            }
        }
    };

    // MultiHeadCache: N개 head의 캐시를 묶어 관리
    struct MultiHeadCache {
        std::vector<HeadCache> heads;

        explicit MultiHeadCache(int nhead) {
            init(nhead);
        }

        MultiHeadCache(const MultiHeadCache& other) {
            clone(other);
        }

        MultiHeadCache(MultiHeadCache&& other) noexcept
            : heads(std::move(other.heads)) {}

        MultiHeadCache& operator=(const MultiHeadCache& other) {
            if (this != &other) {
                clone(other);
            }
            return *this;
        }

        MultiHeadCache& operator=(MultiHeadCache&& other) noexcept {
            if (this != &other) {
                heads = std::move(other.heads);
            }
            return *this;
        }

        // Use vector assignment instead of element-by-element push_back
        void clone(const MultiHeadCache& multihead_cache) {
            heads = multihead_cache.heads;
        }

        void init(int nhead) {
            heads.clear();
            heads.reserve(nhead);
            for (int i = 0; i < nhead; ++i) {
                heads.emplace_back();
            }
        }

        void clear() {
            for (size_t i = 0; i < heads.size(); ++i) {
                heads[i].clear();
            }
        }
    };

    // DecoderCache: 전체 디코더의 캐시 (num_layer x nhead)
    class DecoderCache {
    public:
        DecoderCache() : num_layer(0), nhead(0) {}
        DecoderCache(const DecoderCache& other) = default;
        DecoderCache(DecoderCache&& other) noexcept
            : num_layer(other.num_layer)
            , nhead(other.nhead)
            , multihead_caches(std::move(other.multihead_caches)) {}
        DecoderCache& operator=(const DecoderCache& other) = default;
        DecoderCache& operator=(DecoderCache&& other) noexcept {
            if (this != &other) {
                num_layer = other.num_layer;
                nhead = other.nhead;
                multihead_caches = std::move(other.multihead_caches);
            }
            return *this;
        }
        ~DecoderCache() = default;

        void clear() {
            for (size_t i = 0; i < multihead_caches.size(); ++i) {
                multihead_caches[i].clear();
            }
        }

        // Fix: clear existing caches before appending to avoid duplication bug
        void clone(const std::shared_ptr<DecoderCache>& decoder_cache) {
            num_layer = decoder_cache->num_layer;
            nhead = decoder_cache->nhead;
            if (multihead_caches.size() == static_cast<size_t>(num_layer)) {
                // Reuse existing structure - just deep-copy the data
                for (int i = 0; i < num_layer; ++i) {
                    multihead_caches[i].clone(decoder_cache->multihead_caches[i]);
                }
            } else {
                // Structure mismatch - use vector assignment (copy constructor per element)
                multihead_caches = decoder_cache->multihead_caches;
            }
        }

        void init(int _num_layer, int _nhead) {
            num_layer = _num_layer;
            nhead = _nhead;
            multihead_caches.reserve(_num_layer);
            for (int i = 0; i < num_layer; ++i) {
                multihead_caches.emplace_back(nhead);
            }
        }

        int num_layer;
        int nhead;
        std::vector<MultiHeadCache> multihead_caches;
    };
} // namespace MERCURY

#endif
