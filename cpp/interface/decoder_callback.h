#ifndef DECODER_CALLBACK_H
#define DECODER_CALLBACK_H

#include <memory>

namespace MERCURY {

/**
 * @brief Abstract interface for decoder callbacks across different platforms
 * 
 * This interface provides a unified way to handle decoder results
 * from different platforms (Java, Objective-C, etc.)
 */
class IDecoderCallback {
public:
    virtual ~IDecoderCallback() = default;
    
    /**
     * @brief Called when decoder produces a result
     * @param asrResult The decoded result as a string
     */
    virtual void onResult(std::string asrResult) = 0;
};

using DecoderCallbackPtr = std::shared_ptr<IDecoderCallback>;

} // namespace MERCURY

#endif // DECODER_CALLBACK_H 