#ifndef OBJECTIVEC_CALLBACK_H
#define OBJECTIVEC_CALLBACK_H

#include "decoder_callback.h"

#ifdef __OBJC__
// Objective-C 로 컴파일될 때 id / SEL 은 <objc/objc.h> 가 이미 정의한다.
// 여기서 다시 typedef 하면 "typedef redefinition with different types" 로 깨진다.
#include <objc/objc.h>
#else
// 순수 C++ 번역 단위에서는 불투명 포인터로만 다룬다
typedef void* id;
typedef void* SEL;
#endif

namespace MERCURY {

/**
 * @brief Objective-C implementation of decoder callback
 * 
 * This class wraps Objective-C callback objects and provides
 * a C++ interface for the decoder to call back to Objective-C
 */
class ObjectiveCDecoderCallback : public IDecoderCallback {
private:
    id callback_target;
    SEL callback_selector;
    bool retain_target;

public:
    /**
     * @brief Constructor
     * @param target Objective-C target object
     * @param selector Objective-C selector for the callback
     * @param retain Whether to retain the target object
     */
    ObjectiveCDecoderCallback(id target, SEL selector, bool retain = true);
    
    /**
     * @brief Destructor - releases target if retained
     */
    ~ObjectiveCDecoderCallback();
    
    /**
     * @brief Implementation of the callback interface
     * @param ids Pointer to decoded token IDs
     * @param size Number of tokens
     */
    void onResult(std::string asrResult) override;
    
    /**
     * @brief Retain an Objective-C object
     * @param obj Object to retain
     * @return Retained object
     */
    static id retainObject(id obj);
    
    /**
     * @brief Release an Objective-C object
     * @param obj Object to release
     */
    static void releaseObject(id obj);
    
    /**
     * @brief Send message to Objective-C object
     * @param target Target object
     * @param selector Selector to call
     * @param ids Token IDs array
     * @param size Number of tokens
     */
    static void sendMessage(id target, SEL selector, std::string asrResult);
};

} // namespace MERCURY

#endif // OBJECTIVEC_CALLBACK_H 