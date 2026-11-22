#ifndef CALLBACK_FACTORY_H
#define CALLBACK_FACTORY_H

#include "decoder_callback.h"
#ifdef JAVA
#include "java_callback.h"
#endif
#ifdef OBJC
#include "objectivec_callback.h"
#endif
#include "linux_callback.h"

namespace MERCURY {

/**
 * @brief Factory class for creating platform-specific callback objects
 * 
 * This factory provides a unified way to create callback objects
 * for different platforms (Java, Objective-C, Linux, etc.)
 */
class CallbackFactory {
public:
    /**
     * @brief Create Java callback (method name is fixed to "onResult")
     * @param env JNI environment
     * @param callback_obj Java callback object implementing onResult method    
     * @return Shared pointer to callback object
     */
    #ifdef JAVA
    static DecoderCallbackPtr createJavaCallback(JNIEnv* env, jobject callback_obj);
    #endif

    /**
     * @brief Create Objective-C callback
     * @param target Objective-C target object
     * @param selector_name Selector name (e.g., "onResult:size:")
     * @param retain Whether to retain the target object
     * @return Shared pointer to callback object
     */
    #ifdef OBJC
    static DecoderCallbackPtr createObjectiveCCallback(
        id target, 
        const char* selector_name, 
        bool retain = true
    );
    
    /**
     * @brief Create Objective-C callback with pre-resolved selector
     * @param target Objective-C target object
     * @param selector Pre-resolved selector
     * @param retain Whether to retain the target object
     * @return Shared pointer to callback object
     */
    static DecoderCallbackPtr createObjectiveCCallback(
        id target, 
        SEL selector, 
        bool retain = true
    );
    #endif
    /**
     * @brief Create Linux callback with function pointer
     * @param func C-style function pointer
     * @return Shared pointer to callback object
     */
    static DecoderCallbackPtr createLinuxCallback(LinuxDecoderCallback::FunctionPtr func);
    
    /**
     * @brief Create Linux callback with lambda function
     * @param func Lambda or std::function callback
     * @return Shared pointer to callback object
     */
    static DecoderCallbackPtr createLinuxCallback(LinuxDecoderCallback::LambdaFunc func);
    
    /**
     * @brief Create Linux callback with member function pointer
     * @param obj Object pointer
     * @param func Member function pointer
     * @return Shared pointer to callback object
     */
    template<typename T>
    static DecoderCallbackPtr createLinuxCallback(T* obj, void (T::*func)(std::string)) {
        return std::make_shared<LinuxDecoderCallback>(obj, func);
    }
    
    /**
     * @brief Create a default callback (for testing or when no callback is needed)
     * @return Shared pointer to default callback object
     */
    static DecoderCallbackPtr createDefaultCallback();
    
private:
    /**
     * @brief Default callback implementation for testing
     */
    class DefaultCallback : public IDecoderCallback {
    public:
        void onResult(std::string asrResult) {
            // Do nothing
        }
    };
};

} // namespace MERCURY

#endif // CALLBACK_FACTORY_H 