#include "callback_factory.h"
#include <iostream>

#if defined(OBJC) && defined(__OBJC__)
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#endif

namespace MERCURY {

#ifdef JAVA
DecoderCallbackPtr CallbackFactory::createJavaCallback(JNIEnv* env, jobject callback_obj) {
    if (env == nullptr || callback_obj == nullptr) {
        std::cerr << "CallbackFactory: Invalid parameters for Java callback creation" << std::endl;
        return createDefaultCallback();
    }
    
    try {
        return std::make_shared<JavaDecoderCallback>(env, callback_obj);
    } catch (...) {
        std::cerr << "CallbackFactory: Exception occurred while creating Java callback" << std::endl;
        return createDefaultCallback();
    }
}
#endif

#ifdef OBJC
DecoderCallbackPtr CallbackFactory::createObjectiveCCallback(
    id target, 
    const char* selector_name, 
    bool retain
) {
    if (target == nullptr || selector_name == nullptr) {
        std::cerr << "CallbackFactory: Invalid parameters for Objective-C callback creation" << std::endl;
        return createDefaultCallback();
    }
    
#ifdef OBJC
    try {
        // Get the selector
        SEL selector = sel_registerName(selector_name);
        if (selector == nullptr) {
            std::cerr << "CallbackFactory: Failed to register selector " << selector_name << std::endl;
            return createDefaultCallback();
        }
        
        return std::make_shared<ObjectiveCDecoderCallback>(target, selector, retain);
        
    } catch (...) {
        std::cerr << "CallbackFactory: Exception occurred while creating Objective-C callback" << std::endl;
        return createDefaultCallback();
    }
#else
    std::cerr << "CallbackFactory: Objective-C runtime not available" << std::endl;
    return createDefaultCallback();
#endif
}
#endif

#ifdef OBJC
DecoderCallbackPtr CallbackFactory::createObjectiveCCallback(
    id target, 
    SEL selector, 
    bool retain
) {
    if (target == nullptr || selector == nullptr) {
        std::cerr << "CallbackFactory: Invalid parameters for Objective-C callback creation" << std::endl;
        return createDefaultCallback();
    }
    
    try {
        return std::make_shared<ObjectiveCDecoderCallback>(target, selector, retain);
    } catch (...) {
        std::cerr << "CallbackFactory: Exception occurred while creating Objective-C callback" << std::endl;
        return createDefaultCallback();
    }
}
#endif

DecoderCallbackPtr CallbackFactory::createLinuxCallback(LinuxDecoderCallback::FunctionPtr func) {
    if (func == nullptr) {
        std::cerr << "CallbackFactory: Invalid function pointer for Linux callback creation" << std::endl;
        return createDefaultCallback();
    }
    
    try {
        return std::make_shared<LinuxDecoderCallback>(func);
    } catch (...) {
        std::cerr << "CallbackFactory: Exception occurred while creating Linux callback" << std::endl;
        return createDefaultCallback();
    }
}

DecoderCallbackPtr CallbackFactory::createLinuxCallback(LinuxDecoderCallback::LambdaFunc func) {
    if (!func) {
        std::cerr << "CallbackFactory: Invalid lambda function for Linux callback creation" << std::endl;
        return createDefaultCallback();
    }
    
    try {
        return std::make_shared<LinuxDecoderCallback>(func);
    } catch (...) {
        std::cerr << "CallbackFactory: Exception occurred while creating Linux callback" << std::endl;
        return createDefaultCallback();
    }
}

DecoderCallbackPtr CallbackFactory::createDefaultCallback() {
    return std::make_shared<DefaultCallback>();
}

} // namespace MERCURY 