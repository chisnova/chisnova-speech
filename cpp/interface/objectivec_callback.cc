#include "objectivec_callback.h"
#include <iostream>

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#endif

namespace MERCURY {

ObjectiveCDecoderCallback::ObjectiveCDecoderCallback(id target, SEL selector, bool retain)
    : callback_target(target), callback_selector(selector), retain_target(retain) {
    if (retain_target && callback_target != nullptr) {
        callback_target = retainObject(callback_target);
    }
}

ObjectiveCDecoderCallback::~ObjectiveCDecoderCallback() {
    if (retain_target && callback_target != nullptr) {
        releaseObject(callback_target);
        callback_target = nullptr;
    }
}

void ObjectiveCDecoderCallback::onResult(std::string asrResult) {
    if (callback_target == nullptr || callback_selector == nullptr) {
        std::cerr << "ObjectiveCDecoderCallback: Invalid callback target or selector" << std::endl;
        return;
    }
    
    try {
        sendMessage(callback_target, callback_selector, asrResult);
    } catch (...) {
        std::cerr << "ObjectiveCDecoderCallback: Exception occurred during callback execution" << std::endl;
    }
}

// [obj retain] / [obj release] 는 ARC 번역 단위에서 금지되고, 앱 쪽은 대개 ARC 다.
// CFRetain/CFRelease 는 ARC 여부와 무관하게 같은 참조 계수를 다룬다.
id ObjectiveCDecoderCallback::retainObject(id obj) {
#ifdef __OBJC__
    if (obj != nullptr) {
        CFRetain((CFTypeRef)obj);
    }
#endif
    return obj;
}

void ObjectiveCDecoderCallback::releaseObject(id obj) {
#ifdef __OBJC__
    if (obj != nullptr) {
        CFRelease((CFTypeRef)obj);
    }
#endif
}

void ObjectiveCDecoderCallback::sendMessage(id target, SEL selector, std::string asrResult) {
#ifdef __OBJC__
    if (target == nullptr || selector == nullptr) {
        return;
    }

    // 인식 결과는 문자열 하나다. 예전에는 문자 단위 NSArray 로 넘겼는데
    // 디코더 인터페이스가 onResult(std::string) 으로 바뀐 뒤로 맞지 않는 코드였다.
    NSString* result = [[NSString alloc] initWithBytes:asrResult.data()
                                                length:asrResult.size()
                                              encoding:NSUTF8StringEncoding];
    if (result == nil) {
        // 잘린 UTF-8 등으로 변환에 실패하면 빈 문자열로 대신한다
        result = @"";
    }

    // arm64 에서는 objc_msgSend 를 실제 시그니처로 캐스팅해 불러야 한다
    ((void (*)(id, SEL, NSString*))objc_msgSend)(target, selector, result);
#else
    // Fallback for non-Objective-C compilation
    (void)target;
    (void)selector;
    (void)asrResult;
    std::cerr << "ObjectiveCDecoderCallback: Objective-C runtime not available" << std::endl;
#endif
}

} // namespace MERCURY 