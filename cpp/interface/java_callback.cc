#include "java_callback.h"
#include <iostream>

namespace MERCURY {

JavaDecoderCallback::JavaDecoderCallback(JNIEnv* env, jobject obj)
    : callback_obj(obj), callback_method(nullptr), is_global_ref(false) {
    // Get JavaVM from JNIEnv
    env->GetJavaVM(&java_vm);
    
    // Get the method ID for onResult method
    callback_method = getOnResultMethodID(env, obj);
    
    // For local references, we need to create a global reference
    callback_obj = createGlobalRef(env, obj);
    is_global_ref = true;
}

JavaDecoderCallback::JavaDecoderCallback(JNIEnv* env, jobject obj, bool is_global)
    : callback_obj(obj), callback_method(nullptr), is_global_ref(is_global) {
    // Get JavaVM from JNIEnv
    env->GetJavaVM(&java_vm);
    
    // Get the method ID for onResult method
    callback_method = getOnResultMethodID(env, obj);
}

JavaDecoderCallback::~JavaDecoderCallback() {
    if (is_global_ref && callback_obj != nullptr) {
        JNIEnv* env = getJNIEnv();
        if (env) {
            deleteGlobalRef(env, callback_obj);
        }
        callback_obj = nullptr;
    }
}

JNIEnv* JavaDecoderCallback::getJNIEnv() {
    JNIEnv* env = nullptr;
    int result = java_vm->GetEnv((void**)&env, JNI_VERSION_1_6);
    
    if (result == JNI_EDETACHED) {
        // Current thread is not attached to JVM, attach it
        // Attach thread as daemon so it does not block JVM shutdown
        // C++ 의 JavaVM 헬퍼는 GetEnv 만 void** 를 받고, Attach 계열은 JNIEnv** 를 받는다.
        // (void**) 로 캐스팅하면 NDK r27 의 jni.h 에서 타입이 맞지 않아 컴파일되지 않는다.
#if defined(__ANDROID__) || (defined(JNI_VERSION_1_6) && defined(ANDROID))
        // Some environments only support AttachCurrentThread
        result = java_vm->AttachCurrentThread(&env, nullptr);
#else
        result = java_vm->AttachCurrentThreadAsDaemon(&env, nullptr);
#endif
        if (result != JNI_OK) {
            std::cerr << "JavaDecoderCallback: Failed to attach current thread to JVM" << std::endl;
            return nullptr;
        }
    } else if (result != JNI_OK) {
        std::cerr << "JavaDecoderCallback: Failed to get JNIEnv for current thread" << std::endl;
        return nullptr;
    }
    
    return env;
}

void JavaDecoderCallback::onResult(std::string asrResult) {
    if (callback_obj == nullptr || callback_method == nullptr) {
        std::cerr << "JavaDecoderCallback: Invalid callback object or method" << std::endl;
        return;
    }
    
    // Get JNIEnv for current thread
    JNIEnv* env = getJNIEnv();
    if (env == nullptr) {
        std::cerr << "JavaDecoderCallback: Failed to get JNIEnv for callback" << std::endl;
        return;
    }
    
    try {
        // Create Java string from C++ string
        jstring java_result = env->NewStringUTF(asrResult.c_str());
        if (java_result == nullptr) {
            std::cerr << "JavaDecoderCallback: Failed to create Java string" << std::endl;
            return;
        }
        
        // Call Java callback method
        env->CallVoidMethod(callback_obj, callback_method, java_result);
        
        // Check for exceptions
        if (env->ExceptionCheck()) {
            std::cerr << "JavaDecoderCallback: Exception occurred in Java callback" << std::endl;
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        
        // Clean up local reference
        env->DeleteLocalRef(java_result);
        
    } catch (...) {
        std::cerr << "JavaDecoderCallback: Exception occurred during callback execution" << std::endl;
    }
}

jobject JavaDecoderCallback::createGlobalRef(JNIEnv* env, jobject local_obj) {
    if (local_obj == nullptr) {
        return nullptr;
    }
    return env->NewGlobalRef(local_obj);
}

void JavaDecoderCallback::deleteGlobalRef(JNIEnv* env, jobject global_obj) {
    if (global_obj != nullptr) {
        env->DeleteGlobalRef(global_obj);
    }
}

jmethodID JavaDecoderCallback::getOnResultMethodID(JNIEnv* env, jobject callback_obj) {
    if (env == nullptr || callback_obj == nullptr) {
        return nullptr;
    }
    
    // Get the class of the callback object
    jclass callback_class = env->GetObjectClass(callback_obj);
    if (callback_class == nullptr) {
        std::cerr << "JavaDecoderCallback: Failed to get Java callback class" << std::endl;
        return nullptr;
    }
    
    // Get the method ID for onResult method with signature (Ljava/lang/String;)V
    jmethodID method_id = env->GetMethodID(callback_class, "onResult", "(Ljava/lang/String;)V");
    if (method_id == nullptr) {
        std::cerr << "JavaDecoderCallback: Failed to get onResult method ID" << std::endl;
        env->DeleteLocalRef(callback_class);
        return nullptr;
    }
    
    // Clean up local reference
    env->DeleteLocalRef(callback_class);
    
    return method_id;
}

} // namespace MERCURY 