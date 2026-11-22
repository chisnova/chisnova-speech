#ifndef JAVA_CALLBACK_H
#define JAVA_CALLBACK_H

#include "decoder_callback.h"
#include <jni.h>

namespace MERCURY {

/**
 * @brief Java implementation of decoder callback using JNI
 * 
 * This class wraps Java callback objects and provides
 * a C++ interface for the decoder to call back to Java.
 * The callback method is fixed to "onResult" with signature "(Ljava/lang/String;)V"
 */
class JavaDecoderCallback : public IDecoderCallback {
private:
    JavaVM* java_vm;
    jobject callback_obj;
    jmethodID callback_method;
    bool is_global_ref;

public:
    /**
     * @brief Constructor with local reference
     * @param env JNI environment
     * @param obj Java callback object (local reference)
     */
    JavaDecoderCallback(JNIEnv* env, jobject obj);
    
    /**
     * @brief Constructor with global reference
     * @param env JNI environment
     * @param obj Java callback object (global reference)
     * @param is_global Whether the object is a global reference
     */
    JavaDecoderCallback(JNIEnv* env, jobject obj, bool is_global);
    
    /**
     * @brief Destructor - cleans up JNI references
     */
    ~JavaDecoderCallback();
    
    /**
     * @brief Implementation of the callback interface
     * @param ids Pointer to decoded token IDs
     * @param size Number of tokens
     */
    void onResult(std::string asrResult) override;
    
    /**
     * @brief Get JNIEnv for current thread
     * @return JNIEnv* for current thread, or nullptr if failed
     */
    JNIEnv* getJNIEnv();
    
    /**
     * @brief Create a global reference from local reference
     * @param env JNI environment
     * @param local_obj Local reference to convert
     * @return Global reference
     */
    static jobject createGlobalRef(JNIEnv* env, jobject local_obj);
    
    /**
     * @brief Delete a global reference
     * @param env JNI environment
     * @param global_obj Global reference to delete
     */
    static void deleteGlobalRef(JNIEnv* env, jobject global_obj);
    
    /**
     * @brief Get the method ID for onResult method
     * @param env JNI environment
     * @param callback_obj Java callback object
     * @return Method ID for onResult method
     */
    static jmethodID getOnResultMethodID(JNIEnv* env, jobject callback_obj);
};

} // namespace MERCURY

#endif // JAVA_CALLBACK_H 