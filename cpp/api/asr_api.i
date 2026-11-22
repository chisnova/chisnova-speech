
%module asr_api

%include "std_string.i"
%include "arrays_java.i"
%include "std_vector.i"
%include "typemaps.i"

%{
#include "asr_api.h"
%}

%ignore MERCURY::AsrAPI::context_manager;
namespace MERCURY { typedef std::shared_ptr<IDecoderCallback> DecoderCallbackPtr; }

// Keep the Java callback mapping in the generator, not in hand-edited output.
%typemap(jni) MERCURY::DecoderCallbackPtr "jobject"
%typemap(jtype) MERCURY::DecoderCallbackPtr "com.chisnova.mercury.itf.DecoderCallback"
%typemap(jstype) MERCURY::DecoderCallbackPtr "com.chisnova.mercury.itf.DecoderCallback"
%typemap(javain) MERCURY::DecoderCallbackPtr "$javainput"
%typemap(in) MERCURY::DecoderCallbackPtr {
    try {
        $1 = MERCURY::CallbackFactory::createJavaCallback(jenv, $input);
        if (!$1) throw std::invalid_argument("null decoder callback");
    } catch (const std::exception& e) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
        return $null;
    }
}

// Inputs are copied into RAII buffers. A rejected later argument or a thrown
// native exception must not leak copied arrays or leave JVM arrays pinned.
%define INPUT_ARRAY(TYPE, JNI_TYPE, REGION)
%typemap(in) TYPE[] (std::vector<TYPE> storage) {
    if (!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null input array");
        return $null;
    }
    try {
        storage.resize(jenv->GetArrayLength($input));
    } catch (const std::exception& e) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
        return $null;
    }
    if (!storage.empty()) jenv->REGION($input, 0, storage.size(), storage.data());
    if (jenv->ExceptionCheck()) return $null;
    $1 = storage.data();
}
%typemap(argout) TYPE[] ""
%typemap(freearg) TYPE[] ""
%enddef
INPUT_ARRAY(int, jint, GetIntArrayRegion)
INPUT_ARRAY(float, jfloat, GetFloatArrayRegion)

%typemap(javacode) MERCURY::AsrAPI %{
  public void sendAudio(float[] audio, int size) {
    sendAudio(audio, size, false);
  }
%}

// 네이티브가 던진 예외를 자바 예외로 바꾼다.
//
// 이것이 없으면 C++ 예외가 JNI 경계를 그대로 넘어가 std::terminate 로 프로세스가 죽는다.
// 모델 파일이 잘려 있을 때가 그렇다 — tensor.cc 가 "truncated tensor file" 을 던지는데,
// 그것을 아무도 받지 않아 앱이 통째로 내려앉는다. 자바 예외로 바꿔야 AsrController 의
// try/catch 가 받아 "초기화 실패" 로 처리할 수 있다.
%exception {
    try {
        $action
    } catch (const std::exception &e) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
        return $null;
    } catch (...) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "unknown native error");
        return $null;
    }
}

%exception MERCURY::AsrAPI::sendAudio {
    try {
        if (jarg3 < 0 || jarg3 > jenv->GetArrayLength(jarg2))
            throw std::invalid_argument("audio size exceeds array length");
        $action
    } catch (const std::exception& e) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
        return $null;
    } catch (...) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "unknown native error");
        return $null;
    }
}

%exception MERCURY::AsrAPI::getPronounceScore {
    try {
        if (jarg5 < 2 || jarg8 < 2 ||
            jarg5 != jenv->GetArrayLength(jarg3) || jarg5 != jenv->GetArrayLength(jarg4) ||
            jarg8 != jenv->GetArrayLength(jarg6) || jarg8 != jenv->GetArrayLength(jarg7))
            throw std::invalid_argument("pronunciation array lengths do not match");
        $action
    } catch (const std::exception& e) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, e.what());
        return $null;
    } catch (...) {
        SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, "unknown native error");
        return $null;
    }
}

%include "asr_api.h"
