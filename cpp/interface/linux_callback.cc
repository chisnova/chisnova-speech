#include "linux_callback.h"
#include <iostream>

namespace MERCURY {

LinuxDecoderCallback::LinuxDecoderCallback(FunctionPtr func)
    : callback_func(func), is_function_ptr(true) {
}

LinuxDecoderCallback::LinuxDecoderCallback(LambdaFunc func)
    : callback_func(func), is_function_ptr(false) {
}

void LinuxDecoderCallback::onResult(std::string asrResult) {
    if (!isValid()) {
        std::cerr << "LinuxDecoderCallback: Invalid callback function" << std::endl;
        return;
    }
    
    try {
        callback_func(asrResult);
    } catch (const std::exception& e) {
        std::cerr << "LinuxDecoderCallback: Exception occurred: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "LinuxDecoderCallback: Unknown exception occurred during callback execution" << std::endl;
    }
}

bool LinuxDecoderCallback::isValid() const {
    return static_cast<bool>(callback_func);
}

void LinuxDecoderCallback::reset() {
    callback_func = nullptr;
    is_function_ptr = false;
}

void LinuxDecoderCallback::setCallback(FunctionPtr func) {
    callback_func = func;
    is_function_ptr = true;
}

void LinuxDecoderCallback::setCallback(LambdaFunc func) {
    callback_func = func;
    is_function_ptr = false;
}

} // namespace MERCURY 