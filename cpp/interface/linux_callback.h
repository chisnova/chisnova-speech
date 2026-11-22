#ifndef LINUX_CALLBACK_H
#define LINUX_CALLBACK_H

#include "decoder_callback.h"
#include <functional>
#include <memory>
#include <string>

namespace MERCURY {

/**
 * @brief Linux implementation of decoder callback using function pointers and lambdas
 * 
 * This class provides a flexible callback mechanism for Linux systems
 * supporting both function pointers and lambda functions
 */
class LinuxDecoderCallback : public IDecoderCallback {
public:
    // Function pointer type for C-style callbacks
    using FunctionPtr = void (*)(std::string);
    
    // Lambda function type for modern C++ callbacks
    using LambdaFunc = std::function<void(std::string)>;

private:
    LambdaFunc callback_func;
    bool is_function_ptr;

public:
    /**
     * @brief Constructor with function pointer
     * @param func C-style function pointer
     */
    explicit LinuxDecoderCallback(FunctionPtr func);
    
    /**
     * @brief Constructor with lambda function
     * @param func Lambda or std::function callback
     */
    explicit LinuxDecoderCallback(LambdaFunc func);
    
    /**
     * @brief Constructor with member function pointer
     * @param obj Object pointer
     * @param func Member function pointer
     */
    template<typename T>
    LinuxDecoderCallback(T* obj, void (T::*func)(std::string))
        : callback_func([obj, func](std::string asrResult) { (obj->*func)(asrResult); })
        , is_function_ptr(false) {}
    
    /**
     * @brief Destructor
     */
    ~LinuxDecoderCallback() = default;
    
    /**
     * @brief Implementation of the callback interface
     * @param ids Pointer to decoded token IDs
     * @param size Number of tokens
     */
    void onResult(std::string asrResult) override;
    
    /**
     * @brief Check if callback is valid
     * @return true if callback is valid, false otherwise
     */
    bool isValid() const;
    
    /**
     * @brief Reset callback to null
     */
    void reset();
    
    /**
     * @brief Set new function pointer callback
     * @param func New function pointer
     */
    void setCallback(FunctionPtr func);
    
    /**
     * @brief Set new lambda callback
     * @param func New lambda function
     */
    void setCallback(LambdaFunc func);
};

} // namespace MERCURY

#endif // LINUX_CALLBACK_H 