package com.chisnova.mercury.itf;

/**
 * Callback interface for Mercury ASR decoder results
 * 
 * This interface must be implemented by any class that wants to receive
 * decoder results from the native C++ ASR system.
 * 
 * The method name "onResult" is fixed and cannot be changed as it's
 * hardcoded in the native C++ implementation.
 */
public interface DecoderCallback {
    
    /**
     * Called when the ASR decoder produces a result
     * 
     * This method is called from the native C++ code when the decoder
     * has processed audio and generated token IDs.
     * 
     * @param asrResult The decoded result as a string
     */
    void onResult(String asrResult);
} 