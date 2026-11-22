#ifndef _AUDIO_READER
#define _AUDIO_READER

#include "wavreader.h"
#include <vector>
#include <stdint.h>

bool isWave(const char *filename) {
    void* h_x = wav_read_open(filename);
    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res){
        // cerr << "get ref header error: " << res << endl;
        return false;
    }
    return true;
}

std::vector<float> loadWave(std::vector<int16_t> audio) {
    std::vector<float> x(audio.size());
    std::transform(audio.begin(), audio.end(), x.begin(), [](int16_t a) {return static_cast<float>(a) / 32767.f;});
    return x;
}

std::vector<int16_t> wav_read(const char *filename) {
    // void* h_x = wav_read_open("samples/p225_002.wav");
    void* h_x = wav_read_open(filename);
    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    // if (!res){
    //     cerr << "get ref header error: " << res << endl;
    //     return -1;
    // }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> wave(samples);
    res = wav_read_data(h_x, (unsigned char*)wave.data(), data_length);
    // if (res < 0){
    //     cerr << "read wav file error: " << res << endl;
    //     return -1;
    // }
    return wave;
}

std::vector<int16_t> pcm_read(const char *filename) {
    std::vector<int16_t> wave(10);
    return wave;
}

std::vector<int16_t> audio_read(const char *filename) {
    if(!isWave(filename)){
        return pcm_read(filename);
    }    
    return wav_read(filename);
}

#endif