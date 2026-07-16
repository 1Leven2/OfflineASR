#include "offline_asr/audio/wav_reader.h"

#include <spdlog/spdlog.h>
#include <sndfile.h>

namespace offline_asr {

WavReader::AudioData WavReader::Read(const std::string& path) {
    SF_INFO sf_info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        spdlog::error("Failed to open audio file: {} ({})", path, sf_strerror(nullptr));
        return {};
    }

    std::vector<float> samples(static_cast<size_t>(sf_info.frames) * sf_info.channels);
    sf_readf_float(file, samples.data(), sf_info.frames);
    sf_close(file);

    // 多声道 → 单声道
    std::vector<float> mono;
    if (sf_info.channels == 1) {
        mono = std::move(samples);
    } else {
        mono.resize(sf_info.frames);
        for (sf_count_t i = 0; i < sf_info.frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < sf_info.channels; ++ch) {
                sum += samples[i * sf_info.channels + ch];
            }
            mono[i] = sum / sf_info.channels;
        }
    }

    return {mono, sf_info.samplerate, sf_info.channels};
}

WavReader::AudioData WavReader::Read(const void* data, size_t size) {
    SF_VIRTUAL_IO vio{};
    vio.get_filelen = [](void*) -> sf_count_t { return 0; };  // 未实现
    vio.seek = [](sf_count_t, int, void*) -> sf_count_t { return 0; };
    vio.read = [](void* buf, sf_count_t count, void* user) -> sf_count_t {
        return 0;  // 未实现
    };
    vio.tell = [](void*) -> sf_count_t { return 0; };

    SF_INFO sf_info{};
    SNDFILE* file = sf_open_virtual(&vio, SFM_READ, &sf_info,
                                     const_cast<void*>(data));
    if (!file) {
        spdlog::error("Failed to open audio from memory: {}", sf_strerror(nullptr));
        return {};
    }

    std::vector<float> samples(static_cast<size_t>(sf_info.frames) * sf_info.channels);
    sf_readf_float(file, samples.data(), sf_info.frames);
    sf_close(file);

    if (sf_info.channels == 1) {
        return {samples, sf_info.samplerate, sf_info.channels};
    }

    std::vector<float> mono(sf_info.frames);
    for (sf_count_t i = 0; i < sf_info.frames; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < sf_info.channels; ++ch) {
            sum += samples[i * sf_info.channels + ch];
        }
        mono[i] = sum / sf_info.channels;
    }
    return {mono, sf_info.samplerate, sf_info.channels};
}

}  // namespace offline_asr
