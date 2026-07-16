#pragma once

#include "offline_asr/vad/vad.h"

namespace offline_asr {

class EnergyVad : public Vad {
public:
    struct Config {
        float threshold = 0.01f;
        int frame_ms = 30;
        int min_speech_ms = 100;
        int min_silence_ms = 200;
        int sample_rate = 16000;
    };

    explicit EnergyVad(const Config& cfg);

    bool IsSpeech(const float* samples, size_t n) override;
    void Reset() override;
    size_t FrameSize() const override { return static_cast<size_t>(frame_size_); }

private:
    static float ComputeRms(const float* samples, size_t n);

    enum class State { kSilence, kSpeech, kHangover };

    Config cfg_;
    int frame_size_;
    int min_speech_frames_;
    int min_silence_frames_;

    State state_ = State::kSilence;
    int speech_frame_count_ = 0;
    int silence_frame_count_ = 0;
};

}  // namespace offline_asr
