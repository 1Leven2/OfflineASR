#pragma once

#include <cstddef>
#include <vector>

namespace offline_asr {

struct SpeechSegment {
    size_t start_sample = 0;
    size_t end_sample = 0;
};

class Vad {
public:
    virtual ~Vad() = default;

    virtual bool IsSpeech(const float* samples, size_t n) = 0;
    virtual void Reset() = 0;
    virtual size_t FrameSize() const = 0;

    std::vector<SpeechSegment> Segment(const float* samples, size_t n);
};

}  // namespace offline_asr
