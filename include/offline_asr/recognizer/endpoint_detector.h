#pragma once

#include <string>

namespace offline_asr {

class EndpointDetector {
public:
    struct Config {
        bool enabled = false;
        int silence_timeout_ms = 800;
        int decoder_stability_frames = 40;
        int max_utterance_ms = 30000;
    };

    explicit EndpointDetector(const Config& cfg);

    /** Returns true if endpoint triggered. Call each chunk. */
    bool Update(bool is_speech, int frame_ms, const std::string& current_text,
                int64_t total_audio_ms);

    bool IsEndpoint() const;
    void Reset();

private:
    Config cfg_;
    bool endpoint_ = false;
    int consecutive_silence_ms_ = 0;
    int stable_text_frames_ = 0;
    std::string last_text_;
};

}  // namespace offline_asr
