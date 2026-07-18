#include "offline_asr/recognizer/endpoint_detector.h"

namespace offline_asr {

EndpointDetector::EndpointDetector(const Config& cfg) : cfg_(cfg) {}

bool EndpointDetector::Update(bool is_speech, int frame_ms,
                              const std::string& current_text,
                              int64_t total_audio_ms) {
    if (!cfg_.enabled || endpoint_)
        return endpoint_;

    // Rule 1: consecutive silence timeout
    if (!is_speech) {
        consecutive_silence_ms_ += frame_ms;
        if (consecutive_silence_ms_ >= cfg_.silence_timeout_ms) {
            endpoint_ = true;
            return true;
        }
    } else {
        consecutive_silence_ms_ = 0;
    }

    // Rule 2: decoder stability (text unchanged for N frames)
    if (!current_text.empty() && current_text == last_text_) {
        if (stable_text_frames_ == 0)
            stable_text_frames_ = 1;
        else
            stable_text_frames_++;
        if (stable_text_frames_ >= cfg_.decoder_stability_frames) {
            endpoint_ = true;
            return true;
        }
    } else {
        stable_text_frames_ = 0;
        last_text_ = current_text;
    }

    // Rule 3: max utterance duration
    if (total_audio_ms >= cfg_.max_utterance_ms) {
        endpoint_ = true;
        return true;
    }

    return false;
}

bool EndpointDetector::IsEndpoint() const { return endpoint_; }

void EndpointDetector::Reset() {
    endpoint_ = false;
    consecutive_silence_ms_ = 0;
    stable_text_frames_ = 0;
    last_text_.clear();
}

}  // namespace offline_asr
