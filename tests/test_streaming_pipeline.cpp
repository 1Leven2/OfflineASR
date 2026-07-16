#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

#include "offline_asr/recognizer/recognizer.h"

using namespace offline_asr;

namespace {

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::vector<float> make_sine(size_t n, int sample_rate, float freq) {
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::sin(2.0f * 3.14159f * freq * i / sample_rate) * 0.5f;
    }
    return s;
}

}  // namespace

TEST(StreamingPipeline, PartialResultNonEmpty) {
    if (!file_exists("configs/default.yaml")) GTEST_SKIP() << "config file not found";

    Recognizer recognizer("configs/default.yaml");

    auto audio = make_sine(48000, 16000, 440.0f);  // 3 seconds

    // Feed in chunks of 6400 samples (400ms)
    size_t chunk_size = 6400;
    bool had_partial = false;
    for (size_t i = 0; i < audio.size(); i += chunk_size) {
        size_t n = std::min(chunk_size, audio.size() - i);
        recognizer.AcceptWaveform(audio.data() + i, n);
        auto partial = recognizer.GetPartialResult();
        if (!partial.empty()) had_partial = true;
    }

    auto final_text = recognizer.Decode();
    EXPECT_TRUE(had_partial || !final_text.empty());
}

TEST(StreamingPipeline, ResetClearsStreamingState) {
    if (!file_exists("configs/default.yaml")) GTEST_SKIP() << "config file not found";

    Recognizer recognizer("configs/default.yaml");

    auto audio = make_sine(16000, 16000, 440.0f);
    recognizer.AcceptWaveform(audio.data(), 8000);
    auto partial1 = recognizer.GetPartialResult();

    recognizer.Reset();

    recognizer.AcceptWaveform(audio.data(), 8000);
    auto partial2 = recognizer.GetPartialResult();

    // After reset + re-feed with same input, partial results should match
    EXPECT_EQ(partial1, partial2);
}

TEST(StreamingPipeline, DecodeWithoutResetContinuesCleanly) {
    if (!file_exists("configs/default.yaml")) GTEST_SKIP() << "config file not found";

    Recognizer recognizer("configs/default.yaml");

    auto audio = make_sine(16000, 16000, 440.0f);
    recognizer.AcceptWaveform(audio.data(), audio.size());
    auto text1 = recognizer.Decode();

    // Feed again without explicit Reset — should not crash
    recognizer.AcceptWaveform(audio.data(), audio.size());
    auto text2 = recognizer.Decode();

    // Both calls should complete without crash
    SUCCEED();
}

TEST(StreamingPipeline, EmptyAudioProducesEmptyResult) {
    if (!file_exists("configs/default.yaml")) GTEST_SKIP() << "config file not found";

    Recognizer recognizer("configs/default.yaml");

    recognizer.AcceptWaveform(nullptr, 0);
    auto partial = recognizer.GetPartialResult();
    auto final_text = recognizer.Decode();

    // Empty input should not crash
    EXPECT_TRUE(partial.empty());
    EXPECT_TRUE(final_text.empty());
}

TEST(StreamingPipeline, StreamingMatchesOffline) {
    if (!file_exists("configs/default.yaml")) GTEST_SKIP() << "config file not found";

    Recognizer recognizer("configs/default.yaml");

    auto audio = make_sine(32000, 16000, 440.0f);  // 2 seconds

    // Offline recognition
    auto offline_result = recognizer.Recognize(audio.data(), audio.size());

    // Streaming recognition (need a fresh recognizer since Decode() finalizes)
    Recognizer recognizer2("configs/default.yaml");
    recognizer2.AcceptWaveform(audio.data(), audio.size());
    auto stream_text = recognizer2.Decode();

    // Both should produce the same final text (same model, same input, full audio at once)
    EXPECT_EQ(offline_result.text, stream_text);
}
