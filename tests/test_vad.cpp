#include <cmath>
#include <gtest/gtest.h>

#include "offline_asr/vad/energy_vad.h"

using namespace offline_asr;

namespace {

static std::vector<float> make_sine(size_t n, int sample_rate, float freq,
                                    float amplitude = 0.5f) {
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::sin(2.0f * 3.14159f * freq * i / sample_rate) * amplitude;
    }
    return s;
}

static std::vector<float> make_silence(size_t n) {
    return std::vector<float>(n, 0.0f);
}

}  // namespace

TEST(EnergyVad, FrameSize) {
    EnergyVad::Config cfg;
    cfg.sample_rate = 16000;
    cfg.frame_ms = 30;
    EnergyVad vad(cfg);
    EXPECT_EQ(vad.FrameSize(), 480u);
}

TEST(EnergyVad, SilenceIsNotSpeech) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);
    auto frame = make_silence(cfg.frame_ms * cfg.sample_rate / 1000);
    EXPECT_FALSE(vad.IsSpeech(frame.data(), frame.size()));
}

TEST(EnergyVad, LoudToneIsSpeech) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);
    auto frame = make_sine(cfg.frame_ms * cfg.sample_rate / 1000,
                           cfg.sample_rate, 440.0f, 0.5f);
    EXPECT_TRUE(vad.IsSpeech(frame.data(), frame.size()));
}

TEST(EnergyVad, ThresholdBoundary) {
    EnergyVad::Config cfg;
    cfg.threshold = 0.05f;
    EnergyVad vad(cfg);

    int frame_size = cfg.frame_ms * cfg.sample_rate / 1000;
    auto loud = make_sine(frame_size, cfg.sample_rate, 440.0f, 0.5f);
    auto quiet = make_sine(frame_size, cfg.sample_rate, 440.0f, 0.005f);

    EXPECT_TRUE(vad.IsSpeech(loud.data(), loud.size()));
    vad.Reset();
    EXPECT_FALSE(vad.IsSpeech(quiet.data(), quiet.size()));
}

TEST(EnergyVad, HangoverExtendsSpeech) {
    EnergyVad::Config cfg;
    cfg.min_speech_ms = 100;
    cfg.min_silence_ms = 200;
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    auto speech_frame = make_sine(frame_size, cfg.sample_rate, 440.0f, 0.5f);
    auto silence_frame = make_silence(frame_size);

    // Feed several speech frames to pass min_speech_ms
    for (int i = 0; i < 6; ++i) {
        vad.IsSpeech(speech_frame.data(), speech_frame.size());
    }

    // Feed silence — should enter hangover, still returning true
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(vad.IsSpeech(silence_frame.data(), silence_frame.size()));
    }
}

TEST(EnergyVad, RejectsNoiseBurst) {
    EnergyVad::Config cfg;
    cfg.min_speech_ms = 100;  // 100ms
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    auto speech_frame = make_sine(frame_size, cfg.sample_rate, 440.0f, 0.5f);
    auto silence_frame = make_silence(frame_size);

    // Feed just 1 speech frame (< 100ms) then silence
    vad.IsSpeech(speech_frame.data(), speech_frame.size());
    for (int i = 0; i < 3; ++i) {
        vad.IsSpeech(silence_frame.data(), silence_frame.size());
    }

    // Should be back to silence (short burst rejected as noise)
    vad.Reset();
    EXPECT_FALSE(vad.IsSpeech(silence_frame.data(), silence_frame.size()));
}

TEST(EnergyVad, ResetClearsState) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    auto speech_frame = make_sine(frame_size, cfg.sample_rate, 440.0f, 0.5f);
    auto silence_frame = make_silence(frame_size);

    // Feed speech to change internal state
    for (int i = 0; i < 10; ++i) {
        vad.IsSpeech(speech_frame.data(), speech_frame.size());
    }

    vad.Reset();

    // After reset, silence should not be flagged as speech
    EXPECT_FALSE(vad.IsSpeech(silence_frame.data(), silence_frame.size()));
}

TEST(EnergyVad, EmptyInput) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);
    EXPECT_FALSE(vad.IsSpeech(nullptr, 0));
}

TEST(EnergyVad, MultiFrameChunk) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    // 5 frames worth of speech in a single call
    auto chunk = make_sine(frame_size * 5, cfg.sample_rate, 440.0f, 0.5f);
    EXPECT_TRUE(vad.IsSpeech(chunk.data(), chunk.size()));
}

TEST(EnergyVad, SegmentEmpty) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);

    auto audio = make_silence(cfg.sample_rate);  // 1 second silence
    auto segments = vad.Segment(audio.data(), audio.size());
    EXPECT_TRUE(segments.empty());
}

TEST(EnergyVad, SegmentSingle) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    int sr = cfg.sample_rate;

    // 0.5s silence + 1s speech + 0.5s silence
    auto silence1 = make_silence(sr / 2);
    auto speech = make_sine(sr, sr, 440.0f, 0.5f);
    auto silence2 = make_silence(sr / 2);

    std::vector<float> audio;
    audio.insert(audio.end(), silence1.begin(), silence1.end());
    audio.insert(audio.end(), speech.begin(), speech.end());
    audio.insert(audio.end(), silence2.begin(), silence2.end());

    auto segments = vad.Segment(audio.data(), audio.size());
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_GT(segments[0].start_sample, 0u);
    EXPECT_LT(segments[0].end_sample, audio.size());
}

TEST(EnergyVad, SegmentTwoUtterances) {
    EnergyVad::Config cfg;
    cfg.min_silence_ms = 400;
    EnergyVad vad(cfg);

    int sr = cfg.sample_rate;

    // 0.5s speech + 0.5s silence + 0.5s speech
    auto speech1 = make_sine(sr / 2, sr, 440.0f, 0.5f);
    auto silence = make_silence(sr / 2);
    auto speech2 = make_sine(sr / 2, sr, 880.0f, 0.5f);

    std::vector<float> audio;
    audio.insert(audio.end(), speech1.begin(), speech1.end());
    audio.insert(audio.end(), silence.begin(), silence.end());
    audio.insert(audio.end(), speech2.begin(), speech2.end());

    auto segments = vad.Segment(audio.data(), audio.size());
    EXPECT_EQ(segments.size(), 2u);
}

TEST(EnergyVad, SegmentMergesCloseUtterances) {
    EnergyVad::Config cfg;
    cfg.min_silence_ms = 500;  // 500ms — longer than the gap
    EnergyVad vad(cfg);

    int sr = cfg.sample_rate;

    // 0.5s speech + 0.1s silence (< min_silence_ms) + 0.5s speech
    auto speech1 = make_sine(sr / 2, sr, 440.0f, 0.5f);
    auto short_silence = make_silence(sr / 10);
    auto speech2 = make_sine(sr / 2, sr, 880.0f, 0.5f);

    std::vector<float> audio;
    audio.insert(audio.end(), speech1.begin(), speech1.end());
    audio.insert(audio.end(), short_silence.begin(), short_silence.end());
    audio.insert(audio.end(), speech2.begin(), speech2.end());

    auto segments = vad.Segment(audio.data(), audio.size());
    EXPECT_EQ(segments.size(), 1u);  // merged into one
}

TEST(EnergyVad, PartialFrameTrailingIncluded) {
    EnergyVad::Config cfg;
    EnergyVad vad(cfg);

    int frame_size = static_cast<int>(vad.FrameSize());
    // 2.5 frames worth of speech — trailing partial frame included in segment
    auto audio = make_sine(frame_size * 2 + frame_size / 2,
                           cfg.sample_rate, 440.0f, 0.5f);
    auto segments = vad.Segment(audio.data(), audio.size());
    EXPECT_EQ(segments.size(), 1u);
    // The in_segment flag is true at end, so the full buffer is included
    EXPECT_EQ(segments[0].end_sample, audio.size());
}
