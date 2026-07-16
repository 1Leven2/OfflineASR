#include <gtest/gtest.h>

#include "offline_asr/feature/fbank_extractor.h"

using namespace offline_asr;

TEST(FbankExtractor, Construction) {
    FbankExtractor::Config cfg;
    FbankExtractor extractor(cfg);
    EXPECT_EQ(extractor.NumMelBins(), 80);
}

TEST(FbankExtractor, NumFrames) {
    FbankExtractor::Config cfg;
    FbankExtractor extractor(cfg);
    // 1 second of 16kHz audio
    int32_t frames = extractor.NumFrames(16000);
    EXPECT_GT(frames, 0);
    EXPECT_LT(frames, 200);
}

TEST(FbankExtractor, ExtractSilence) {
    FbankExtractor::Config cfg;
    FbankExtractor extractor(cfg);
    std::vector<float> silence(16000, 0.0f);
    auto features = extractor.Extract(silence.data(), silence.size());
    int expected_frames = extractor.NumFrames(silence.size());
    EXPECT_EQ(static_cast<int>(features.size()), expected_frames * cfg.num_mel_bins);
}

TEST(FbankExtractor, ExtractShortAudio) {
    FbankExtractor::Config cfg;
    FbankExtractor extractor(cfg);
    std::vector<float> audio(4000, 0.0f);  // 250ms
    auto features = extractor.Extract(audio.data(), audio.size());
    EXPECT_FALSE(features.empty());
}
