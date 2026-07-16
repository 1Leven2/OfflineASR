#include <cmath>
#include <gtest/gtest.h>

#include "offline_asr/feature/fbank_extractor.h"
#include "offline_asr/feature/streaming_fbank_extractor.h"

using namespace offline_asr;

static std::vector<float> make_sine(size_t n, int sample_rate, float freq) {
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::sin(2.0f * 3.14159f * freq * i / sample_rate) * 0.5f;
    }
    return s;
}

TEST(StreamingFbankExtractor, Construction) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);
    EXPECT_EQ(extractor.NumMelBins(), 80);
}

TEST(StreamingFbankExtractor, FeedEmptyReturnsZero) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);
    auto frames = extractor.Feed(nullptr, 0);
    EXPECT_TRUE(frames.empty());
}

TEST(StreamingFbankExtractor, FeedSilence) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);

    std::vector<float> silence(16000, 0.0f);  // 1 second
    auto frames = extractor.Feed(silence.data(), silence.size());

    int frame_length = static_cast<int>(cfg.frame_length_ms * cfg.sample_rate / 1000.0f);
    int frame_shift = static_cast<int>(cfg.frame_shift_ms * cfg.sample_rate / 1000.0f);
    int expected_frames = (16000 - frame_length) / frame_shift + 1;
    int num_frames = static_cast<int>(frames.size()) / cfg.num_mel_bins;

    EXPECT_EQ(num_frames, expected_frames);
}

TEST(StreamingFbankExtractor, FeedChunkedMatchesBatch) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor streaming(cfg);

    FbankExtractor::Config batch_cfg;
    batch_cfg.sample_rate = cfg.sample_rate;
    batch_cfg.frame_length_ms = cfg.frame_length_ms;
    batch_cfg.frame_shift_ms = cfg.frame_shift_ms;
    batch_cfg.num_mel_bins = cfg.num_mel_bins;
    batch_cfg.pre_emphasis_alpha = cfg.pre_emphasis_alpha;
    FbankExtractor batch(batch_cfg);

    auto audio = make_sine(48000, cfg.sample_rate, 440.0f);  // 3 seconds

    // Batch extraction
    auto batch_frames = batch.Extract(audio.data(), audio.size());

    // Streaming extraction (feed in chunks of 1600 samples = 100ms)
    std::vector<float> stream_frames;
    size_t chunk = 1600;
    for (size_t i = 0; i < audio.size(); i += chunk) {
        size_t n = std::min(chunk, audio.size() - i);
        auto new_frames = streaming.Feed(audio.data() + i, n);
        stream_frames.insert(stream_frames.end(), new_frames.begin(), new_frames.end());
    }
    // Flush remaining
    auto flush_frames = streaming.Flush();
    stream_frames.insert(stream_frames.end(), flush_frames.begin(), flush_frames.end());

    // Both should produce the same number of frames
    EXPECT_EQ(batch_frames.size(), stream_frames.size());

    // Feature values should be close (streaming pre-emphasis state causes minor differences)
    if (!stream_frames.empty()) {
        float max_diff = 0.0f;
        for (size_t i = 0; i < stream_frames.size(); ++i) {
            float diff = std::abs(batch_frames[i] - stream_frames[i]);
            if (diff > max_diff) max_diff = diff;
        }
        // Streaming and batch use slightly different code paths through
        // the FBank library (per-frame vs batched), so allow a small tolerance
        EXPECT_LT(max_diff, 5e-3f);
    }
}

TEST(StreamingFbankExtractor, FlushProducesFrame) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);

    int frame_length = static_cast<int>(cfg.frame_length_ms * cfg.sample_rate / 1000.0f);

    // Feed just under one frame
    std::vector<float> partial(frame_length - 1, 0.1f);
    auto frames = extractor.Feed(partial.data(), partial.size());
    EXPECT_TRUE(frames.empty());  // Not enough for a full frame

    auto flush = extractor.Flush();
    EXPECT_EQ(static_cast<int>(flush.size()), cfg.num_mel_bins);  // One zero-padded frame
}

TEST(StreamingFbankExtractor, ResetClearsState) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);

    auto audio = make_sine(16000, cfg.sample_rate, 440.0f);
    extractor.Feed(audio.data(), 8000);

    extractor.Reset();

    // After reset, should start fresh
    auto frames = extractor.Feed(audio.data(), 8000);
    EXPECT_GT(static_cast<int>(frames.size()), 0);
}

TEST(StreamingFbankExtractor, MultipleFlushSafe) {
    StreamingFbankExtractor::Config cfg;
    StreamingFbankExtractor extractor(cfg);

    // First flush on empty buffer
    auto f1 = extractor.Flush();
    EXPECT_TRUE(f1.empty());

    // Second flush after first
    auto f2 = extractor.Flush();
    EXPECT_TRUE(f2.empty());
}
