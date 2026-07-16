#include <gtest/gtest.h>
#include <cmath>

#include "offline_asr/decoder/ctc_wrapper.h"

using namespace offline_asr;

// Helper: create simple log_probs where token k has highest prob at frame t
static std::vector<float> make_log_probs(int num_frames, int vocab_size,
                                           const std::vector<int>& target) {
    std::vector<float> probs(num_frames * vocab_size, -10.0f);
    for (int t = 0; t < num_frames; ++t) {
        int token = target[t % target.size()];
        probs[t * vocab_size + token] = -0.1f;
    }
    return probs;
}

TEST(CtcDecoderWrapper, DecodeSimple) {
    CtcDecoderWrapper::Config cfg;
    cfg.beam_size = 10;
    CtcDecoderWrapper decoder(cfg);

    int vocab = 5;
    // token 1 for 3 frames, then token 2 for 3 frames
    std::vector<int> target = {1, 1, 1, 2, 2, 2};
    auto probs = make_log_probs(6, vocab, target);

    auto results = decoder.Decode(probs.data(), 6, vocab);
    ASSERT_FALSE(results.empty());
    ASSERT_EQ(results[0].tokens.size(), 2u);
    EXPECT_EQ(results[0].tokens[0], 1);
    EXPECT_EQ(results[0].tokens[1], 2);
}

TEST(CtcDecoderWrapper, BlankSkipped) {
    CtcDecoderWrapper::Config cfg;
    CtcDecoderWrapper decoder(cfg);

    int vocab = 5;
    // blank(0)-1-blank(0)-2-blank(0)
    std::vector<int> target = {0, 1, 0, 2, 0};
    auto probs = make_log_probs(5, vocab, target);

    auto results = decoder.Decode(probs.data(), 5, vocab);
    ASSERT_FALSE(results.empty());
    ASSERT_EQ(results[0].tokens.size(), 2u);
    EXPECT_EQ(results[0].tokens[0], 1);
    EXPECT_EQ(results[0].tokens[1], 2);
}

TEST(CtcDecoderWrapper, EmptyInput) {
    CtcDecoderWrapper::Config cfg;
    CtcDecoderWrapper decoder(cfg);
    auto results = decoder.Decode(nullptr, 0, 5);
    EXPECT_TRUE(results.empty());
}

TEST(CtcDecoderWrapper, AllBlank) {
    CtcDecoderWrapper::Config cfg;
    CtcDecoderWrapper decoder(cfg);

    std::vector<float> probs(5 * 5, -1.0f);
    for (int t = 0; t < 5; ++t) {
        probs[t * 5 + 0] = -0.1f;  // token 0 (blank) has highest prob
    }

    auto results = decoder.Decode(probs.data(), 5, 5);
    ASSERT_FALSE(results.empty());
    EXPECT_TRUE(results[0].tokens.empty());
}
