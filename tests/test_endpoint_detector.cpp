#include <gtest/gtest.h>

#include "offline_asr/recognizer/endpoint_detector.h"

using namespace offline_asr;

TEST(EndpointDetector, SilenceTimeout) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.silence_timeout_ms = 500;
    EndpointDetector ed(cfg);

    // Speech frames — no endpoint
    for (int i = 0; i < 5; ++i)
        EXPECT_FALSE(ed.Update(true, 100, "hello", 500 + i * 100));

    // Silence frames — should trigger after 500ms
    EXPECT_FALSE(ed.Update(false, 100, "hello", 1000));
    EXPECT_FALSE(ed.Update(false, 100, "hello", 1100));
    EXPECT_FALSE(ed.Update(false, 100, "hello", 1200));
    EXPECT_FALSE(ed.Update(false, 100, "hello", 1300));
    EXPECT_TRUE(ed.Update(false, 100, "hello", 1400));
    EXPECT_TRUE(ed.IsEndpoint());
}

TEST(EndpointDetector, DecoderStability) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.decoder_stability_frames = 4;
    cfg.silence_timeout_ms = 999999;  // effectively disable silence rule
    EndpointDetector ed(cfg);

    // Speech with changing text — no endpoint
    for (int i = 0; i < 3; ++i)
        EXPECT_FALSE(ed.Update(true, 100, "text_" + std::to_string(i), i * 100));

    // First "stable" changes text → resets counter. Then 4 more → counter=4 → triggers
    for (int i = 0; i < 4; ++i)
        EXPECT_FALSE(ed.Update(true, 100, "stable", 300 + i * 100));
    EXPECT_TRUE(ed.Update(true, 100, "stable", 700));
}

TEST(EndpointDetector, MaxUtterance) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.max_utterance_ms = 2000;
    cfg.silence_timeout_ms = 999999;
    cfg.decoder_stability_frames = 999999;
    EndpointDetector ed(cfg);

    EXPECT_FALSE(ed.Update(true, 100, "text", 1900));
    EXPECT_TRUE(ed.Update(true, 100, "text", 2000));
}

TEST(EndpointDetector, NoSpeech) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.silence_timeout_ms = 300;
    EndpointDetector ed(cfg);

    // Pure silence from the start
    EXPECT_FALSE(ed.Update(false, 100, "", 100));
    EXPECT_FALSE(ed.Update(false, 100, "", 200));
    EXPECT_TRUE(ed.Update(false, 100, "", 300));
}

TEST(EndpointDetector, Reset) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.silence_timeout_ms = 300;
    EndpointDetector ed(cfg);

    EXPECT_FALSE(ed.Update(false, 100, "", 100));
    EXPECT_FALSE(ed.Update(false, 100, "", 200));
    EXPECT_TRUE(ed.Update(false, 100, "", 300));
    EXPECT_TRUE(ed.IsEndpoint());

    ed.Reset();
    EXPECT_FALSE(ed.IsEndpoint());
    EXPECT_FALSE(ed.Update(false, 100, "", 100));
}

TEST(EndpointDetector, Disabled) {
    EndpointDetector::Config cfg;
    cfg.enabled = false;
    EndpointDetector ed(cfg);

    // Should never trigger when disabled
    for (int i = 0; i < 20; ++i)
        EXPECT_FALSE(ed.Update(false, 100, "", i * 100));
}

TEST(EndpointDetector, ShortUtterance) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.silence_timeout_ms = 500;
    EndpointDetector ed(cfg);

    // Speech immediately followed by long silence
    EXPECT_FALSE(ed.Update(true, 100, "hi", 100));
    EXPECT_FALSE(ed.Update(false, 100, "hi", 200));
    EXPECT_FALSE(ed.Update(false, 100, "hi", 300));
    EXPECT_FALSE(ed.Update(false, 100, "hi", 400));
    EXPECT_FALSE(ed.Update(false, 100, "hi", 500));
    EXPECT_TRUE(ed.Update(false, 100, "hi", 600));
}

TEST(EndpointDetector, ContinuousSpeech) {
    EndpointDetector::Config cfg;
    cfg.enabled = true;
    cfg.silence_timeout_ms = 500;
    cfg.max_utterance_ms = 5000;
    EndpointDetector ed(cfg);

    // Continuous speech with no silence should not trigger (until max utterance)
    for (int i = 0; i < 40; ++i) {
        bool is_endpoint = ed.Update(true, 100, "speaking...", i * 100);
        if (i * 100 >= 5000)
            EXPECT_TRUE(is_endpoint);
        else
            EXPECT_FALSE(is_endpoint);
    }
}
