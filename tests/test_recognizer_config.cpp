#include <gtest/gtest.h>
#include <fstream>

#include "offline_asr/recognizer/recognizer_config.h"

using namespace offline_asr;

TEST(RecognizerConfig, Defaults) {
    RecognizerConfig cfg;
    EXPECT_EQ(cfg.sample_rate, 16000);
    EXPECT_EQ(cfg.num_mel_bins, 80);
    EXPECT_EQ(cfg.beam_size, 10);
    EXPECT_EQ(cfg.runtime_backend, "onnx");
}

TEST(RecognizerConfig, FromYaml) {
    RecognizerConfig cfg = RecognizerConfig::FromYaml(TEST_DATA_DIR "/configs/default.yaml");
    EXPECT_EQ(cfg.sample_rate, 16000);
    EXPECT_EQ(cfg.num_mel_bins, 80);
    EXPECT_EQ(cfg.beam_size, 10);
    EXPECT_EQ(cfg.runtime_backend, "onnx");
    EXPECT_FALSE(cfg.model_path.empty());
}

TEST(RecognizerConfig, FromYamlNonExistent) {
    RecognizerConfig cfg = RecognizerConfig::FromYaml("nonexistent.yaml");
    // Should not crash, fall back to defaults
    EXPECT_EQ(cfg.sample_rate, 16000);
}
