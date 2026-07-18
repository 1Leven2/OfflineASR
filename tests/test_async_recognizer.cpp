#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

#include "offline_asr/recognizer/async_recognizer.h"

using namespace offline_asr;

namespace {
const std::string kModelPath = TEST_DATA_DIR "/models/model.onnx";
const std::string kTokensPath = TEST_DATA_DIR "/models/tokens.txt";
bool model_exists() {
    std::ifstream f(kModelPath);
    return f.good();
}
}  // namespace

TEST(AsyncRecognizer, Construction) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 2);
    SUCCEED();
}

TEST(AsyncRecognizer, PendingIsZeroInitially) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 2);
    EXPECT_EQ(recognizer.Pending(), 0u);
}

TEST(AsyncRecognizer, WaitAllOnIdlePool) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 2);
    recognizer.WaitAll();
    SUCCEED();
}

TEST(AsyncRecognizer, RecognizeAsyncSingle) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 2);

    std::vector<float> audio(16000);
    for (int i = 0; i < 16000; ++i) {
        audio[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }

    auto future = recognizer.RecognizeAsync(audio.data(), audio.size());
    auto result = future.get();
    EXPECT_GE(result.audio_duration_ms, 0.0);
}

TEST(AsyncRecognizer, RecognizeAsyncMultiple) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 4);

    std::vector<float> audio(16000);
    for (int i = 0; i < 16000; ++i) {
        audio[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }

    std::vector<std::future<RecognitionResult>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(recognizer.RecognizeAsync(audio.data(), audio.size()));
    }

    for (auto& f : futures) {
        auto result = f.get();
        EXPECT_GE(result.audio_duration_ms, 0.0);
    }
}

TEST(AsyncRecognizer, WaitAllDrainsPending) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;
    RecognizerConfig cfg;
    cfg.model_path = kModelPath;
    cfg.tokens_path = kTokensPath;

    AsyncRecognizer recognizer(cfg, 2);

    std::vector<float> audio(16000);
    for (int i = 0; i < 16000; ++i) {
        audio[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f) * 0.5f;
    }

    for (int i = 0; i < 4; ++i) {
        recognizer.RecognizeAsync(audio.data(), audio.size());
    }
    EXPECT_GT(recognizer.Pending(), 0u);

    recognizer.WaitAll();
    EXPECT_EQ(recognizer.Pending(), 0u);
}
