#ifdef ENABLE_TENSORRT

#include <fstream>
#include <gtest/gtest.h>

#include "offline_asr/runtime/tensorrt_runtime.h"

using namespace offline_asr;

namespace {
const std::string kModelPath = TEST_DATA_DIR "/models/model.onnx";
bool model_exists() {
    std::ifstream f(kModelPath);
    return f.good();
}
}  // namespace

TEST(TensorRTRuntime, Construction) {
    TensorRTRuntime::Config cfg;
    cfg.device_id = 0;
    TensorRTRuntime runtime(cfg);
    EXPECT_EQ(runtime.BackendName(), "TensorRT");
}

TEST(TensorRTRuntime, LoadModel) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;

    TensorRTRuntime::Config cfg;
    cfg.device_id = 0;
    cfg.engine_cache_dir = "/tmp";
    TensorRTRuntime runtime(cfg);

    bool ok = runtime.LoadModel(kModelPath, "input", "output");
    EXPECT_TRUE(ok);
    EXPECT_GT(runtime.VocabSize(), 0);
}

TEST(TensorRTRuntime, Forward) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;

    TensorRTRuntime::Config cfg;
    cfg.device_id = 0;
    cfg.engine_cache_dir = "/tmp";
    TensorRTRuntime runtime(cfg);

    ASSERT_TRUE(runtime.LoadModel(kModelPath, "input", "output"));

    int vocab_size = static_cast<int>(runtime.VocabSize());
    ASSERT_GT(vocab_size, 0);

    std::vector<float> input(100 * 80, 0.0f);
    auto output = runtime.Forward(input.data(), 100, 80);

    int expected_frames = static_cast<int>(runtime.NumOutputFrames(100));
    EXPECT_EQ(output.size(), expected_frames * vocab_size);
}

TEST(TensorRTRuntime, NumOutputFrames) {
    TensorRTRuntime::Config cfg;
    TensorRTRuntime runtime(cfg);
    EXPECT_EQ(runtime.NumOutputFrames(400), 100);
    EXPECT_EQ(runtime.NumOutputFrames(100), 25);
}

TEST(TensorRTRuntime, EngineCache) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;

    TensorRTRuntime::Config cfg;
    cfg.device_id = 0;
    cfg.engine_cache_dir = "/tmp";
    TensorRTRuntime runtime(cfg);

    ASSERT_TRUE(runtime.LoadModel(kModelPath, "input", "output"));

    TensorRTRuntime::Config cfg2;
    cfg2.device_id = 0;
    cfg2.engine_cache_dir = "/tmp";
    TensorRTRuntime runtime2(cfg2);
    ASSERT_TRUE(runtime2.LoadModel(kModelPath, "input", "output"));
}

TEST(TensorRTRuntime, MultipleContexts) {
    if (!model_exists()) GTEST_SKIP() << "Model not found: " << kModelPath;

    TensorRTRuntime::Config cfg;
    cfg.device_id = 0;
    cfg.engine_cache_dir = "/tmp";
    TensorRTRuntime runtime(cfg);

    ASSERT_TRUE(runtime.LoadModel(kModelPath, "input", "output"));

    std::vector<float> input(200 * 80, 0.1f);
    auto out1 = runtime.Forward(input.data(), 200, 80);
    auto out2 = runtime.Forward(input.data(), 200, 80);

    EXPECT_EQ(out1.size(), out2.size());
}

#endif  // ENABLE_TENSORRT
