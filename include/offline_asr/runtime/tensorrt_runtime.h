#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include "offline_asr/runtime/runtime.h"

namespace offline_asr {

class TensorRTRuntime : public Runtime {
public:
    struct Config {
        int device_id = 0;
        bool use_fp16 = false;
        size_t max_workspace_size = 1ULL << 30;  // 1 GiB
        std::string engine_cache_dir = "models";
    };

    explicit TensorRTRuntime(const Config& cfg);
    ~TensorRTRuntime() override;

    bool LoadModel(const std::string& path,
                   const std::string& input_name,
                   const std::string& output_name) override;

    std::vector<float> Forward(const float* input,
                               int64_t num_frames,
                               int64_t feat_dim) override;

    int64_t NumOutputFrames(int64_t num_input_frames) const override {
        return num_input_frames / 4;
    }
    int64_t VocabSize() const override { return output_vocab_dim_; }

    std::string BackendName() const override { return "TensorRT"; }

private:
    struct PerThreadContext {
        nvinfer1::IExecutionContext* context = nullptr;
        cudaStream_t stream = nullptr;
        void *device_input = nullptr;
        void *device_output = nullptr;
        size_t input_capacity = 0;   // bytes
        size_t output_capacity = 0;  // bytes
    };

    PerThreadContext& GetOrCreateContext(int64_t num_frames, int64_t feat_dim);

    // Build TensorRT engine from ONNX, serialize to cache
    bool BuildEngine(const std::string& onnx_path, const std::string& cache_path);
    // Load serialized engine
    bool LoadEngine(const std::string& engine_path);

    Config cfg_;
    nvinfer1::IRuntime* trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    int64_t output_vocab_dim_ = -1;

    // Per-thread execution contexts (engine shared, context not thread-safe)
    std::unordered_map<std::thread::id, PerThreadContext> contexts_;
    mutable std::mutex contexts_mutex_;

    cudaStream_t h2d_stream_ = nullptr;  // dedicated H2D stream
};

}  // namespace offline_asr
