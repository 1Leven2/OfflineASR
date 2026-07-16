#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "offline_asr/runtime/runtime.h"

namespace offline_asr {

/**
 * ONNX Runtime 推理后端
 *
 * 通过 ONNX Runtime C++ API 加载 .onnx 模型并执行推理。
 * 支持动态 shape (变长音频) 和多线程配置。
 */
class OnnxRuntime : public Runtime {
public:
    /**
     * @param num_threads 推理线程数 (0 = 自动)
     * @param use_gpu     是否使用 GPU (当前不支持)
     */
    explicit OnnxRuntime(int num_threads = 0, bool use_gpu = false);

    bool LoadModel(const std::string& path,
                   const std::string& input_name,
                   const std::string& output_name) override;

    std::vector<float> Forward(const float* input,
                               int64_t num_frames,
                               int64_t feat_dim) override;

    int64_t NumOutputFrames(int64_t num_input_frames) const override;
    int64_t VocabSize() const override;

    std::string BackendName() const override { return "ONNX"; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Cached per-inference objects (avoid per-call allocation)
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::RunOptions run_opts_;

    int64_t output_time_dim_ = -1;
    int64_t output_vocab_dim_ = -1;

    int num_threads_;
    bool use_gpu_;
};

}  // namespace offline_asr
