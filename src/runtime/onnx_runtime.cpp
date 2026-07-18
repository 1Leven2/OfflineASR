#include "offline_asr/runtime/onnx_runtime.h"

#include <spdlog/spdlog.h>

namespace offline_asr {

OnnxRuntime::OnnxRuntime(int num_threads, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "offline_asr"), num_threads_(num_threads),
      use_gpu_(use_gpu), memory_info_(Ort::MemoryInfo::CreateCpu(
                             OrtArenaAllocator, OrtMemTypeDefault)) {
  session_opts_.SetIntraOpNumThreads(num_threads_ > 0 ? num_threads_ : 1);
  session_opts_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_opts_.EnableCpuMemArena();
}

bool OnnxRuntime::LoadModel(const std::string &path,
                            const std::string &input_name,
                            const std::string &output_name) {
  try {
    session_ =
        std::make_unique<Ort::Session>(env_, path.c_str(), session_opts_);

    input_name_ = input_name;
    output_name_ = output_name;
    input_names_ = {input_name_.c_str()};
    output_names_ = {output_name_.c_str()};

    // Cache static output dimension (vocab size)
    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_shape = output_tensor_info.GetShape();
    if (output_shape.size() >= 2 && output_shape.back() > 0) {
      output_vocab_dim_ = output_shape.back();
    }

    spdlog::info("ONNX model loaded: {} (backend: ONNX, threads: {})", path,
                 num_threads_);
    return true;
  } catch (const Ort::Exception &e) {
    spdlog::error("Failed to load ONNX model: {} ({})", path, e.what());
    return false;
  }
}

std::vector<float> OnnxRuntime::Forward(const float *input, int64_t num_frames,
                                        int64_t feat_dim) {
  if (!session_) {
    spdlog::error("Model not loaded");
    return {};
  }

  try {
    std::vector<int64_t> input_shape = {1, num_frames, feat_dim};
    size_t total_elements = num_frames * feat_dim;

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float *>(input), total_elements,
        input_shape.data(), input_shape.size());

    auto output_tensors =
        session_->Run(run_opts_, input_names_.data(), &input_tensor, 1,
                      output_names_.data(), 1);

    auto &output_tensor = output_tensors.front();
    auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    auto output_count = output_info.GetElementCount();

    // Cache vocab size from static output dimension
    if (output_shape.size() >= 2 && output_shape.back() > 0) {
      output_vocab_dim_ = output_shape.back();
    }

    std::vector<float> result(output_count);
    std::memcpy(result.data(), output_tensor.GetTensorData<float>(),
                output_count * sizeof(float));

    return result;
  } catch (const Ort::Exception &e) {
    spdlog::error("ONNX inference failed: {}", e.what());
    return {};
  }
}

int64_t OnnxRuntime::NumOutputFrames(int64_t num_input_frames) const { return num_input_frames / 4; }

int64_t OnnxRuntime::VocabSize() const { return output_vocab_dim_; }

} // namespace offline_asr
