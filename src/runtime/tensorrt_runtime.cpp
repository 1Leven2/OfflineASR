#include "offline_asr/runtime/tensorrt_runtime.h"

#include <cstring>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sys/stat.h>

namespace offline_asr {

namespace {

static bool file_mtime(const std::string &path, time_t *out) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0)
    return false;
  *out = st.st_mtime;
  return true;
}

class TRTLogger : public nvinfer1::ILogger {
public:
  static TRTLogger &Instance() {
    static TRTLogger logger;
    return logger;
  }

  void log(Severity severity, const char *msg) noexcept override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      spdlog::error("[TRT] {}", msg);
      break;
    case Severity::kWARNING:
      spdlog::warn("[TRT] {}", msg);
      break;
    case Severity::kINFO:
      spdlog::info("[TRT] {}", msg);
      break;
    default:
      spdlog::debug("[TRT] {}", msg);
      break;
    }
  }

private:
  TRTLogger() = default;
};

} // namespace

TensorRTRuntime::TensorRTRuntime(const Config &cfg) : cfg_(cfg) {
  cudaSetDevice(cfg_.device_id); // 指定当前线程绑定的 GPU 设备
  cudaStreamCreate(
      &h2d_stream_); // 创建异步 CUDA 流，用于内存拷贝和 kernel 执行的异步流水线
}

TensorRTRuntime::~TensorRTRuntime() {
  // Must destroy contexts before engine
  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    for (auto &[tid, ctx] : contexts_) {
      if (ctx.device_input)
        cudaFree(ctx.device_input);
      if (ctx.device_output)
        cudaFree(ctx.device_output);
      if (ctx.stream)
        cudaStreamDestroy(ctx.stream);
      delete ctx.context;
    }
    contexts_.clear();
  }
  delete engine_;
  delete trt_runtime_;
  if (h2d_stream_)
    cudaStreamDestroy(h2d_stream_);
}

// 典型的编译缓存策略：TensorRT Engine
// 构建非常耗时（可能几分钟），所以编译一次后序列化到磁盘，下次直接加载
bool TensorRTRuntime::LoadModel(const std::string &path,
                                const std::string & /*input_name*/,
                                const std::string & /*output_name*/) {
  // path = "models/model.onnx"
  std::string cache_path = cfg_.engine_cache_dir + "/" +
                           path.substr(path.find_last_of("/\\") + 1) + ".trt";
  // "models/model.onnx.trt"

  // Check cache freshness
  time_t onnx_mtime = 0, trt_mtime = 0;
  bool onnx_ok = file_mtime(path, &onnx_mtime);
  bool trt_ok = file_mtime(cache_path, &trt_mtime);

  if (!onnx_ok) {
    spdlog::error("ONNX model not found: {}", path);
    return false;
  }

  // 通过检查模型文件修改时间来检查缓存是否存在且比 ONNX 新
  if (trt_ok && trt_mtime >= onnx_mtime) {
    // Cache hit — deserialize
    // 缓存存在，反序列化加载
    if (LoadEngine(cache_path)) {
      spdlog::info("TensorRT engine loaded from cache: {}", cache_path);
      return true;
    }
    spdlog::warn("Failed to load cached engine, will rebuild");
  }

  // Cache miss or stale — build from ONNX
  // 从 ONNX 重新构建并缓存
  spdlog::info("Building TensorRT engine from {} ...", path);
  if (!BuildEngine(path, cache_path)) {
    spdlog::error("Failed to build TensorRT engine");
    return false;
  }
  return true;
}

bool TensorRTRuntime::BuildEngine(const std::string &onnx_path,
                                  const std::string &cache_path) {
  /*
| API                            | 含义                                     |
| ------------------------------ | -------------------------------------- |
| `createInferBuilder(ILogger&)` | 创建 `IBuilder` 对象，用于构建优化后的引擎 |
| `IBuilder`                     | TensorRT
的**网络构建器**，负责创建网络定义、设置配置、生成引擎 |
*/
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(TRTLogger::Instance()));
  if (!builder)
    return false;

  // kEXPLICIT_BATCH is default in TRT 10.x, pass 0
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
  if (!network)
    return false;

  auto parser =
      std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(
          *network, TRTLogger::Instance())); // 创建 ONNX 解释器
  if (!parser)
    return false;

  // 从 ONNX 文件解释并填充 network 对象，verbosity 控制日志级别
  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    spdlog::error("Failed to parse ONNX model: {}", onnx_path);
    return false;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig()); // 创建构建配置对象
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE,
      cfg_.max_workspace_size); // 设置工作空间显存上限。TensorRT
                                // 在构建时需要进行 layer
                                // fusion、算法选择等，需要临时显存。值越大，越可能找到更快的
                                // kernel

  if (cfg_.use_fp16) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
      spdlog::warn(
          "FP16 requested but not supported by platform, falling back to FP32");
    }
#pragma GCC diagnostic pop
  }

  // Dynamic batch = 1, let time axis be dynamic
  auto *profile = builder->createOptimizationProfile();
  if (network->getInput(0)) {
    auto *input = network->getInput(0);
    auto dims = input->getDimensions(); // [1, T, F]
    int feat_dim = dims.d[2];
    profile->setDimensions(
        input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims3{1, 1, feat_dim});
    profile->setDimensions(
        input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims3{1, 500, feat_dim});
    profile->setDimensions(
        input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims3{1, 3000, feat_dim});
    config->addOptimizationProfile(profile);
  }

  // IHostMemory：主机内存缓冲区，data() 返回指针，size() 返回字节数
  auto plan =
      std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(
          *network, *config)); // 构建引擎并序列化。返回
                               // IHostMemory*，包含引擎的二进制数据
  if (!plan) {
    spdlog::error("Failed to build TensorRT serialized network");
    return false;
  }

  // Write cache
  {
    // 将序列化后的引擎写入 .trt 文件，下次直接加载
    std::ofstream ofs(cache_path, std::ios::binary);
    if (ofs) {
      ofs.write(static_cast<const char *>(plan->data()), plan->size());
      spdlog::info("TensorRT engine cached: {} ({} MiB)", cache_path,
                   plan->size() / (1024 * 1024));
    }
  }

  // Deserialize
  trt_runtime_ = nvinfer1::createInferRuntime(
      TRTLogger::Instance()); // 创建运行时对象，用于反序列化引擎
  if (!trt_runtime_)
    return false;

  engine_ = trt_runtime_->deserializeCudaEngine(
      plan->data(), plan->size()); // 将序列化数据反序列化为 ICudaEngine
  if (!engine_) {
    spdlog::error("Failed to deserialize TensorRT engine");
    return false;
  }

  // Cache vocab size from output dim
  /*
  | API                                | 含义                                |
| ---------------------------------- | --------------------------------- |
| `getIOTensorName(int index)`       | 获取第 N 个输入/输出张量的名称 | |
`getTensorShape(const char* name)` | 获取张量的 shape 信息（`nvinfer1::Dims*`）
| | `nbDims`                           | 维度数                               |
| `d[]`                              | 各维度大小数组 |
*/
  auto output = engine_->getTensorShape(engine_->getIOTensorName(1));
  if (output.nbDims >= 2) {
    output_vocab_dim_ = output.d[output.nbDims - 1];
  }

  return true;
}

bool TensorRTRuntime::LoadEngine(const std::string &engine_path) {
  std::ifstream ifs(engine_path, std::ios::binary | std::ios::ate);
  if (!ifs)
    return false;

  std::streamsize size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!ifs.read(buffer.data(), size))
    return false;

  trt_runtime_ = nvinfer1::createInferRuntime(TRTLogger::Instance());
  if (!trt_runtime_)
    return false;

  engine_ = trt_runtime_->deserializeCudaEngine(buffer.data(), size);
  if (!engine_)
    return false;

  // Cache vocab size
  auto output = engine_->getTensorShape(engine_->getIOTensorName(1));
  if (output.nbDims >= 2) {
    output_vocab_dim_ = output.d[output.nbDims - 1];
  }

  return true;
}

TensorRTRuntime::PerThreadContext &
TensorRTRuntime::GetOrCreateContext(int64_t num_frames, int64_t feat_dim) {
  auto tid = std::this_thread::get_id();

  {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto it = contexts_.find(tid);
    if (it != contexts_.end()) {
      // Resize device buffers if needed
      size_t input_bytes = num_frames * feat_dim * sizeof(float);
      if (input_bytes > it->second.input_capacity) {
        cudaFree(it->second.device_input);
        cudaMalloc(&it->second.device_input, input_bytes);
        it->second.input_capacity = input_bytes;
      }
      return it->second;
    }
  }

  // Create new context for this thread
  PerThreadContext ctx;
  ctx.context = engine_->createExecutionContext();
  cudaStreamCreate(&ctx.stream);

  size_t input_bytes = num_frames * feat_dim * sizeof(float);
  cudaMalloc(&ctx.device_input, input_bytes);
  ctx.input_capacity = input_bytes;

  // Allocate generous output buffer, will be resized as needed
  // 输入缓冲区按当前音频长度分配，如果后续音频更长则重新 cudaMalloc
  int64_t output_frames = num_frames / 4;
  size_t output_bytes = output_frames * output_vocab_dim_ * sizeof(float);
  cudaMalloc(&ctx.device_output, output_bytes);
  ctx.output_capacity = output_bytes;

  std::lock_guard<std::mutex> lock(contexts_mutex_);
  auto [it, _] = contexts_.emplace(tid, std::move(ctx));
  return it->second;
}

std::vector<float> TensorRTRuntime::Forward(const float *input,
                                            int64_t num_frames,
                                            int64_t feat_dim) {
  if (!engine_) {
    spdlog::error("TensorRT engine not loaded");
    return {};
  }

  auto &ctx = GetOrCreateContext(num_frames, feat_dim);

  size_t input_bytes = num_frames * feat_dim * sizeof(float);

  // H2D copy
  cudaMemcpyAsync(ctx.device_input, input, input_bytes, cudaMemcpyHostToDevice,
                  ctx.stream); // 异步内存拷贝，不会阻塞 CPU，关联到指定 CUDA
                               // 流，实现流水线并行

  // Set input tensor address + shape
  // 这是 TensorRT 8.6+ 的 Explicit Batch + Dynamic Shape 用法。输入 shape
  // 可以每次推理都不同（只要不超过 kMAX）
  nvinfer1::Dims3 input_shape{1, static_cast<int32_t>(num_frames),
                              static_cast<int32_t>(feat_dim)};
  ctx.context->setInputShape(
      engine_->getIOTensorName(0),
      input_shape); // 设置动态输入的实际 shape。必须在 enqueueV3 之前调用
  ctx.context->setTensorAddress(engine_->getIOTensorName(0),
                                ctx.device_input); // 绑定张量的显存地址

  // Allocate/resize output buffer
  int64_t output_frames = num_frames / 4;
  size_t output_bytes = output_frames * output_vocab_dim_ * sizeof(float);
  if (output_bytes > ctx.output_capacity) {
    cudaFree(ctx.device_output);
    cudaMalloc(&ctx.device_output, output_bytes);
    ctx.output_capacity = output_bytes;
  }
  ctx.context->setTensorAddress(engine_->getIOTensorName(1), ctx.device_output);

  // Enqueue
  if (!ctx.context->enqueueV3(
          ctx.stream)) { // TensorRT 8.6+ 的异步执行
                         // API。将所有设置的输入/输出在指定流上启动推理
    spdlog::error("TensorRT enqueueV3 failed");
    return {};
  }
  cudaStreamSynchronize(ctx.stream); // 阻塞 CPU，等待流上所有操作完成

  // D2H copy
  // 同步将结果从显存拷贝回主机内存的 std::vector
  size_t total_elements = output_frames * output_vocab_dim_;
  std::vector<float> result(total_elements);
  cudaMemcpy(result.data(), ctx.device_output, output_bytes,
             cudaMemcpyDeviceToHost);

  return result;
}

} // namespace offline_asr
