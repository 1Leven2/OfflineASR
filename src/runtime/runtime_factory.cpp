#include "offline_asr/runtime/runtime_factory.h"
#include "offline_asr/recognizer/recognizer_config.h"
#include "offline_asr/runtime/onnx_runtime.h"

#ifdef ENABLE_TENSORRT
#include "offline_asr/runtime/tensorrt_runtime.h"
#endif

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>

namespace offline_asr {

std::unique_ptr<Runtime> RuntimeFactory::Create(const std::string& backend,
                                                 int num_threads) {
    std::string backend_lower = backend;
    std::transform(backend_lower.begin(), backend_lower.end(),
                   backend_lower.begin(), ::tolower);

    if (backend_lower == "onnx") {
        return std::make_unique<OnnxRuntime>(num_threads);
    }

#ifdef ENABLE_TENSORRT
    if (backend_lower == "tensorrt") {
        TensorRTRuntime::Config trt_cfg;
        return std::make_unique<TensorRTRuntime>(trt_cfg);
    }
#endif

    spdlog::error("Unknown runtime backend: {}", backend);
    return nullptr;
}

std::unique_ptr<Runtime> RuntimeFactory::Create(const std::string& backend,
                                                 const RecognizerConfig& cfg) {
    std::string backend_lower = backend;
    std::transform(backend_lower.begin(), backend_lower.end(),
                   backend_lower.begin(), ::tolower);

    if (backend_lower == "onnx") {
        return std::make_unique<OnnxRuntime>(cfg.num_threads);
    }

#ifdef ENABLE_TENSORRT
    if (backend_lower == "tensorrt") {
        TensorRTRuntime::Config trt_cfg;
        trt_cfg.device_id = cfg.tensorrt.device_id;
        trt_cfg.use_fp16 = cfg.tensorrt.use_fp16;
        trt_cfg.max_workspace_size = cfg.tensorrt.max_workspace_size;
        trt_cfg.engine_cache_dir = cfg.tensorrt.engine_cache_dir;
        return std::make_unique<TensorRTRuntime>(trt_cfg);
    }
#endif

    spdlog::error("Unknown runtime backend: {}", backend);
    return nullptr;
}

}  // namespace offline_asr
