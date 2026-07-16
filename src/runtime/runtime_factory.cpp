#include "offline_asr/runtime/runtime_factory.h"
#include "offline_asr/runtime/onnx_runtime.h"

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

    spdlog::error("Unknown runtime backend: {}", backend);
    return nullptr;
}

}  // namespace offline_asr
