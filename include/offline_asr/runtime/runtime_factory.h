#pragma once

#include <memory>
#include <string>

#include "offline_asr/runtime/runtime.h"

namespace offline_asr {

struct RecognizerConfig;

/** 根据 backend 名称创建对应的 Runtime 实例 */
class RuntimeFactory {
public:
    /**
     * @param backend     后端名称 ("onnx", "tensorrt", ...)
     * @param num_threads 推理线程数
     * @return Runtime 实例指针
     */
    static std::unique_ptr<Runtime> Create(const std::string& backend,
                                           int num_threads = 0);

    /**
     * 带完整配置的工厂方法（用于需要额外参数的 backend，如 TensorRT）
     * @param backend 后端名称
     * @param cfg     识别器配置（包含 tensorrt 子配置）
     * @return Runtime 实例指针
     */
    static std::unique_ptr<Runtime> Create(const std::string& backend,
                                           const RecognizerConfig& cfg);
};

}  // namespace offline_asr
