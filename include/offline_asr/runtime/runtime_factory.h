#pragma once

#include <memory>
#include <string>

#include "offline_asr/runtime/runtime.h"

namespace offline_asr {

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
};

}  // namespace offline_asr
